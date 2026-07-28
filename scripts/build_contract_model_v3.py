"""
Contract model v3 -- the current recommended contract-prediction pipeline,
superseding v2 (build_contract_model.py, still describable in
docs/contract_model.md's history but no longer built or exported).

Two changes from v2, both validated directly against the free-agent eval
harness (scripts/evaluate_free_agents.py) rather than assumed:

1. Trained only on real free-agent signings (data/free_agents/), not the
   full historical player-season pool. v2's target (a player's
   contemporaneous SALARY_PCT_OF_CAP) is contaminated by contract-lag noise
   -- most rows in the full pool are players mid-way through a multi-year
   deal reflecting *past* performance (see contract_model.md's Westbrook/
   Beal/Simmons examples). Every row here is instead a fresh market
   valuation event. Despite the much smaller training population (~580-670
   rows/fold here vs. ~1,900 veterans/fold in v2), this measurably reduced
   error: free-agent-eval MAE dropped from v2's 3.88pp to 3.42pp.

2. A minutes-reliability gate (see apply_minutes_gate.py): predictions for
   players under 2,400 trailing-3-year minutes are shrunk toward 0 by
   sqrt(total_min / 2400), since these are cases the model has no real
   signal to work from (deep-bench/two-way players whose vector looks
   deceptively "average" in small samples). Dropped MAE further to 3.31pp.

Same feature set as v2 (12-category vector + EXPERIENCE_YRS + USG_PCT +
ALL_STAR_SELECTION + ALL_NBA_SELECTION), same XGBoost hyperparameters, same
leave-one-season-out validation discipline -- just a different, narrower
training population, plus the gate as a final step.

Usage:
    python scripts/build_contract_model_v3.py
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from apply_minutes_gate import three_year_total_minutes
from build_team_vectors import STANDARD_SEASONS

DATA_DIR = Path("data")
VECTOR_DIR = DATA_DIR / "player_vectors_json"
FEATURES_PATH = DATA_DIR / "salary_prediction" / "player_season_features.csv"
FREE_AGENTS_DIR = DATA_DIR / "free_agents"
SALARIES_DIR = DATA_DIR / "salaries"
OUT_PATH = DATA_DIR / "salary_prediction" / "contract_model_v3_predictions.csv"

CATEGORIES = [
    "ISOLATION", "PNR_HANDLER", "PNR_ROLLMAN", "POST_UP", "SPOT_UP",
    "HANDOFF", "CUT", "OFF_SCREEN", "PUT_BACKS",
    "RIM_DEFENSE", "PERIMETER_DEFENSE", "REBOUNDING",
]
AUX_FEATURES = ["EXPERIENCE_YRS", "USG_PCT", "ALL_STAR_SELECTION", "ALL_NBA_SELECTION"]
FEATURE_COLUMNS = CATEGORIES + AUX_FEATURES
XGB_PARAMS = dict(max_depth=3, reg_alpha=0.1, reg_lambda=1.0, n_estimators=200, random_state=0)
MINUTES_THRESHOLD = 2400

FREE_AGENT_FILE_RE = re.compile(r"^(\d{2})-(\d{2})\.txt$")
ROW_RE = re.compile(r"^([A-Za-z\.’'\-\ ]+?)\t([A-Z/]+)\t(\d+)\t\$([\d,]+)\t\$([\d,]+)")


def normalize_name(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stripped = re.sub(r"\b(jr\.?|sr\.?|iii|ii|iv)\b", "", stripped.lower())
    return re.sub(r"[^a-z ]", "", stripped).strip()


def season_pair(filename_stem: str) -> tuple[str, str]:
    start, end = filename_stem.split("-")
    return f"20{start}-{end}", f"20{int(start) + 1:02d}-{int(end) + 1:02d}"


def parse_freeagents(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if m:
            names.append(m.group(1).strip())
    return names


def load_vector_pool() -> pd.DataFrame:
    rows = []
    for season in STANDARD_SEASONS:
        data = json.loads((VECTOR_DIR / f"player_vectors_{season}.json").read_text(encoding="utf-8"))
        for pid, p in data.items():
            row = {"PLAYER_ID": int(pid), "SEASON": season, "PLAYER_NAME": p["PLAYER_NAME"],
                   "NORM": normalize_name(p["PLAYER_NAME"])}
            row.update({cat: p["VECTOR"][cat] for cat in CATEGORIES})
            rows.append(row)
    return pd.DataFrame(rows)


def build_freeagent_rows(vector_pool: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fa_path in sorted(FREE_AGENTS_DIR.glob("*.txt")):
        m = FREE_AGENT_FILE_RE.match(fa_path.name)
        if not m:
            continue
        prev_season, target_season = season_pair(fa_path.stem)

        sal_path = SALARIES_DIR / f"salaries_{target_season}.csv"
        if not sal_path.exists():
            continue
        sal_df = pd.read_csv(sal_path)
        sal_df["NORM"] = sal_df["PLAYER_NAME"].apply(normalize_name)
        sal_by_name = sal_df.set_index("NORM")["SALARY_PCT_OF_CAP"].to_dict()

        vec_prev = vector_pool[vector_pool["SEASON"] == prev_season].set_index("NORM")
        feat_prev = features[features["SEASON"] == prev_season].copy()
        feat_prev["NORM"] = feat_prev["PLAYER_NAME"].apply(normalize_name)
        feat_by_name = feat_prev.set_index("NORM")

        for name in parse_freeagents(fa_path):
            key = normalize_name(name)
            if key not in vec_prev.index or key not in sal_by_name or key not in feat_by_name.index:
                continue
            vrow = vec_prev.loc[key]
            if isinstance(vrow, pd.DataFrame):
                vrow = vrow.iloc[0]
            frow = feat_by_name.loc[key]
            if isinstance(frow, pd.DataFrame):
                frow = frow.iloc[0]
            if pd.isna(frow.get("USG_PCT")):
                continue

            row = {
                "PLAYER_ID": vrow["PLAYER_ID"], "PLAYER_NAME": name,
                "PREV_SEASON": prev_season, "TARGET_SEASON": target_season,
                "ACTUAL_SALARY_PCT_OF_CAP": sal_by_name[key],
            }
            row.update({cat: vrow[cat] for cat in CATEGORIES})
            row["EXPERIENCE_YRS"] = frow["EXPERIENCE_YRS"]
            row["USG_PCT"] = frow["USG_PCT"]
            row["ALL_STAR_SELECTION"] = int(frow["ALL_STAR_SELECTION"])
            row["ALL_NBA_SELECTION"] = int(frow["ALL_NBA_SELECTION"])
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    vector_pool = load_vector_pool()
    features = pd.read_csv(FEATURES_PATH)
    fa_rows = build_freeagent_rows(vector_pool, features)

    target_seasons = sorted(fa_rows["TARGET_SEASON"].unique())
    print(f"Built {len(fa_rows)} free-agent training rows across {len(target_seasons)} target seasons.")

    all_preds = []
    per_year = []
    for held_out in target_seasons:
        train = fa_rows[fa_rows["TARGET_SEASON"] != held_out]
        test = fa_rows[fa_rows["TARGET_SEASON"] == held_out]
        if len(train) < 20 or test.empty:
            print(f"  Skipping {held_out}: too few training rows ({len(train)}).")
            continue

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(train[FEATURE_COLUMNS], train["ACTUAL_SALARY_PCT_OF_CAP"])
        preds = model.predict(test[FEATURE_COLUMNS])

        out = test[["PLAYER_ID", "PLAYER_NAME", "PREV_SEASON", "TARGET_SEASON", "ACTUAL_SALARY_PCT_OF_CAP"]].copy()
        out["PREDICTED_SALARY_PCT_OF_CAP"] = preds
        all_preds.append(out)

    result = pd.concat(all_preds, ignore_index=True)

    # minutes-reliability gate: shrink predictions toward 0 for players below
    # MINUTES_THRESHOLD trailing-3yr minutes, computed off PREV_SEASON (the
    # season whose vector/usage actually feeds the prediction)
    total_min = three_year_total_minutes(result["PLAYER_ID"], result["PREV_SEASON"])
    shrink = np.sqrt((total_min / MINUTES_THRESHOLD).clip(max=1.0, min=0.0))
    n_gated = int((total_min < MINUTES_THRESHOLD).sum())
    print(f"\n{n_gated} of {len(result)} rows below {MINUTES_THRESHOLD} trailing-3yr minutes -- "
          f"shrinking toward 0 by sqrt(total_min/{MINUTES_THRESHOLD}).")
    result["PREDICTED_SALARY_PCT_OF_CAP"] = result["PREDICTED_SALARY_PCT_OF_CAP"] * shrink
    result["RESIDUAL_PP"] = result["ACTUAL_SALARY_PCT_OF_CAP"] - result["PREDICTED_SALARY_PCT_OF_CAP"]

    per_year_rows = []
    for season, g in result.groupby("TARGET_SEASON"):
        r = g["RESIDUAL_PP"].to_numpy()
        corr = np.corrcoef(g["PREDICTED_SALARY_PCT_OF_CAP"], g["ACTUAL_SALARY_PCT_OF_CAP"])[0, 1] if len(g) > 1 else float("nan")
        per_year_rows.append({"TARGET_SEASON": season, "N": len(g), "MAE": np.abs(r).mean(),
                               "RMSE": np.sqrt((r**2).mean()), "CORR": corr})
    print("\nPer-season (leave-one-season-out on the free-agent list, post-gate):")
    print(pd.DataFrame(per_year_rows).round(3).to_string(index=False))

    resid = result["RESIDUAL_PP"].to_numpy()
    print(f"\nPooled: N={len(result)}  MAE={np.abs(resid).mean():.2f}pp  "
          f"RMSE={np.sqrt((resid**2).mean()):.2f}pp  "
          f"corr={np.corrcoef(result['PREDICTED_SALARY_PCT_OF_CAP'], result['ACTUAL_SALARY_PCT_OF_CAP'])[0,1]:.3f}")

    result.sort_values(["TARGET_SEASON", "PREDICTED_SALARY_PCT_OF_CAP"], ascending=[True, False]).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
