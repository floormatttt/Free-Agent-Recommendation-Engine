"""
Evaluates two contract-prediction pipelines (docs/similarity_model.md's
kernel regression and docs/contract_model.md's v3 XGBoost model, trained
directly on free-agent signings -- see build_contract_model_v3.py) against
every real free-agent class on file in data/free_agents/, not just one
offseason.

Each file is named <season_just_played_short>.txt -- the season these free
agents just finished playing, NOT the season their new contract covers
(e.g. 25-26.txt's free agents just played 2025-26 and signed contracts
covering 2026-27). See season_pair()'s docstring for how this was verified
empirically. Predictions are drawn from that named season's player
vectors/predictions -- the data that would have actually been available at
signing time -- and compared against the real deal in the *following*
season.

Actual pay is the target season's real salary (data/salaries/
salaries_<target_season>.csv, i.e. actual first-year cap hit) where known,
falling back to AAV (from the free-agent file itself) otherwise -- these
differ for back/front-loaded deals.

Usage:
    python scripts/evaluate_free_agents.py
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
FREE_AGENTS_DIR = DATA_DIR / "free_agents"
SIMILARITY_DIR = DATA_DIR / "similarity"
SUPERVISED_PATH = DATA_DIR / "salary_prediction" / "contract_model_v3_predictions.csv"
SALARIES_DIR = DATA_DIR / "salaries"
OUT_CSV = DATA_DIR / "salary_prediction" / "free_agent_eval_multi_year.csv"

FREE_AGENT_FILE_RE = re.compile(r"^(\d{2})-(\d{2})\.txt$")
ROW_RE = re.compile(r"^([A-Za-z\.’'\-\ ]+?)\t([A-Z/]+)\t(\d+)\t\$([\d,]+)\t\$([\d,]+)")


def normalize_name(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stripped = re.sub(r"\b(jr\.?|sr\.?|iii|ii|iv)\b", "", stripped.lower())
    return re.sub(r"[^a-z ]", "", stripped).strip()


def season_pair(filename_stem: str) -> tuple[str, str]:
    """'25-26' -> (previous_season, target_season) = ('2025-26', '2026-27').

    The filename is the season these free agents *just finished playing*
    (their vectors are predicted from this season), not the season their
    new contract covers. Verified empirically: for every file on hand,
    matching each player's AAV against the following season's actual salary
    gives an 82-100% close-match rate, vs. only 21-38% when matching
    against the filename's own season -- e.g. Hassan Whiteside's row in
    19-20.txt ($2,320,044 AAV) matches his real 2020-21 Kings salary
    exactly, not anything in 2019-20."""
    start, end = filename_stem.split("-")
    prev_season = f"20{start}-{end}"
    target_season = f"20{int(start) + 1:02d}-{int(end) + 1:02d}"
    return prev_season, target_season


def parse_freeagents(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    players = []
    for line in lines:
        m = ROW_RE.match(line)
        if not m:
            continue
        name, pos, yrs, total, aav = m.groups()
        players.append({
            "PLAYER_NAME": name.strip(),
            "POS": pos,
            "YRS": int(yrs),
            "TOTAL": int(total.replace(",", "")),
            "AAV": int(aav.replace(",", "")),
        })
    return players


def load_available_seasons() -> set[str]:
    return {p.stem.replace("player_vectors_", "") for p in (DATA_DIR / "player_vectors_json").glob("player_vectors_*.json")}


def main():
    available_vector_seasons = load_available_seasons()
    sup_df = pd.read_csv(SUPERVISED_PATH)
    sup_df["NORM"] = sup_df["PLAYER_NAME"].apply(normalize_name)

    all_rows = []
    per_year_summary = []

    for fa_path in sorted(FREE_AGENTS_DIR.glob("*.txt")):
        m = FREE_AGENT_FILE_RE.match(fa_path.name)
        if not m:
            continue
        prev_season, target_season = season_pair(fa_path.stem)

        if prev_season not in available_vector_seasons:
            print(f"Skipping {fa_path.name} (target {target_season}): no player vectors for prior season {prev_season}.")
            continue

        sim_path = SIMILARITY_DIR / f"similarity_{prev_season}.json"
        similarity = json.loads(sim_path.read_text(encoding="utf-8")) if sim_path.exists() else {}
        sim_by_name = {normalize_name(e["PLAYER_NAME"]): e for e in similarity.values()}

        sup_prev = sup_df[sup_df["PREV_SEASON"] == prev_season]
        sup_by_name = {row["NORM"]: row for _, row in sup_prev.iterrows()}

        sal_path = SALARIES_DIR / f"salaries_{target_season}.csv"
        sal_df = pd.read_csv(sal_path) if sal_path.exists() else pd.DataFrame(columns=["PLAYER_NAME", "SALARY_PCT_OF_CAP"])
        sal_df["NORM"] = sal_df["PLAYER_NAME"].apply(normalize_name)
        sal_by_name = {row["NORM"]: row for _, row in sal_df.iterrows()}

        free_agents = parse_freeagents(fa_path)

        year_matched_sim, year_matched_sup, year_matched_actual = 0, 0, 0
        for fa in free_agents:
            key = normalize_name(fa["PLAYER_NAME"])
            actual_row = sal_by_name.get(key)
            actual_pct = float(actual_row["SALARY_PCT_OF_CAP"]) if actual_row is not None else None

            sim_entry = sim_by_name.get(key)
            sim_pred = sim_entry.get("PREDICTED_SALARY_PCT_OF_CAP") if sim_entry else None

            sup_row = sup_by_name.get(key)
            sup_pred = float(sup_row["PREDICTED_SALARY_PCT_OF_CAP"]) if sup_row is not None else None

            if actual_pct is not None:
                year_matched_actual += 1
                if sim_pred is not None:
                    year_matched_sim += 1
                if sup_pred is not None:
                    year_matched_sup += 1

            all_rows.append({
                "TARGET_SEASON": target_season,
                "PREV_SEASON": prev_season,
                "PLAYER_NAME": fa["PLAYER_NAME"],
                "POS": fa["POS"],
                "AAV": fa["AAV"],
                "ACTUAL_SALARY_PCT_OF_CAP": actual_pct,
                "SIMILARITY_PREDICTED_PCT_OF_CAP": sim_pred,
                "SUPERVISED_PREDICTED_PCT_OF_CAP": sup_pred,
            })

        print(f"{fa_path.name} (predict from {prev_season} -> actual {target_season}): "
              f"{len(free_agents)} free agents, {year_matched_actual} with known actual salary, "
              f"{year_matched_sim} similarity preds, {year_matched_sup} supervised preds")
        per_year_summary.append(target_season)

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(df)} rows across {len(per_year_summary)} seasons)")

    def summarize(label: str, pred_col: str, by_year: bool = False):
        known = df.dropna(subset=["ACTUAL_SALARY_PCT_OF_CAP", pred_col])
        if known.empty:
            print(f"\n[{label}] no matched rows")
            return
        resid = known["ACTUAL_SALARY_PCT_OF_CAP"] - known[pred_col]
        mae = resid.abs().mean()
        rmse = (resid ** 2).mean() ** 0.5
        bias = resid.mean()
        corr = known["ACTUAL_SALARY_PCT_OF_CAP"].corr(known[pred_col])
        print(f"\n[{label}] pooled across all years -- n={len(known)}  MAE={mae:.2f}pp  "
              f"RMSE={rmse:.2f}pp  bias={bias:+.2f}pp  corr={corr:.3f}")
        if by_year:
            for season, g in known.groupby("TARGET_SEASON"):
                r = g["ACTUAL_SALARY_PCT_OF_CAP"] - g[pred_col]
                print(f"    {season}: n={len(g):3d}  MAE={r.abs().mean():.2f}pp  "
                      f"RMSE={(r**2).mean()**0.5:.2f}pp  corr={g['ACTUAL_SALARY_PCT_OF_CAP'].corr(g[pred_col]):.3f}")

    summarize("Similarity/kernel-regression model", "SIMILARITY_PREDICTED_PCT_OF_CAP", by_year=True)
    summarize("Contract model v3 (free-agent-trained XGBoost + minutes gate)", "SUPERVISED_PREDICTED_PCT_OF_CAP", by_year=True)


if __name__ == "__main__":
    main()
