"""
Did the contract model call free-agent value better than the GMs who
actually signed the checks?

You can't score that by comparing PREDICTED_SALARY_PCT_OF_CAP to
ACTUAL_SALARY_PCT_OF_CAP (contract_model_v3_predictions.csv already does
that, and it's a circular yardstick here -- the actual contract IS what
we'd be trying to "beat", so it can't also be the ground truth we grade it
against). Instead this builds a third, independent estimate of what each
free agent actually turned out to be worth, and scores both the model's
pre-signing prediction AND the real signed contract against it:

  HINDSIGHT VALUE: run the exact same LOSO-trained model (same weights,
  same fold, same feature set as build_contract_model_v3.py) but feed it
  the player's real TARGET_SEASON vector/usage/awards instead of the
  PREV_SEASON one used for the actual forecast. This answers "what would
  the model have paid this player if it had known in advance exactly how
  they were about to play, in the very season the contract covers?" --
  the best production-grounded proxy available in this repo for what a
  free agent was actually worth, holding the valuation function itself
  fixed so it's a fair comparison. Same minutes-reliability gate as v3,
  just anchored on TARGET_SEASON instead of PREV_SEASON.

Then for every free-agent row with hindsight data available (TARGET_SEASON
already played -- excludes the current, not-yet-played free-agent class):
  MODEL_ERR_PP = |PREDICTED_SALARY_PCT_OF_CAP - HINDSIGHT_PCT_OF_CAP|
  NBA_ERR_PP   = |ACTUAL_SALARY_PCT_OF_CAP    - HINDSIGHT_PCT_OF_CAP|
"Model beat the NBA" on a row when MODEL_ERR_PP < NBA_ERR_PP -- i.e. the
pre-signing prediction landed closer to what the player actually turned
out to produce that season than the contract real GMs gave them did.

Output: data/free_agency/contract_vs_nba_eval.csv (row-level), plus a
printed pooled/per-season summary (MAE, win rate, Wilcoxon signed-rank
test on the paired |error| gap).

Usage:
    python scripts/evaluate_contract_vs_nba.py
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from xgboost import XGBRegressor

from apply_minutes_gate import three_year_total_minutes
from build_contract_model_v3 import (
    CATEGORIES,
    FEATURE_COLUMNS,
    XGB_PARAMS,
    MINUTES_THRESHOLD,
    build_freeagent_rows,
    load_vector_pool,
    FEATURES_PATH,
)
from pathlib import Path

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "free_agency" / "contract_vs_nba_eval.csv"


def build_hindsight_features(fa_rows: pd.DataFrame, vector_pool: pd.DataFrame,
                              features: pd.DataFrame) -> pd.DataFrame:
    """Same FEATURE_COLUMNS as fa_rows, but sourced from each row's
    TARGET_SEASON instead of PREV_SEASON. NaN where the target season's
    vector/features aren't available (not yet played)."""
    vec_idx = vector_pool.set_index(["SEASON", "PLAYER_ID"])
    feat_idx = features.set_index(["SEASON", "PLAYER_ID"])

    out_rows = []
    for _, r in fa_rows.iterrows():
        key = (r["TARGET_SEASON"], r["PLAYER_ID"])
        if key not in vec_idx.index or key not in feat_idx.index:
            out_rows.append({c: np.nan for c in FEATURE_COLUMNS})
            continue
        vrow = vec_idx.loc[key]
        if isinstance(vrow, pd.DataFrame):
            vrow = vrow.iloc[0]
        frow = feat_idx.loc[key]
        if isinstance(frow, pd.DataFrame):
            frow = frow.iloc[0]
        if pd.isna(frow.get("USG_PCT")):
            out_rows.append({c: np.nan for c in FEATURE_COLUMNS})
            continue
        row = {cat: vrow[cat] for cat in CATEGORIES}
        row["EXPERIENCE_YRS"] = frow["EXPERIENCE_YRS"]
        row["USG_PCT"] = frow["USG_PCT"]
        row["ALL_STAR_SELECTION"] = int(frow["ALL_STAR_SELECTION"])
        row["ALL_NBA_SELECTION"] = int(frow["ALL_NBA_SELECTION"])
        out_rows.append(row)
    return pd.DataFrame(out_rows, index=fa_rows.index)


def main():
    vector_pool = load_vector_pool()
    features = pd.read_csv(FEATURES_PATH)
    fa_rows = build_freeagent_rows(vector_pool, features).reset_index(drop=True)
    hindsight_feat = build_hindsight_features(fa_rows, vector_pool, features)

    target_seasons = sorted(fa_rows["TARGET_SEASON"].unique())
    print(f"{len(fa_rows)} free-agent rows across {len(target_seasons)} target seasons.")

    all_out = []
    for held_out in target_seasons:
        train = fa_rows[fa_rows["TARGET_SEASON"] != held_out]
        test_mask = fa_rows["TARGET_SEASON"] == held_out
        test = fa_rows[test_mask]
        test_hindsight = hindsight_feat[test_mask]
        if len(train) < 20 or test.empty:
            print(f"  Skipping {held_out}: too few training rows ({len(train)}).")
            continue

        # rows in this fold with a usable hindsight feature row (season already played)
        usable = test_hindsight[FEATURE_COLUMNS].notna().all(axis=1)
        if not usable.any():
            print(f"  Skipping {held_out}: no played-season data yet for hindsight ground truth.")
            continue

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(train[FEATURE_COLUMNS], train["ACTUAL_SALARY_PCT_OF_CAP"])

        test_u = test[usable]
        pred = model.predict(test_u[FEATURE_COLUMNS])
        hindsight = model.predict(test_hindsight[usable][FEATURE_COLUMNS])

        out = test_u[["PLAYER_ID", "PLAYER_NAME", "PREV_SEASON", "TARGET_SEASON",
                       "ACTUAL_SALARY_PCT_OF_CAP"]].copy()
        out["PREDICTED_PCT"] = pred
        out["HINDSIGHT_PCT"] = hindsight
        all_out.append(out)

    result = pd.concat(all_out, ignore_index=True)

    # minutes-reliability gate, same shape as v3: predicted gated off PREV_SEASON
    # minutes (what was knowable pre-signing), hindsight gated off TARGET_SEASON
    # minutes (how much signal that season's own vector actually carries).
    pred_min = three_year_total_minutes(result["PLAYER_ID"], result["PREV_SEASON"])
    hind_min = three_year_total_minutes(result["PLAYER_ID"], result["TARGET_SEASON"])
    result["PREDICTED_PCT"] *= np.sqrt((pred_min / MINUTES_THRESHOLD).clip(max=1.0, min=0.0))
    result["HINDSIGHT_PCT"] *= np.sqrt((hind_min / MINUTES_THRESHOLD).clip(max=1.0, min=0.0))

    result["MODEL_ERR_PP"] = (result["PREDICTED_PCT"] - result["HINDSIGHT_PCT"]).abs()
    result["NBA_ERR_PP"] = (result["ACTUAL_SALARY_PCT_OF_CAP"] - result["HINDSIGHT_PCT"]).abs()
    result["MODEL_CLOSER"] = result["MODEL_ERR_PP"] < result["NBA_ERR_PP"]

    result = result.sort_values(["TARGET_SEASON", "HINDSIGHT_PCT"], ascending=[True, False])
    result.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(result)} rows to {OUT_PATH}")

    print("\nPer-season (against hindsight ground truth):")
    rows = []
    for season, g in result.groupby("TARGET_SEASON"):
        rows.append({
            "TARGET_SEASON": season, "N": len(g),
            "MODEL_MAE": g["MODEL_ERR_PP"].mean(), "NBA_MAE": g["NBA_ERR_PP"].mean(),
            "MODEL_WIN_RATE": g["MODEL_CLOSER"].mean(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    model_err = result["MODEL_ERR_PP"].to_numpy()
    nba_err = result["NBA_ERR_PP"].to_numpy()
    win_rate = result["MODEL_CLOSER"].mean()
    stat, pval = wilcoxon(model_err, nba_err)

    print(f"\nPooled (N={len(result)}):")
    print(f"  Model MAE vs hindsight value: {model_err.mean():.2f}pp  (RMSE {np.sqrt((model_err**2).mean()):.2f}pp)")
    print(f"  NBA contract MAE vs hindsight value: {nba_err.mean():.2f}pp  (RMSE {np.sqrt((nba_err**2).mean()):.2f}pp)")
    print(f"  Model was closer to hindsight value on {100*win_rate:.1f}% of rows.")
    print(f"  Wilcoxon signed-rank on paired |error| gap: stat={stat:.1f}, p={pval:.4f}"
          f"  ({'significant' if pval < 0.05 else 'not significant'} at 0.05)")


if __name__ == "__main__":
    main()
