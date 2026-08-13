"""
Tests the top recommendation from the win_pct-model bias analysis
(data/win_pct_model/win_pct_bias_analysis.csv / scripts/analyze_win_pct_bias.py): the model's
residual error correlates strongly with OFF_RATING/NET_RATING (r=-0.32/-0.33)
even though neither rating is a model input -- an omitted-variable gap. This
asks whether that gap is fixable from data already on hand: can a team's
OFF_RATING and DEF_RATING be recovered from its roster's player vectors +
minutes alone (the same 25 roster-aggregate features used for the "direct"
win% model), with no team-level box-score stats involved?

Three stages, all LOSO across the 8 STANDARD_SEASONS (240 team-seasons):
  1. Roster-vectors -> OFF_RATING (RidgeCV), roster-vectors -> DEF_RATING
     (RidgeCV). RMSE + correlation vs. a train-mean naive baseline, plus a
     full-data refit's coefficients (which skill categories predict scoring
     efficiency vs. defensive efficiency).
  2. Two-stage win%: augment the 25 roster features with each fold's
     out-of-fold predicted OFF_RATING/DEF_RATING (never the real ones) and
     refit the win% RidgeCV, to see whether this closes any of the gap
     between the direct model (RMSE 0.0946) and the true-NEED_VECTOR model
     (0.0630).

Usage:
    python scripts/analyze_player_vectors_vs_ratings.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import RidgeCV

from nba_api.stats.static import teams as static_teams

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import build_dataset as build_roster_dataset, feature_columns
from predict_win_pct_from_need import RIDGE_ALPHAS, rmse
from evaluate_roster_need_win_pct import attach_win_pct, loso_predict, _naive_preds

ROOT = Path(__file__).resolve().parent.parent
TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}


def attach_ratings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in STANDARD_SEASONS:
        stat_df = pd.read_csv(ROOT / "data" / "team_stats" / f"team_stats_{season}.csv")
        stat_df["TEAM"] = stat_df["TEAM_ID"].map(TEAM_ID_TO_ABBR)
        stat_df["OFF_RATING"] = stat_df["PTS"] / stat_df["POSS"] * 100
        for _, r in stat_df.iterrows():
            rows.append({"SEASON": season, "TEAM": r["TEAM"],
                         "OFF_RATING": r["OFF_RATING"], "DEF_RATING": r["DEF_RATING"]})
    ratings = pd.DataFrame(rows)
    return df.merge(ratings, on=["SEASON", "TEAM"], how="inner")


def corr(a, b):
    return stats.pearsonr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0]


def main():
    roster_df = build_roster_dataset()
    feat_cols = feature_columns(roster_df)
    df = attach_ratings(attach_win_pct(roster_df))
    print(f"{len(df)} team-seasons, {len(feat_cols)} roster-vector features")

    print("\n=== Stage 1: LOSO RidgeCV, roster-vectors -> rating ===")
    results = {}
    for target in ["OFF_RATING", "DEF_RATING"]:
        d = df.drop(columns=["W_PCT"]).rename(columns={target: "W_PCT"})  # reuse loso_predict/_naive_preds (expect W_PCT col)
        naive_preds = _naive_preds(d)
        preds, models = loso_predict(d, feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
        r = corr(preds["ACTUAL"], preds["PRED"])
        results[target] = preds
        print(f"  {target}: naive RMSE={rmse(naive_preds):.3f}  "
              f"roster-vector RMSE={rmse(preds):.3f}  corr(actual,pred)={r:.3f}")

    # full-data refit for interpretable coefficients
    print("\n=== Ridge coefficients (fit on all 8 seasons), sorted by |coef| ===")
    for target in ["OFF_RATING", "DEF_RATING"]:
        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(df[feat_cols], df[target])
        coefs = sorted(zip(feat_cols, ridge.coef_), key=lambda kv: -abs(kv[1]))
        print(f"\n  {target} (intercept {ridge.intercept_:.2f}):")
        for name, c in coefs[:10]:
            print(f"    {name:28s} {c:+.4f}")

    # ---- stage 2: does predicted OFF/DEF_RATING help win% prediction? ----
    print("\n=== Stage 2: augmenting the direct win% model with predicted OFF/DEF_RATING ===")
    off_preds = results["OFF_RATING"].rename(columns={"PRED": "PRED_OFF_RATING"})[["SEASON", "TEAM", "PRED_OFF_RATING"]]
    def_preds = results["DEF_RATING"].rename(columns={"PRED": "PRED_DEF_RATING"})[["SEASON", "TEAM", "PRED_DEF_RATING"]]
    augmented = df.merge(off_preds, on=["SEASON", "TEAM"]).merge(def_preds, on=["SEASON", "TEAM"])
    augmented["PRED_NET_RATING"] = augmented["PRED_OFF_RATING"] - augmented["PRED_DEF_RATING"]
    aug_feat_cols = feat_cols + ["PRED_OFF_RATING", "PRED_DEF_RATING", "PRED_NET_RATING"]

    naive_preds = _naive_preds(augmented)
    direct_preds, _ = loso_predict(augmented, feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    aug_preds, _ = loso_predict(augmented, aug_feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    print(f"  Naive baseline RMSE:                          {rmse(naive_preds):.4f}")
    print(f"  Direct roster-vector model (25 feat) RMSE:    {rmse(direct_preds):.4f}")
    print(f"  + predicted OFF/DEF/NET rating (28 feat) RMSE: {rmse(aug_preds):.4f}")
    print("  (for reference: blend model 0.0915, true-NEED_VECTOR Ridge 0.0630 -- "
          "see evaluate_roster_need_win_pct.py)")

    out_path = ROOT / "data" / "win_pct_model" / "player_vectors_vs_ratings.csv"
    merged = df.merge(off_preds, on=["SEASON", "TEAM"]).merge(def_preds, on=["SEASON", "TEAM"])
    merged["ERR_OFF"] = merged["PRED_OFF_RATING"] - merged["OFF_RATING"]
    merged["ERR_DEF"] = merged["PRED_DEF_RATING"] - merged["DEF_RATING"]
    merged[["SEASON", "TEAM", "OFF_RATING", "PRED_OFF_RATING", "ERR_OFF",
            "DEF_RATING", "PRED_DEF_RATING", "ERR_DEF", "W_PCT"]].round(3).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
