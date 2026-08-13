"""
Production version of the winning combination found in
compare_models_vectors_ratings_wins.py: RidgeCV roster-vectors -> OFF/DEF
rating, then RidgeCV (25 roster features + predicted OFF/DEF/NET rating)
-> win%. LOSO RMSE 0.0777 across 8 historical seasons -- beats both the
direct 25-feature model (0.0946) and the NEED_VECTOR blend (0.0915) used
on the other two artifact pages, though still short of a Ridge fit on the
real, team-stats-derived NEED_VECTOR (0.0630, which needs a season that's
already been played and so isn't available for 2026-27).

Two outputs:
  1. Historical validation: genuine LOSO out-of-fold predictions (240
     team-seasons), same discipline as evaluate_direct_roster_win_pct.py --
     for a held-out season, both the rating-prediction model and the win%
     model are trained on the other 7 seasons only.
  2. 2026-27 projection: both stages refit on all 8 seasons (no held-out
     season -- there's no 2026-27 ground truth to hold one out against),
     applied to the real 2026-27 rosters via each player's most recent
     known skill vector and season-total minutes (same convention as
     predict_2026_27_team_profile.py and evaluate_direct_roster_win_pct.py).

Output: data/win_pct_model/rating_detour_win_pct_results.json

Usage:
    python scripts/predict_rating_detour_win_pct.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import build_dataset as build_roster_dataset, feature_columns
from predict_win_pct_from_need import RIDGE_ALPHAS, rmse
from evaluate_roster_need_win_pct import attach_win_pct, loso_predict
from analyze_player_vectors_vs_ratings import attach_ratings
from predict_2026_27_team_profile import build_2026_27_team_rows, GAMES_PER_SEASON, W_PCT_CLIP
from predict_2026_27_minutes import build_latest_vectors, build_latest_minutes

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "win_pct_model" / "rating_detour_win_pct_results.json"


def main():
    roster_df = build_roster_dataset()
    feat_cols = feature_columns(roster_df)
    df = attach_ratings(attach_win_pct(roster_df))
    print(f"{len(df)} team-seasons, {len(feat_cols)} roster-vector features")

    # ---- historical LOSO validation (honest backtest) ----
    print("Stage 1 LOSO: roster-vectors -> OFF_RATING / DEF_RATING...")
    d_off = df.drop(columns=["W_PCT"]).rename(columns={"OFF_RATING": "W_PCT"})
    d_def = df.drop(columns=["W_PCT"]).rename(columns={"DEF_RATING": "W_PCT"})
    off_loso, _ = loso_predict(d_off, feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    def_loso, _ = loso_predict(d_def, feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    off_p = off_loso.rename(columns={"PRED": "PRED_OFF_RATING"})[["SEASON", "TEAM", "PRED_OFF_RATING"]]
    def_p = def_loso.rename(columns={"PRED": "PRED_DEF_RATING"})[["SEASON", "TEAM", "PRED_DEF_RATING"]]
    aug = df.merge(off_p, on=["SEASON", "TEAM"]).merge(def_p, on=["SEASON", "TEAM"])
    aug["PRED_NET_RATING"] = aug["PRED_OFF_RATING"] - aug["PRED_DEF_RATING"]
    aug_feat_cols = feat_cols + ["PRED_OFF_RATING", "PRED_DEF_RATING", "PRED_NET_RATING"]

    print("Stage 2 LOSO: 28 features -> win%...")
    win_loso, _ = loso_predict(aug, aug_feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    print(f"  LOSO win% RMSE: {rmse(win_loso):.4f}")

    gp_lookup, w_lookup = {}, {}
    for season in STANDARD_SEASONS:
        wins = pd.read_csv(ROOT / "data" / "team_wins" / f"team_wins_{season}.csv").set_index("TEAM")
        gp_lookup[season] = wins["GP"]
        w_lookup[season] = wins["W"]

    historical = []
    for _, r in win_loso.iterrows():
        gp = int(gp_lookup[r["SEASON"]].loc[r["TEAM"]])
        historical.append({
            "season": r["SEASON"], "team": r["TEAM"],
            "actual_wins": int(w_lookup[r["SEASON"]].loc[r["TEAM"]]),
            "pred_wins": round(float(r["PRED"]) * gp, 1),
        })

    # ---- production fit on all 8 seasons, applied to 2026-27 rosters ----
    print("Refitting stage 1 + stage 2 on all 8 seasons...")
    off_full = RidgeCV(alphas=RIDGE_ALPHAS)
    off_full.fit(df[feat_cols], df["OFF_RATING"])
    def_full = RidgeCV(alphas=RIDGE_ALPHAS)
    def_full.fit(df[feat_cols], df["DEF_RATING"])

    win_full = RidgeCV(alphas=RIDGE_ALPHAS)
    win_full.fit(aug[aug_feat_cols], aug["W_PCT"])

    print("Building 2026-27 team-roster features and projecting...")
    latest_vectors = build_latest_vectors()
    latest_minutes = build_latest_minutes()
    team_2026 = build_2026_27_team_rows(latest_vectors, latest_minutes)
    team_2026["PRED_OFF_RATING"] = off_full.predict(team_2026[feat_cols])
    team_2026["PRED_DEF_RATING"] = def_full.predict(team_2026[feat_cols])
    team_2026["PRED_NET_RATING"] = team_2026["PRED_OFF_RATING"] - team_2026["PRED_DEF_RATING"]
    pred = np.clip(win_full.predict(team_2026[aug_feat_cols]), *W_PCT_CLIP)
    team_2026["PRED_W_PCT"] = pred
    team_2026["PROJECTED_WINS"] = (pred * GAMES_PER_SEASON).round(1)
    projection = team_2026[[
        "TEAM", "ROSTER_N", "PRED_W_PCT", "PROJECTED_WINS",
        "PRED_OFF_RATING", "PRED_DEF_RATING", "PRED_NET_RATING",
    ]].sort_values("PRED_W_PCT", ascending=False)

    out = {
        "feature_columns": aug_feat_cols,
        "rmse": {"stage1_off": rmse(off_loso), "stage1_def": rmse(def_loso), "stage2_win": rmse(win_loso)},
        "historical": historical,
        "projection_2026_27": [
            {"team": row.TEAM, "roster_n": int(row.ROSTER_N),
             "pred_wpct": round(float(row.PRED_W_PCT), 4),
             "projected_wins": float(row.PROJECTED_WINS),
             "pred_off_rating": round(float(row.PRED_OFF_RATING), 2),
             "pred_def_rating": round(float(row.PRED_DEF_RATING), 2),
             "pred_net_rating": round(float(row.PRED_NET_RATING), 2)}
            for row in projection.itertuples()
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(out, indent=None))
    print(f"\nWrote {OUTPUT_PATH}")
    print(projection.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
