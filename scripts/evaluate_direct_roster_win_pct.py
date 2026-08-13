"""
Isolates the "skip NEED_VECTOR entirely" experiment buried inside
evaluate_roster_need_win_pct.py / predict_2026_27_team_profile.py: fit a
single RidgeCV directly on each team's 25 roster-aggregate features
(minutes-weighted mean/std of the 12 player-vector skill categories, plus
TOTAL_MIN and ROSTER_N) to predict W_PCT -- no NEED_VECTOR reconstruction
step at all, just "player vectors + minutes -> win%".

Two outputs:
  1. Historical validation: the same LOSO-across-8-seasons discipline as
     evaluate_roster_need_win_pct.py's `direct_preds` (240 team-seasons,
     out-of-fold).
  2. 2026-27 projection: RidgeCV refit on all 8 seasons, applied to the
     real 2026-27 rosters (predict_2026_27_team_profile.py's
     build_2026_27_team_rows(), each player's most recent known vector +
     minutes), clipped to [0.05, 0.95] win-pct, same as the blended model.

Written for a side-by-side comparison against the NEED_VECTOR-mediated
model already tracked in the "2026-27 Team Profile & Win Projections"
artifact -- does bypassing the 12-category bottleneck lose signal, or was
the bottleneck not costing anything?

Output: data/win_pct_model/direct_roster_win_pct_results.json

Usage:
    python scripts/evaluate_direct_roster_win_pct.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import (
    build_dataset as build_roster_dataset,
    feature_columns,
)
from predict_win_pct_from_need import RIDGE_ALPHAS, rmse
from evaluate_roster_need_win_pct import attach_win_pct, loso_predict, _naive_preds
from predict_2026_27_team_profile import build_2026_27_team_rows, GAMES_PER_SEASON, W_PCT_CLIP
from predict_2026_27_minutes import build_latest_vectors, build_latest_minutes

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "win_pct_model" / "direct_roster_win_pct_results.json"


def main():
    print("Building roster-aggregate features for all 8 historical seasons...")
    roster_df = build_roster_dataset()
    feat_cols = feature_columns(roster_df)
    df = attach_win_pct(roster_df)
    print(f"  {len(df)} team-seasons, {len(feat_cols)} features: {feat_cols}")

    print("LOSO validation: RidgeCV(25 roster features) -> W_PCT...")
    naive_preds = _naive_preds(df)
    direct_preds, models = loso_predict(df, feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    print(f"  Naive baseline RMSE: {rmse(naive_preds):.4f}")
    print(f"  Direct roster-vector RMSE: {rmse(direct_preds):.4f}")

    gp_lookup, w_lookup = {}, {}
    for season in STANDARD_SEASONS:
        wins = pd.read_csv(ROOT / "data" / "team_wins" / f"team_wins_{season}.csv").set_index("TEAM")
        gp_lookup[season] = wins["GP"]
        w_lookup[season] = wins["W"]

    historical = []
    for _, r in direct_preds.iterrows():
        gp = int(gp_lookup[r["SEASON"]].loc[r["TEAM"]])
        historical.append({
            "season": r["SEASON"], "team": r["TEAM"],
            "actual_wins": int(w_lookup[r["SEASON"]].loc[r["TEAM"]]),
            "pred_wins": round(float(r["PRED"]) * gp, 1),
        })

    print("Refitting RidgeCV on all 8 seasons, projecting 2026-27 rosters...")
    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit(df[feat_cols], df["W_PCT"])

    latest_vectors = build_latest_vectors()
    latest_minutes = build_latest_minutes()
    team_2026 = build_2026_27_team_rows(latest_vectors, latest_minutes)
    pred = np.clip(ridge.predict(team_2026[feat_cols]), *W_PCT_CLIP)
    team_2026["PRED_W_PCT"] = pred
    team_2026["PROJECTED_WINS"] = (pred * GAMES_PER_SEASON).round(1)
    projection = team_2026[["TEAM", "ROSTER_N", "PRED_W_PCT", "PROJECTED_WINS"]] \
        .sort_values("PRED_W_PCT", ascending=False)

    out = {
        "feature_columns": feat_cols,
        "rmse": {"naive": rmse(naive_preds), "direct_roster_vector": rmse(direct_preds)},
        "historical": historical,
        "projection_2026_27": [
            {"team": row.TEAM, "roster_n": int(row.ROSTER_N),
             "pred_wpct": round(float(row.PRED_W_PCT), 4),
             "projected_wins": float(row.PROJECTED_WINS)}
            for row in projection.itertuples()
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(out, indent=None))
    print(f"\nWrote {OUTPUT_PATH}")
    print(projection.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
