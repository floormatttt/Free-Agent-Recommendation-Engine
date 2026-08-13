"""
Projects each team's 12-category NEED_VECTOR profile and win percentage for
the 2026-27 season from the actual 2026-27 rosters
(data/rosters/rosters_2026-27.csv) -- genuine forward inference, chaining
predict_team_need_from_roster.py's roster -> NEED_VECTOR reconstruction into
predict_win_pct_from_need.py's NEED_VECTOR -> win-pct regression, the same
two stages evaluate_roster_need_win_pct.py validates via LOSO, but composed
here as a single production fit on ALL 8 STANDARD_SEASONS (no held-out
season -- there's no 2026-27 ground truth to hold out against).

Like predict_2026_27_minutes.py, 2026-27 has no team-level stats yet, so
each roster player is assigned their most recent available skill vector and
most recent available season-total minutes (as the aggregation weight) --
reused directly from predict_2026_27_minutes.py's build_latest_vectors()/
build_latest_minutes().

Pipeline (mirrors evaluate_roster_need_win_pct.py's best-performing
"blend" model):
  1. Roster -> NEED_VECTOR: PLS/XGBoost ensemble (predict_team_need_from_
     roster.py), fit on all 8 seasons' real roster-aggregate -> real
     NEED_VECTOR pairs, applied to 2026-27 rosters' aggregate features.
  2. NEED_VECTOR -> win-pct, blended:
       - RidgeCV on the 25 direct roster-aggregate features alone
         (skips the NEED_VECTOR bottleneck entirely), fit on all 8
         seasons' real (roster features, real W_PCT) pairs.
       - PLSRegression on 25 direct features + 12 LOSO-reconstructed
         NEED_VECTOR categories (37 total, matching evaluate_roster_need_
         win_pct.py's build_augmented_dataset()), fit on all 8 seasons.
     The two out-of-fold-trained predictions are averaged.
  3. 2026-27 rosters get the same 37 features (25 direct + the step-1
     ensemble's reconstructed NEED_VECTOR, since no real one exists), and
     the blend's average is the projected W_PCT.

Win percentages are clipped to [0.05, 0.95] -- extrapolation on a genuinely
new roster (e.g. an unprecedented superteam or teardown) could otherwise
land outside what a regression fit on 8 seasons of real outcomes can
plausibly claim.

Output: data/2026_27_projections/multi_year_team_profile_predictions.csv, one row per
team: the 12 reconstructed NEED_VECTOR categories, PRED_W_PCT, and
PROJECTED_WINS (PRED_W_PCT * 82, rounded).

Usage:
    python scripts/predict_2026_27_team_profile.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import (
    NEED_CATEGORIES, XGB_PARAMS, PLS_MAX_COMPONENTS,
    build_dataset as build_roster_dataset,
    feature_columns as roster_feature_columns,
    _select_pls_components,
    ensemble_predictions,
)
from predict_win_pct_from_need import RIDGE_ALPHAS
from evaluate_roster_need_win_pct import attach_win_pct, build_augmented_dataset
from predict_2026_27_minutes import build_latest_vectors, build_latest_minutes, ROSTER_PATH

OUTPUT_PATH = Path("data/2026_27_projections/multi_year_team_profile_predictions.csv")
GAMES_PER_SEASON = 82
W_PCT_CLIP = (0.05, 0.95)

# A team's real on-court minutes are fixed at 240/game (5 players x 48 min,
# ignoring OT) regardless of roster churn, so a season sums to ~19,680 --
# matching the historical TOTAL_MIN feature's actual range (~11k-23k, mean
# ~19,156, per a follow-up review of this pipeline). The 2026-27 roster
# rows instead summed each player's own most-recent *season-total* minutes
# independently, which isn't bounded by a shared team pool: teams with
# several recent full-time starters (e.g. Houston, Atlanta) landed above
# the historical max, teams whose vets missed time to injury/trade (e.g.
# Chicago) landed near the min -- out-of-distribution inputs to a model
# that never saw values outside that historical band. Rescaling every
# team's roster minutes proportionally to this fixed pool fixes TOTAL_MIN
# without changing the WMEAN/WSTD skill-weighting (those use normalized
# weights, scale-invariant to a uniform per-team rescale) -- it does NOT
# fix the deeper issue the same review flagged, that a player's *relative*
# share within their own team is still last season's raw minutes, so a
# recently-traded/injury-shortened star (e.g. a just-acquired Trae Young)
# is still underweighted relative to a healthy returning bench player.
TEAM_TOTAL_MIN_TARGET = 240 * GAMES_PER_SEASON


def build_2026_27_team_rows(latest_vectors: dict, latest_minutes: dict) -> pd.DataFrame:
    """25 roster-aggregate features per 2026-27 team, same shape as
    predict_team_need_from_roster.py's load_team_rows() -- minutes-weighted
    mean/std of each of the 12 skill categories plus total tracked minutes
    -- but sourced from each player's most recent historical vector/minutes
    instead of a single completed season."""
    rosters = pd.read_csv(ROSTER_PATH, encoding="utf-8")

    by_team: dict[str, list[tuple[float, dict]]] = {}
    for _, r in rosters.iterrows():
        pid = int(r["PLAYER_ID"])
        if pid not in latest_vectors:
            continue
        vector, _ = latest_vectors[pid]
        minutes, _ = latest_minutes.get(pid, (0.0, None))
        if not minutes or minutes <= 0:
            continue
        by_team.setdefault(r["TEAM"], []).append((minutes, vector))

    rows = []
    for team, roster in by_team.items():
        if len(roster) < 2:
            print(f"  WARNING: {team} has fewer than 2 players with historical data -- skipped")
            continue
        mins = np.array([m for m, _ in roster], dtype=float)
        weights = mins / mins.sum()  # scale-invariant -- unaffected by the rescale below
        total_min = TEAM_TOTAL_MIN_TARGET  # rescaled to a fixed 240-min/game pool, see module docstring

        row = {"TEAM": team, "TOTAL_MIN": total_min, "ROSTER_N": len(roster)}
        for cat in NEED_CATEGORIES:
            vals = np.array([v[cat] for _, v in roster], dtype=float)
            w_mean = np.sum(weights * vals)
            w_std = np.sqrt(np.sum(weights * (vals - w_mean) ** 2))
            row[f"{cat}_WMEAN"] = w_mean
            row[f"{cat}_WSTD"] = w_std
        rows.append(row)

    return pd.DataFrame(rows)


def fit_ensemble_full(train_df: pd.DataFrame, feat_cols: list[str]):
    """Full (non-LOSO) fit of predict_team_need_from_roster.py's PLS +
    XGBoost ensemble on every available historical team-season -- there's no
    2026-27 ground truth to hold a fold out against."""
    target_cols = [f"TARGET_{c}" for c in NEED_CATEGORIES]
    X = train_df[feat_cols].to_numpy()
    Y = train_df[target_cols].to_numpy()

    n_components = _select_pls_components(X, Y, PLS_MAX_COMPONENTS)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)

    xgb_models = {}
    for cat in NEED_CATEGORIES:
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(train_df[feat_cols], train_df[f"TARGET_{cat}"])
        xgb_models[cat] = model

    return pls, xgb_models, n_components


def predict_ensemble(pls, xgb_models, new_df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    X_new = new_df[feat_cols].to_numpy()
    pls_pred = pls.predict(X_new)
    xgb_pred = np.zeros_like(pls_pred)
    for i, cat in enumerate(NEED_CATEGORIES):
        xgb_pred[:, i] = xgb_models[cat].predict(new_df[feat_cols])
    return (pls_pred + xgb_pred) / 2


def main():
    print("Stage 1: fitting roster -> NEED_VECTOR ensemble on all 8 historical seasons...")
    roster_df = build_roster_dataset()
    direct_feat_cols = roster_feature_columns(roster_df)
    pls, xgb_models, n_components = fit_ensemble_full(roster_df, direct_feat_cols)
    print(f"  PLS n_components: {n_components}")

    print("Stage 2: fitting NEED_VECTOR -> win-pct blend (Ridge + augmented PLS) on all 8 seasons...")
    augmented = attach_win_pct(build_augmented_dataset())
    combo_feat_cols = direct_feat_cols + [f"PRED_{c}" for c in NEED_CATEGORIES]

    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit(augmented[direct_feat_cols], augmented["W_PCT"])

    win_n_components = _select_pls_components(
        augmented[combo_feat_cols].to_numpy(), augmented["W_PCT"].to_numpy(), PLS_MAX_COMPONENTS,
    )
    win_pls = PLSRegression(n_components=win_n_components)
    win_pls.fit(augmented[combo_feat_cols], augmented["W_PCT"])
    print(f"  Win-pct PLS n_components: {win_n_components}")

    print("Building 2026-27 team-roster features (latest known vector/minutes per player)...")
    latest_vectors = build_latest_vectors()
    latest_minutes = build_latest_minutes()
    team_2026 = build_2026_27_team_rows(latest_vectors, latest_minutes)
    print(f"  {len(team_2026)} teams scored, roster sizes (known-data players): "
          f"{team_2026['ROSTER_N'].min()}-{team_2026['ROSTER_N'].max()}")

    print("Reconstructing 2026-27 NEED_VECTOR (Stage 1 ensemble)...")
    need_pred = predict_ensemble(pls, xgb_models, team_2026, direct_feat_cols)
    for i, cat in enumerate(NEED_CATEGORIES):
        team_2026[cat] = need_pred[:, i]
        team_2026[f"PRED_{cat}"] = need_pred[:, i]
    team_2026["COMPOSITE_NEED"] = team_2026[NEED_CATEGORIES].mean(axis=1)

    print("Projecting 2026-27 win percentage (Stage 2 blend)...")
    ridge_pred = ridge.predict(team_2026[direct_feat_cols])
    pls_pred = win_pls.predict(team_2026[combo_feat_cols]).ravel()
    blend_pred = (ridge_pred + pls_pred) / 2
    clipped = np.clip(blend_pred, *W_PCT_CLIP)
    n_clipped = int(np.sum(clipped != blend_pred))

    team_2026["PRED_W_PCT"] = clipped.round(4)
    team_2026["PROJECTED_WINS"] = (clipped * GAMES_PER_SEASON).round(1)

    out_cols = ["TEAM", "ROSTER_N", "PRED_W_PCT", "PROJECTED_WINS", "COMPOSITE_NEED"] + NEED_CATEGORIES
    out = team_2026[out_cols].sort_values("PRED_W_PCT", ascending=False)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{n_clipped} teams clipped to the [{W_PCT_CLIP[0]}, {W_PCT_CLIP[1]}] win-pct range.")
    print(f"Wrote {OUTPUT_PATH}\n")
    print(out[["TEAM", "PRED_W_PCT", "PROJECTED_WINS", "COMPOSITE_NEED"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
