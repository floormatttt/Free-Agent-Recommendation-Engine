"""
Model search for the roster-vectors -> OFF/DEF_RATING -> win% pipeline
built in analyze_player_vectors_vs_ratings.py, which used RidgeCV
throughout without comparing alternatives. Tries several regressor
families at each of two stages, all under the same LOSO-across-8-seasons
discipline, to see whether something beats plain Ridge:

  Stage 1 (roster-vectors -> OFF_RATING / DEF_RATING), candidates:
    Ridge, ElasticNet, PLS, XGBoost, RandomForest
  Stage 2 (25 roster features + stage-1 out-of-fold rating predictions
           -> win%), same five candidates, using each target's stage-1
    *winner* as the rating predictions it's built on.
  Also: stage-2 candidates fit on the 25 roster features ALONE (no rating
    features at all), to check whether a nonlinear model applied directly
    can recover what Ridge needs the rating detour for -- the bias
    analysis flagged unmodeled nonlinear effects as a candidate cause.

Reports LOSO RMSE for every combination. Whichever stage-2 setup wins is
the one worth wiring into the production 2026-27 pipeline.

Usage:
    python scripts/compare_models_vectors_ratings_wins.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from xgboost import XGBRegressor

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import (
    build_dataset as build_roster_dataset, feature_columns,
    _select_pls_components, PLS_MAX_COMPONENTS, XGB_PARAMS,
)
from predict_win_pct_from_need import RIDGE_ALPHAS, ENET_ALPHAS, ENET_L1_RATIOS, rmse
from evaluate_roster_need_win_pct import attach_win_pct, loso_predict, _naive_preds
from analyze_player_vectors_vs_ratings import attach_ratings

warnings.filterwarnings("ignore", category=UserWarning)

RF_PARAMS = dict(n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=0)


def make_pls_predictor(max_components=PLS_MAX_COMPONENTS):
    """loso_predict expects model.fit(X, y)/model.predict(X) on a plain
    estimator; PLSRegression needs its component count chosen per-fold and
    returns a column vector, so it's wrapped instead of passed as a lambda."""
    class _PLS:
        def fit(self, X, y):
            X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
            upper = min(max_components, X.shape[1] - 1, len(X) - 1)
            n_components = _select_pls_components(X, y, upper)
            self.model = PLSRegression(n_components=n_components)
            self.model.fit(X, y)
            return self
        def predict(self, X):
            return self.model.predict(np.asarray(X, dtype=float)).ravel()
    return _PLS()


CANDIDATES = {
    "Ridge": lambda: RidgeCV(alphas=RIDGE_ALPHAS),
    "ElasticNet": lambda: ElasticNetCV(alphas=ENET_ALPHAS, l1_ratio=ENET_L1_RATIOS, max_iter=10000),
    "PLS": make_pls_predictor,
    "XGBoost": lambda: XGBRegressor(**XGB_PARAMS),
    "RandomForest": lambda: RandomForestRegressor(**RF_PARAMS),
}


def run_stage1(df: pd.DataFrame, feat_cols: list[str], target: str) -> dict[str, pd.DataFrame]:
    d = df.drop(columns=["W_PCT"]).rename(columns={target: "W_PCT"})
    naive_preds = _naive_preds(d)
    print(f"\n  {target}  (naive RMSE {rmse(naive_preds):.3f})")
    out = {}
    for name, make_model in CANDIDATES.items():
        preds, _ = loso_predict(d, feat_cols, make_model)
        out[name] = preds
        print(f"    {name:14s} RMSE={rmse(preds):.4f}")
    return out


def run_stage2(df: pd.DataFrame, feat_cols: list[str], label: str) -> None:
    naive_preds = _naive_preds(df)
    print(f"\n  {label}  (naive RMSE {rmse(naive_preds):.4f})")
    for name, make_model in CANDIDATES.items():
        preds, _ = loso_predict(df, feat_cols, make_model)
        print(f"    {name:14s} RMSE={rmse(preds):.4f}")


def main():
    roster_df = build_roster_dataset()
    feat_cols = feature_columns(roster_df)
    df = attach_ratings(attach_win_pct(roster_df))
    print(f"{len(df)} team-seasons, {len(feat_cols)} roster-vector features")

    print("\n=== Stage 1: roster-vectors -> rating, model comparison (LOSO RMSE) ===")
    off_results = run_stage1(df, feat_cols, "OFF_RATING")
    def_results = run_stage1(df, feat_cols, "DEF_RATING")

    off_best = min(off_results, key=lambda k: rmse(off_results[k]))
    def_best = min(def_results, key=lambda k: rmse(def_results[k]))
    print(f"\n  Stage-1 winners: OFF_RATING -> {off_best} ({rmse(off_results[off_best]):.4f}), "
          f"DEF_RATING -> {def_best} ({rmse(def_results[def_best]):.4f})")

    off_preds = off_results[off_best].rename(columns={"PRED": "PRED_OFF_RATING"})[["SEASON", "TEAM", "PRED_OFF_RATING"]]
    def_preds = def_results[def_best].rename(columns={"PRED": "PRED_DEF_RATING"})[["SEASON", "TEAM", "PRED_DEF_RATING"]]
    augmented = df.merge(off_preds, on=["SEASON", "TEAM"]).merge(def_preds, on=["SEASON", "TEAM"])
    augmented["PRED_NET_RATING"] = augmented["PRED_OFF_RATING"] - augmented["PRED_DEF_RATING"]
    aug_feat_cols = feat_cols + ["PRED_OFF_RATING", "PRED_DEF_RATING", "PRED_NET_RATING"]

    print("\n=== Stage 2: win% prediction, model comparison (LOSO RMSE) ===")
    run_stage2(df, feat_cols, "25 roster features only (no rating detour)")
    run_stage2(augmented, aug_feat_cols, f"25 roster features + stage-1-winner ({off_best}/{def_best}) OFF/DEF/NET rating")

    print("\n  (reference points: blend model 0.0915, true team-stats NEED_VECTOR Ridge 0.0630)")


if __name__ == "__main__":
    main()
