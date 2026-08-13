"""
Tests whether a team's 12-category NEED_VECTOR (data/team_vectors_json/,
see team.md) predicts its actual win percentage (data/team_wins/, pulled via
nba_api's LeagueDashTeamStats Base measure type) -- i.e. do the weaknesses
this pipeline identifies actually cost teams wins.

Dataset: ~240 team-seasons (30 teams x 8 STANDARD_SEASONS), 12 features
(the NEED_VECTOR categories) predicting a single target (W_PCT).

Models, both regularized linear regressions chosen for interpretability at
this sample size (~210 training rows per LOSO fold, 12 correlated
features -- a black-box model answers "what's the RMSE" but not "which
categories actually predict losing", which is the more interesting
question here):
  - Ridge (L2 only)
  - Elastic Net (L1 + L2) -- also does soft feature selection, useful for
    checking whether some categories carry no real signal for winning
  - A single-feature "composite need" baseline (mean of the 12 categories)
    regressed alone against W_PCT, to check whether the 12-category
    breakdown adds anything beyond "how needy is this team overall"

Regularization strength (alpha) is chosen per LOSO fold via inner 5-fold CV
on that fold's training rows only (RidgeCV / ElasticNetCV), so the
held-out season never influences hyperparameter selection -- same
discipline as pls_loso_cv() in predict_team_need_from_roster.py.

Evaluation: leave-one-season-out (LOSO) cross-validation across
STANDARD_SEASONS, matching the LOSO convention used throughout this repo.

Usage:
    python scripts/predict_win_pct_from_need.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, RidgeCV

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_team_vectors import PLAY_TYPES, STANDARD_SEASONS

DATA_DIR = Path(__file__).parent.parent / "data"
TEAM_VECTOR_DIR = DATA_DIR / "team_vectors_json"
TEAM_WINS_DIR = DATA_DIR / "team_wins"

NEED_CATEGORIES = PLAY_TYPES + ["RIM_DEFENSE", "PERIMETER_DEFENSE", "REBOUNDING"]

RIDGE_ALPHAS = np.logspace(-2, 3, 30)
ENET_ALPHAS = np.logspace(-3, 1, 30)
ENET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def load_season_rows(season: str) -> pd.DataFrame:
    needs = json.loads((TEAM_VECTOR_DIR / f"team_vectors_{season}.json").read_text(encoding="utf-8"))
    wins = pd.read_csv(TEAM_WINS_DIR / f"team_wins_{season}.csv").set_index("TEAM")

    rows = []
    for team, entry in needs.items():
        if team not in wins.index:
            continue
        row = {"TEAM": team, "SEASON": season, "W_PCT": float(wins.loc[team, "W_PCT"])}
        for cat in NEED_CATEGORIES:
            row[cat] = entry["NEED_VECTOR"][cat]
        row["COMPOSITE_NEED"] = float(np.mean([entry["NEED_VECTOR"][c] for c in NEED_CATEGORIES]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(seasons: list[str] = STANDARD_SEASONS) -> pd.DataFrame:
    return pd.concat([load_season_rows(s) for s in seasons], ignore_index=True)


def loso_predict(df: pd.DataFrame, feat_cols: list[str], make_model, seasons: list[str] = STANDARD_SEASONS):
    """Generic LOSO runner: make_model() returns a fresh unfit sklearn
    estimator (already wired for its own inner-CV alpha selection). Returns
    a DataFrame with TEAM, SEASON, actual W_PCT, and out-of-fold predictions,
    plus the list of fitted models (one per fold, for coefficient inspection)."""
    records = []
    models = []
    for held_out in seasons:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue

        model = make_model()
        model.fit(train[feat_cols], train["W_PCT"])
        preds = model.predict(test[feat_cols])
        models.append(model)

        for team, actual, pred in zip(test["TEAM"], test["W_PCT"], preds):
            records.append({"SEASON": held_out, "TEAM": team, "ACTUAL": actual, "PRED": pred})

    return pd.DataFrame(records), models


def rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((df["ACTUAL"] - df["PRED"]) ** 2)))


def main():
    df = build_dataset()
    print(f"Built {len(df)} team-season rows across {df['SEASON'].nunique()} seasons.")
    print(f"W_PCT range: {df['W_PCT'].min():.3f} - {df['W_PCT'].max():.3f}, std: {df['W_PCT'].std():.3f}")

    composite_preds, _ = loso_predict(df, ["COMPOSITE_NEED"], lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    ridge_preds, ridge_models = loso_predict(df, NEED_CATEGORIES, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    enet_preds, enet_models = loso_predict(
        df, NEED_CATEGORIES,
        lambda: ElasticNetCV(alphas=ENET_ALPHAS, l1_ratio=ENET_L1_RATIOS, max_iter=10000),
    )

    naive_preds = df.copy()
    naive_preds = naive_preds.rename(columns={"W_PCT": "ACTUAL"})
    # naive baseline: same LOSO discipline, predict training seasons' mean W_PCT
    naive_records = []
    for held_out in STANDARD_SEASONS:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue
        pred_value = train["W_PCT"].mean()
        for team, actual in zip(test["TEAM"], test["W_PCT"]):
            naive_records.append({"SEASON": held_out, "TEAM": team, "ACTUAL": actual, "PRED": pred_value})
    naive_preds = pd.DataFrame(naive_records)

    print("\nLOSO RMSE (predicting W_PCT):")
    print(f"  Naive baseline (train-mean):     {rmse(naive_preds):.4f}")
    print(f"  Composite need (1 feature):      {rmse(composite_preds):.4f}")
    print(f"  Ridge (12 features):             {rmse(ridge_preds):.4f}")
    print(f"  Elastic Net (12 features):       {rmse(enet_preds):.4f}")

    print("\nRidge coefficients per fold (per-category effect on W_PCT; negative = more need -> fewer wins):")
    coef_df = pd.DataFrame([m.coef_ for m in ridge_models], columns=NEED_CATEGORIES, index=STANDARD_SEASONS)
    print(coef_df.round(3).to_string())
    print("\nRidge coefficients, averaged across folds, sorted:")
    print(coef_df.mean().sort_values().round(4).to_string())

    print("\nElastic Net coefficients, averaged across folds, sorted (0 = dropped by L1):")
    enet_coef_df = pd.DataFrame([m.coef_ for m in enet_models], columns=NEED_CATEGORIES, index=STANDARD_SEASONS)
    print(enet_coef_df.mean().sort_values().round(4).to_string())

    # dump everything a downstream visualization needs
    out = {
        "seasons": STANDARD_SEASONS,
        "categories": NEED_CATEGORIES,
        "rmse": {
            "naive": rmse(naive_preds),
            "composite": rmse(composite_preds),
            "ridge": rmse(ridge_preds),
            "enet": rmse(enet_preds),
        },
        "calibration": {
            "ridge": ridge_preds.to_dict(orient="records"),
            "enet": enet_preds.to_dict(orient="records"),
            "naive": naive_preds.to_dict(orient="records"),
        },
        "ridge_coef_by_fold": coef_df.reset_index().rename(columns={"index": "SEASON"}).to_dict(orient="records"),
        "ridge_coef_mean": coef_df.mean().to_dict(),
        "enet_coef_by_fold": enet_coef_df.reset_index().rename(columns={"index": "SEASON"}).to_dict(orient="records"),
        "enet_coef_mean": enet_coef_df.mean().to_dict(),
    }
    out_path = Path(__file__).parent.parent / "data" / "win_pct_model" / "win_pct_model_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
