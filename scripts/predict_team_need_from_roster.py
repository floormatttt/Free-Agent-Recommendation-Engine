"""
Tests whether a team's NEED_VECTOR (data/team_vectors_json/, computed
directly from team-level stats -- see team.md) can be reconstructed from its
roster's *player* vectors (data/player_vectors_json/) plus each player's
total minutes (MIN, data/player_stats/), instead. This revisits the
minutes-weighted-aggregation approach team.md pivoted away from (see its
Design Notes / vector_process.md Changelog) -- but here as a *learned*
XGBoost mapping rather than treating the aggregate directly as the answer,
and evaluated against the team vectors already on file rather than assumed
correct by construction.

Feature engineering: rosters have different player counts, so each
team-season's roster is collapsed into a fixed-size feature vector rather
than fed in as a variable-length list of player vectors:
  - minutes-weighted mean of each of the 12 player-vector categories
  - minutes-weighted std of each of the 12 categories (roster depth/spread)
  - total tracked minutes for the team-season
-> 25 features per team-season.

Target: the team's actual 12-category NEED_VECTOR for that season.

Two models are compared, both against a naive baseline (predict the
training-seasons' mean for every team):

  - XGBoost: one XGBRegressor per category (12 independent single-output
    models -- xgboost's native API doesn't do multi-output), max_depth=3,
    reg_alpha=0.1, reg_lambda=1.0. A follow-up critical review of this
    model's LOSO results found it only modestly beats the naive baseline
    overall, and that "modest" win is carried almost entirely by one
    category (PERIMETER_DEFENSE) -- most other categories are statistically
    indistinguishable from, or slightly worse than, the naive baseline, and
    per-fold train-vs-test RMSE gaps (e.g. PERIMETER_DEFENSE 0.014 train vs.
    0.081 test on one fold) indicate real overfitting at ~210 training rows.
  - Partial Least Squares Regression (sklearn PLSRegression): tried at the
    user's request as an alternative better suited to this problem's shape
    -- correlated features, correlated targets, n not much larger than p --
    since it jointly models all 12 targets from shared latent components
    instead of fitting them independently, which may recover cross-category
    correlation (e.g. RIM_DEFENSE/REBOUNDING) the 12-independent-XGBoost
    setup discards. `n_components` is chosen per LOSO fold via inner 5-fold
    CV on that fold's training rows only (never touching the held-out
    season), to avoid picking the hyperparameter using information the
    model wouldn't have at prediction time.

Evaluation: leave-one-season-out (LOSO) cross-validation across
STANDARD_SEASONS (matches the LOSO convention already used/recommended
elsewhere in this repo -- see docs/contract_model.md). For each held-out
season, train on the other 7 seasons' team-rows and predict the held-out
season's 30 teams; RMSE is computed against the actual NEED_VECTOR values.

Usage:
    python scripts/predict_team_need_from_roster.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from build_team_vectors import PLAY_TYPES, STANDARD_SEASONS

DATA_DIR = Path("data")
PLAYER_VECTOR_DIR = DATA_DIR / "player_vectors_json"
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
TEAM_VECTOR_DIR = DATA_DIR / "team_vectors_json"

NEED_CATEGORIES = PLAY_TYPES + ["RIM_DEFENSE", "PERIMETER_DEFENSE", "REBOUNDING"]

XGB_PARAMS = dict(max_depth=3, reg_alpha=0.1, reg_lambda=1.0, n_estimators=200, random_state=0)
# Inner CV picks 9-13 components in practice (out of 25 features); raising this
# cap to 24 (near-full-rank) changed pooled RMSE by <0.001, so 15 leaves
# comfortable headroom without being an artificial ceiling on the search.
PLS_MAX_COMPONENTS = 15


def load_team_rows(season: str) -> pd.DataFrame:
    """One row per team for `season`: 25 roster-aggregate features (see
    module docstring) plus the 12 actual NEED_VECTOR targets."""
    player_vectors = json.loads((PLAYER_VECTOR_DIR / f"player_vectors_{season}.json").read_text(encoding="utf-8"))
    minutes_by_player = pd.read_csv(PLAYER_STATS_DIR / f"player_stats_{season}.csv", encoding="utf-8").set_index("PLAYER_ID")["MIN"]
    team_needs = json.loads((TEAM_VECTOR_DIR / f"team_vectors_{season}.json").read_text(encoding="utf-8"))

    by_team: dict[str, list[tuple[float, dict]]] = {}
    for player_id_str, entry in player_vectors.items():
        player_id = int(player_id_str)
        if player_id not in minutes_by_player.index:
            continue
        minutes = minutes_by_player.loc[player_id]
        if isinstance(minutes, pd.Series):  # duplicate PLAYER_ID (traded mid-season row) -- sum
            minutes = minutes.sum()
        if not minutes or minutes <= 0:
            continue
        by_team.setdefault(entry["TEAM"], []).append((minutes, entry["VECTOR"]))

    rows = []
    for team, roster in by_team.items():
        if team not in team_needs:
            continue
        mins = np.array([m for m, _ in roster], dtype=float)
        total_min = mins.sum()
        weights = mins / total_min

        row = {"TEAM": team, "SEASON": season, "TOTAL_MIN": total_min}
        for cat in NEED_CATEGORIES:
            vals = np.array([v[cat] for _, v in roster], dtype=float)
            w_mean = np.sum(weights * vals)
            w_std = np.sqrt(np.sum(weights * (vals - w_mean) ** 2))
            row[f"{cat}_WMEAN"] = w_mean
            row[f"{cat}_WSTD"] = w_std
        for cat in NEED_CATEGORIES:
            row[f"TARGET_{cat}"] = team_needs[team]["NEED_VECTOR"][cat]
        rows.append(row)

    return pd.DataFrame(rows)


def build_dataset(seasons: list[str] = STANDARD_SEASONS) -> pd.DataFrame:
    return pd.concat([load_team_rows(s) for s in seasons], ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("TEAM", "SEASON") and not c.startswith("TARGET_")]


def _summarize(records: list[dict], all_preds: dict, all_actual: dict):
    """Shared by every *_loso_cv() below: per-(season, category) RMSE table,
    per-category RMSE pooled across folds, and a single pooled RMSE across
    every (team, season, category) prediction."""
    results = pd.DataFrame(records)

    overall_rows = []
    pooled_sq_errors = []
    for cat in NEED_CATEGORIES:
        preds = np.array(all_preds[cat])
        actual = np.array(all_actual[cat])
        overall_rows.append({
            "CATEGORY": cat,
            "RMSE": float(np.sqrt(np.mean((preds - actual) ** 2))),
            "N": len(preds),
        })
        pooled_sq_errors.extend(((preds - actual) ** 2).tolist())
    overall = pd.DataFrame(overall_rows)
    pooled_rmse = float(np.sqrt(np.mean(pooled_sq_errors)))

    return results, overall, pooled_rmse


def loso_cv(df: pd.DataFrame, seasons: list[str] = STANDARD_SEASONS):
    """Leave-one-season-out CV for the 12-independent-XGBRegressor model.
    Returns (per-fold RMSE table, per-category pooled RMSE table, overall
    pooled RMSE)."""
    feat_cols = feature_columns(df)
    records = []
    all_preds = {cat: [] for cat in NEED_CATEGORIES}
    all_actual = {cat: [] for cat in NEED_CATEGORIES}

    for held_out in seasons:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue

        for cat in NEED_CATEGORIES:
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(train[feat_cols], train[f"TARGET_{cat}"])
            preds = model.predict(test[feat_cols])
            actual = test[f"TARGET_{cat}"].to_numpy()

            rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
            records.append({"SEASON": held_out, "CATEGORY": cat, "RMSE": rmse, "N": len(test)})

            all_preds[cat].extend(preds.tolist())
            all_actual[cat].extend(actual.tolist())

    return _summarize(records, all_preds, all_actual)


def _select_pls_components(X_train: np.ndarray, Y_train: np.ndarray, max_components: int) -> int:
    """Picks n_components via 5-fold CV on the training fold only -- never
    touches the held-out season, so the held-out season stays genuinely
    unseen at both model-fitting and hyperparameter-selection time."""
    upper = min(max_components, X_train.shape[1], len(X_train) - 1)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best_n, best_score = 1, np.inf
    for n in range(1, upper + 1):
        fold_scores = []
        for tr_idx, val_idx in kf.split(X_train):
            model = PLSRegression(n_components=n)
            model.fit(X_train[tr_idx], Y_train[tr_idx])
            pred = model.predict(X_train[val_idx])
            fold_scores.append(np.mean((pred - Y_train[val_idx]) ** 2))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score, best_n = score, n
    return best_n


def pls_loso_cv(df: pd.DataFrame, seasons: list[str] = STANDARD_SEASONS, max_components: int = PLS_MAX_COMPONENTS):
    """Leave-one-season-out CV for a single multi-output PLSRegression model
    (all 12 targets predicted jointly from shared latent components, unlike
    the 12 independent XGBRegressors in loso_cv()). Returns (per-fold RMSE
    table, per-category pooled RMSE table, overall pooled RMSE, chosen
    n_components per fold)."""
    feat_cols = feature_columns(df)
    target_cols = [f"TARGET_{c}" for c in NEED_CATEGORIES]
    records = []
    all_preds = {cat: [] for cat in NEED_CATEGORIES}
    all_actual = {cat: [] for cat in NEED_CATEGORIES}
    chosen_components = []

    for held_out in seasons:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue

        X_train = train[feat_cols].to_numpy()
        Y_train = train[target_cols].to_numpy()
        X_test = test[feat_cols].to_numpy()
        Y_test = test[target_cols].to_numpy()

        n_components = _select_pls_components(X_train, Y_train, max_components)
        chosen_components.append({"SEASON": held_out, "N_COMPONENTS": n_components})

        model = PLSRegression(n_components=n_components)
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)

        for i, cat in enumerate(NEED_CATEGORIES):
            p, a = preds[:, i], Y_test[:, i]
            rmse = float(np.sqrt(np.mean((p - a) ** 2)))
            records.append({"SEASON": held_out, "CATEGORY": cat, "RMSE": rmse, "N": len(test)})
            all_preds[cat].extend(p.tolist())
            all_actual[cat].extend(a.tolist())

    results, overall, pooled_rmse = _summarize(records, all_preds, all_actual)
    return results, overall, pooled_rmse, pd.DataFrame(chosen_components)


def ensemble_predictions(
    df: pd.DataFrame, seasons: list[str] = STANDARD_SEASONS, max_components: int = PLS_MAX_COMPONENTS
) -> pd.DataFrame:
    """Per-fold average of the PLS and XGBoost predictions (simple unweighted
    mean of the two models' out-of-fold NEED_VECTOR predictions) -- cheaper
    than a learned stacker, and a reasonable first thing to try since the two
    models make different kinds of errors: PLS shares information across
    correlated categories via shared latent components, XGBoost fits each
    category's nonlinearities independently. Returns one row per (team,
    season) with both PRED_{cat} and ACTUAL_{cat} for all 12 categories, so
    it can feed both an RMSE summary and a downstream win-pct evaluation."""
    feat_cols = feature_columns(df)
    target_cols = [f"TARGET_{c}" for c in NEED_CATEGORIES]
    rows = []

    for held_out in seasons:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue

        X_train = train[feat_cols].to_numpy()
        Y_train = train[target_cols].to_numpy()
        X_test = test[feat_cols].to_numpy()

        n_components = _select_pls_components(X_train, Y_train, max_components)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, Y_train)
        pls_preds = pls.predict(X_test)

        xgb_preds = np.zeros_like(pls_preds)
        for i, cat in enumerate(NEED_CATEGORIES):
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(train[feat_cols], train[f"TARGET_{cat}"])
            xgb_preds[:, i] = model.predict(test[feat_cols])

        ensemble = (pls_preds + xgb_preds) / 2

        for row_idx, (_, row) in enumerate(test.iterrows()):
            entry = {"SEASON": held_out, "TEAM": row["TEAM"]}
            for i, cat in enumerate(NEED_CATEGORIES):
                entry[f"PRED_{cat}"] = float(ensemble[row_idx, i])
                entry[f"ACTUAL_{cat}"] = float(row[f"TARGET_{cat}"])
            rows.append(entry)

    return pd.DataFrame(rows)


def ensemble_loso_cv(df: pd.DataFrame, seasons: list[str] = STANDARD_SEASONS):
    """RMSE summary (same shape as loso_cv()/pls_loso_cv()) for the
    PLS/XGBoost average -- built on top of ensemble_predictions()."""
    preds_df = ensemble_predictions(df, seasons)
    records = []
    all_preds = {cat: [] for cat in NEED_CATEGORIES}
    all_actual = {cat: [] for cat in NEED_CATEGORIES}

    for season in preds_df["SEASON"].unique():
        season_rows = preds_df[preds_df["SEASON"] == season]
        for cat in NEED_CATEGORIES:
            preds = season_rows[f"PRED_{cat}"].to_numpy()
            actual = season_rows[f"ACTUAL_{cat}"].to_numpy()
            rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
            records.append({"SEASON": season, "CATEGORY": cat, "RMSE": rmse, "N": len(season_rows)})
            all_preds[cat].extend(preds.tolist())
            all_actual[cat].extend(actual.tolist())

    return _summarize(records, all_preds, all_actual)


def naive_baseline_loso(df: pd.DataFrame, seasons: list[str] = STANDARD_SEASONS):
    """Leave-one-season-out 'model': predict the training seasons' mean for
    every team. The floor any real model needs to beat."""
    records = []
    all_preds = {cat: [] for cat in NEED_CATEGORIES}
    all_actual = {cat: [] for cat in NEED_CATEGORIES}

    for held_out in seasons:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue

        for cat in NEED_CATEGORIES:
            pred_value = train[f"TARGET_{cat}"].mean()
            actual = test[f"TARGET_{cat}"].to_numpy()
            preds = np.full_like(actual, pred_value)

            rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
            records.append({"SEASON": held_out, "CATEGORY": cat, "RMSE": rmse, "N": len(test)})

            all_preds[cat].extend(preds.tolist())
            all_actual[cat].extend(actual.tolist())

    return _summarize(records, all_preds, all_actual)


def main():
    df = build_dataset()
    print(f"Built {len(df)} team-season rows across {df['SEASON'].nunique()} seasons "
          f"({len(feature_columns(df))} features, {len(NEED_CATEGORIES)} target categories).")

    naive_per_fold, naive_overall, naive_pooled = naive_baseline_loso(df)
    xgb_per_fold, xgb_overall, xgb_pooled = loso_cv(df)
    pls_per_fold, pls_overall, pls_pooled, pls_components = pls_loso_cv(df)
    ens_per_fold, ens_overall, ens_pooled = ensemble_loso_cv(df)

    comparison = (
        naive_overall.rename(columns={"RMSE": "NAIVE_RMSE"})[["CATEGORY", "NAIVE_RMSE"]]
        .merge(xgb_overall.rename(columns={"RMSE": "XGB_RMSE"})[["CATEGORY", "XGB_RMSE"]], on="CATEGORY")
        .merge(pls_overall.rename(columns={"RMSE": "PLS_RMSE"})[["CATEGORY", "PLS_RMSE"]], on="CATEGORY")
        .merge(ens_overall.rename(columns={"RMSE": "ENSEMBLE_RMSE"})[["CATEGORY", "ENSEMBLE_RMSE"]], on="CATEGORY")
    )
    comparison["XGB_VS_NAIVE_%"] = 100 * (1 - comparison["XGB_RMSE"] / comparison["NAIVE_RMSE"])
    comparison["PLS_VS_NAIVE_%"] = 100 * (1 - comparison["PLS_RMSE"] / comparison["NAIVE_RMSE"])
    comparison["ENSEMBLE_VS_NAIVE_%"] = 100 * (1 - comparison["ENSEMBLE_RMSE"] / comparison["NAIVE_RMSE"])

    print("\nPer-category RMSE -- naive baseline vs. XGBoost vs. PLS vs. Ensemble (pooled across all LOSO folds):")
    print(comparison.sort_values("ENSEMBLE_RMSE").round(4).to_string(index=False))

    print(f"\nPooled RMSE across all (team, season, category) predictions:")
    print(f"  Naive baseline: {naive_pooled:.4f}")
    print(f"  XGBoost:        {xgb_pooled:.4f}  ({100 * (1 - xgb_pooled / naive_pooled):+.1f}% vs. naive)")
    print(f"  PLS:            {pls_pooled:.4f}  ({100 * (1 - pls_pooled / naive_pooled):+.1f}% vs. naive)")
    print(f"  Ensemble:       {ens_pooled:.4f}  ({100 * (1 - ens_pooled / naive_pooled):+.1f}% vs. naive)")

    print(f"\nMacro-average RMSE across 12 categories:")
    print(f"  Naive baseline: {naive_overall['RMSE'].mean():.4f}")
    print(f"  XGBoost:        {xgb_overall['RMSE'].mean():.4f}")
    print(f"  PLS:            {pls_overall['RMSE'].mean():.4f}")
    print(f"  Ensemble:       {ens_overall['RMSE'].mean():.4f}")

    print("\nPLS n_components chosen per LOSO fold (via inner 5-fold CV on that fold's training rows):")
    print(pls_components.to_string(index=False))


if __name__ == "__main__":
    main()
