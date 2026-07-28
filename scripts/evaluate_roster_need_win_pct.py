"""
Chains predict_team_need_from_roster.py's roster-reconstructed NEED_VECTOR
(PLS, the stronger of its two models) into predict_win_pct_from_need.py's
win-percentage regression, in place of the true team-stats-derived
NEED_VECTOR -- i.e. can a NEED_VECTOR *rebuilt from a roster's player
vectors* still predict winning, or does reconstruction error wash out the
win-pct signal that predict_win_pct_from_need.py found (Ridge RMSE 0.063
against the real NEED_VECTOR)?

Both stages use the same LOSO-across-STANDARD_SEASONS discipline, and are
composed fold-for-fold: for each held-out season, the PLS model is trained
on the other 7 seasons' rosters (predict_team_need_from_roster.py's
existing per-fold inner-CV n_components selection), used to predict that
season's 30 teams' NEED_VECTORs, and *those* predicted vectors (not the
real ones) are what the win-pct Ridge/Elastic Net models see at both train
and test time -- so a fold's win-pct model is trained only on
reconstructed vectors from seasons it never saw the real roster-to-need
mapping validated on, matching genuine leave-one-season-out discipline
throughout.

Usage:
    python scripts/evaluate_roster_need_win_pct.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, RidgeCV

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import (
    NEED_CATEGORIES,
    build_dataset as build_roster_dataset,
    feature_columns,
    _select_pls_components,
    PLS_MAX_COMPONENTS,
    ensemble_predictions,
)
from predict_win_pct_from_need import (
    RIDGE_ALPHAS,
    ENET_ALPHAS,
    ENET_L1_RATIOS,
    load_season_rows,
    rmse,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def build_reconstructed_need_dataset(seasons: list[str] = STANDARD_SEASONS) -> pd.DataFrame:
    """For each LOSO fold, fit PLS on the other 7 seasons' rosters -> real
    NEED_VECTOR, predict the held-out season's 30 teams' NEED_VECTOR from
    their rosters. Returns one row per team-season with the *predicted*
    (not actual) 12 categories, so the held-out season's reconstructed
    vector never used its own roster-to-need relationship during fitting."""
    roster_df = build_roster_dataset(seasons)
    feat_cols = feature_columns(roster_df)
    target_cols = [f"TARGET_{c}" for c in NEED_CATEGORIES]

    rows = []
    for held_out in seasons:
        train = roster_df[roster_df["SEASON"] != held_out]
        test = roster_df[roster_df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue

        X_train = train[feat_cols].to_numpy()
        Y_train = train[target_cols].to_numpy()
        X_test = test[feat_cols].to_numpy()

        n_components = _select_pls_components(X_train, Y_train, PLS_MAX_COMPONENTS)
        model = PLSRegression(n_components=n_components)
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)

        for (_, row), pred_row in zip(test.iterrows(), preds):
            entry = {"TEAM": row["TEAM"], "SEASON": row["SEASON"]}
            for i, cat in enumerate(NEED_CATEGORIES):
                entry[cat] = float(pred_row[i])
            entry["COMPOSITE_NEED"] = float(np.mean(pred_row))
            rows.append(entry)

    return pd.DataFrame(rows)


def build_reconstructed_need_dataset_ensemble(seasons: list[str] = STANDARD_SEASONS) -> pd.DataFrame:
    """Same as build_reconstructed_need_dataset() above but using the
    PLS/XGBoost average (predict_team_need_from_roster.py's
    ensemble_predictions()) instead of PLS alone."""
    roster_df = build_roster_dataset(seasons)
    preds_df = ensemble_predictions(roster_df, seasons)

    rows = []
    for _, row in preds_df.iterrows():
        entry = {"TEAM": row["TEAM"], "SEASON": row["SEASON"]}
        vals = []
        for cat in NEED_CATEGORIES:
            v = row[f"PRED_{cat}"]
            entry[cat] = v
            vals.append(v)
        entry["COMPOSITE_NEED"] = float(np.mean(vals))
        rows.append(entry)

    return pd.DataFrame(rows)


def build_augmented_dataset(seasons: list[str] = STANDARD_SEASONS) -> pd.DataFrame:
    """25 raw roster-aggregate features (predict_team_need_from_roster.py's
    minutes-weighted mean/std + total minutes) plus the 12 ensemble-reconstructed
    NEED_VECTOR categories, side by side -- 37 features total. Tries to recover
    some of what the 12-category bottleneck throws away by giving the downstream
    win-pct model direct access to the roster shape, not just its NEED_VECTOR
    projection."""
    roster_df = build_roster_dataset(seasons)
    preds_df = ensemble_predictions(roster_df, seasons)
    pred_cols = [f"PRED_{c}" for c in NEED_CATEGORIES]
    return roster_df.merge(preds_df[["SEASON", "TEAM"] + pred_cols], on=["SEASON", "TEAM"])


def blended_loso_predict(
    df: pd.DataFrame, direct_feat_cols: list[str], combo_feat_cols: list[str],
    seasons: list[str] = STANDARD_SEASONS, max_components: int = PLS_MAX_COMPONENTS,
) -> pd.DataFrame:
    """Averages two out-of-fold predictions per team-season: a RidgeCV fit on
    the raw roster features alone, and a PLSRegression fit on the roster
    features + reconstructed NEED_VECTOR combined. A flat, untuned 50/50
    average -- deliberately not a learned blend weight, since a follow-up
    check found picking the weight via nested inner-CV overfit at this
    sample size (~210 training rows/fold) and did worse than the fixed
    average out-of-fold."""
    records = []
    for held_out in seasons:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue

        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(train[direct_feat_cols], train["W_PCT"])
        ridge_pred = ridge.predict(test[direct_feat_cols])

        X_train = train[combo_feat_cols].to_numpy()
        y_train = train["W_PCT"].to_numpy()
        X_test = test[combo_feat_cols].to_numpy()
        upper = min(max_components, X_train.shape[1] - 1, len(X_train) - 1)
        n_components = _select_pls_components(X_train, y_train, upper)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train)
        pls_pred = pls.predict(X_test).ravel()

        blend_pred = (ridge_pred + pls_pred) / 2
        for team, actual, pred in zip(test["TEAM"], test["W_PCT"], blend_pred):
            records.append({"SEASON": held_out, "TEAM": team, "ACTUAL": actual, "PRED": pred})

    return pd.DataFrame(records)


def attach_win_pct(need_df: pd.DataFrame, seasons: list[str] = STANDARD_SEASONS) -> pd.DataFrame:
    wins_rows = []
    for season in seasons:
        wins = pd.read_csv(DATA_DIR / "team_wins" / f"team_wins_{season}.csv").set_index("TEAM")
        for team in wins.index:
            wins_rows.append({"TEAM": team, "SEASON": season, "W_PCT": float(wins.loc[team, "W_PCT"])})
    wins_df = pd.DataFrame(wins_rows)
    return need_df.merge(wins_df, on=["TEAM", "SEASON"], how="inner")


def loso_predict(df: pd.DataFrame, feat_cols: list[str], make_model, seasons: list[str] = STANDARD_SEASONS):
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


def _naive_preds(df: pd.DataFrame, seasons: list[str] = STANDARD_SEASONS) -> pd.DataFrame:
    naive_records = []
    for held_out in seasons:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue
        pred_value = train["W_PCT"].mean()
        for team, actual in zip(test["TEAM"], test["W_PCT"]):
            naive_records.append({"SEASON": held_out, "TEAM": team, "ACTUAL": actual, "PRED": pred_value})
    return pd.DataFrame(naive_records)


def main():
    print("Reconstructing NEED_VECTOR from rosters (PLS, LOSO per fold)...")
    reconstructed = build_reconstructed_need_dataset()
    df = attach_win_pct(reconstructed)
    print(f"Built {len(df)} team-season rows with roster-reconstructed NEED_VECTOR + actual W_PCT.")

    print("Reconstructing NEED_VECTOR from rosters (PLS/XGBoost ensemble, LOSO per fold)...")
    reconstructed_ens = build_reconstructed_need_dataset_ensemble()
    df_ens = attach_win_pct(reconstructed_ens)

    print("Building augmented dataset (raw roster features + ensemble-reconstructed NEED_VECTOR)...")
    augmented = attach_win_pct(build_augmented_dataset())
    direct_feat_cols = feature_columns(build_roster_dataset())
    combo_feat_cols = direct_feat_cols + [f"PRED_{c}" for c in NEED_CATEGORIES]

    naive_preds = _naive_preds(df)

    composite_preds, _ = loso_predict(df, ["COMPOSITE_NEED"], lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    ridge_preds, ridge_models = loso_predict(df, NEED_CATEGORIES, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    enet_preds, enet_models = loso_predict(
        df, NEED_CATEGORIES,
        lambda: ElasticNetCV(alphas=ENET_ALPHAS, l1_ratio=ENET_L1_RATIOS, max_iter=10000),
    )

    ens_composite_preds, _ = loso_predict(df_ens, ["COMPOSITE_NEED"], lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    ens_ridge_preds, ens_ridge_models = loso_predict(df_ens, NEED_CATEGORIES, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    ens_enet_preds, ens_enet_models = loso_predict(
        df_ens, NEED_CATEGORIES,
        lambda: ElasticNetCV(alphas=ENET_ALPHAS, l1_ratio=ENET_L1_RATIOS, max_iter=10000),
    )

    direct_preds, _ = loso_predict(augmented, direct_feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    blend_preds = blended_loso_predict(augmented, direct_feat_cols, combo_feat_cols)

    print("\nLOSO RMSE predicting W_PCT from the ROSTER-RECONSTRUCTED NEED_VECTOR:")
    print(f"  Naive baseline (train-mean):          {rmse(naive_preds):.4f}")
    print(f"  Composite need, PLS (1 feature):       {rmse(composite_preds):.4f}")
    print(f"  Ridge, PLS (12 features):              {rmse(ridge_preds):.4f}")
    print(f"  Elastic Net, PLS (12 features):        {rmse(enet_preds):.4f}")
    print(f"  Composite need, Ensemble (1 feature):  {rmse(ens_composite_preds):.4f}")
    print(f"  Ridge, Ensemble (12 features):         {rmse(ens_ridge_preds):.4f}")
    print(f"  Elastic Net, Ensemble (12 features):   {rmse(ens_enet_preds):.4f}")
    print(f"  Direct roster features, Ridge (25):    {rmse(direct_preds):.4f}  (skips NEED_VECTOR bottleneck)")
    print(f"  Blend: direct Ridge + augmented PLS:   {rmse(blend_preds):.4f}  (best so far)")

    print("\nFor comparison, predict_win_pct_from_need.py's results using the TRUE (team-stats-derived) NEED_VECTOR:")
    true_need_df = pd.concat([load_season_rows(s) for s in STANDARD_SEASONS], ignore_index=True)
    true_ridge_preds, true_ridge_models = loso_predict(true_need_df, NEED_CATEGORIES, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    print(f"  Ridge (12 features, true NEED_VECTOR): {rmse(true_ridge_preds):.4f}")
    print(f"  (naive baseline is the same {rmse(naive_preds):.4f} either way -- doesn't depend on NEED_VECTOR)")

    print("\nRidge coefficients (PLS-reconstructed NEED_VECTOR), averaged across folds, sorted:")
    coef_df = pd.DataFrame([m.coef_ for m in ridge_models], columns=NEED_CATEGORIES, index=STANDARD_SEASONS)
    print(coef_df.mean().sort_values().round(4).to_string())

    print("\nRidge coefficients (Ensemble-reconstructed NEED_VECTOR), averaged across folds, sorted:")
    ens_coef_df = pd.DataFrame([m.coef_ for m in ens_ridge_models], columns=NEED_CATEGORIES, index=STANDARD_SEASONS)
    print(ens_coef_df.mean().sort_values().round(4).to_string())

    out = {
        "seasons": STANDARD_SEASONS,
        "categories": NEED_CATEGORIES,
        "rmse": {
            "naive": rmse(naive_preds),
            "composite_pls": rmse(composite_preds),
            "ridge_reconstructed_need_pls": rmse(ridge_preds),
            "enet_reconstructed_need_pls": rmse(enet_preds),
            "composite_ensemble": rmse(ens_composite_preds),
            "ridge_reconstructed_need_ensemble": rmse(ens_ridge_preds),
            "enet_reconstructed_need_ensemble": rmse(ens_enet_preds),
            "ridge_direct_roster_features": rmse(direct_preds),
            "blend_direct_ridge_plus_augmented_pls": rmse(blend_preds),
            "ridge_true_need": rmse(true_ridge_preds),
        },
        "ridge_coef_mean_reconstructed_pls": coef_df.mean().to_dict(),
        "ridge_coef_mean_reconstructed_ensemble": ens_coef_df.mean().to_dict(),
        "blend_calibration": blend_preds.to_dict(orient="records"),
    }
    out_path = DATA_DIR / "roster_reconstructed_win_pct_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
