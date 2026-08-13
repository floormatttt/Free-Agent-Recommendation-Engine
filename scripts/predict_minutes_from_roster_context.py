"""
Predicts a player's minutes per game (MPG) from their own 12-category skill
vector (data/player_vectors_json/) plus a fixed-size summary of their
teammates' vectors (same team-season, self excluded, minutes-weighted) --
i.e. "does this player's skill set, in the context of who else is on this
roster, predict how many minutes they actually played?"

Feature engineering (mirrors predict_team_need_from_roster.py's roster ->
fixed-size-vector convention, applied at the player level instead of the
team level):
  - the player's own 12 VECTOR categories
  - teammates' minutes-weighted mean of each of the 12 categories (self
    excluded)
  - teammates' minutes-weighted std of each of the 12 categories (roster
    depth/spread around the player)
  - teammate count and total teammate minutes (roster size / competition
    for minutes)
-> 27 features per player-season.

Target: MIN_PER_GAME = MIN / GP (data/player_stats/).

Players below MIN_TOTAL_MINUTES (1000) in a given season are dropped
entirely for that season -- both as prediction targets and as teammate
context for their team's other rows -- rather than gated by games played,
so a player who racked up garbage-time appearances across many GP without
real minutes is filtered the same as one who missed most of the season to
injury.

Model: single XGBRegressor (one scalar target, unlike the 12-category team
model this mirrors). Evaluated via leave-one-season-out CV across every
season in STANDARD_SEASONS -- one held-out season at a time, retrained from
scratch on the other 7 each fold -- same convention as
predict_team_need_from_roster.py and docs/contract_model.md, against a
naive baseline (predict the training seasons' mean MPG for everyone).

Output: prints LOSO RMSE/MAE vs. naive baseline, and writes a predicted-vs-
actual scatter plot (out-of-fold predictions, pooled across all seasons) to
data/minutes_prediction/predicted_vs_actual_mpg.png.

Usage:
    python scripts/predict_minutes_from_roster_context.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from build_team_vectors import STANDARD_SEASONS

DATA_DIR = Path("data")
PLAYER_VECTOR_DIR = DATA_DIR / "player_vectors_json"
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
OUTPUT_DIR = DATA_DIR / "minutes_prediction"

VECTOR_CATEGORIES = [
    "ISOLATION", "PNR_HANDLER", "PNR_ROLLMAN", "POST_UP", "SPOT_UP",
    "HANDOFF", "CUT", "OFF_SCREEN", "PUT_BACKS",
    "RIM_DEFENSE", "PERIMETER_DEFENSE", "REBOUNDING",
]

XGB_PARAMS = dict(max_depth=3, reg_alpha=0.1, reg_lambda=1.0, n_estimators=200, random_state=0)
MIN_TOTAL_MINUTES = 1000  # drop players below this season minutes total (both as targets and as teammate context) -- cuts garbage-time/injury-shortened noise more directly than a GP floor


def load_player_rows(
    season: str,
    target_min_minutes: float = MIN_TOTAL_MINUTES,
    context_min_minutes: float = MIN_TOTAL_MINUTES,
) -> pd.DataFrame:
    """One row per player-season: own vector (12) + teammate-context features
    (15) + target MIN_PER_GAME.

    Two independent floors, both in season-total minutes:
      - context_min_minutes: who counts as "a teammate" for the weighted
        context features -- keeping this at MIN_TOTAL_MINUTES means context
        is always built from real rotation players, regardless of the
        target floor below.
      - target_min_minutes: who gets a *row* (i.e. whose minutes the model
        is trained to predict). Defaults match the module's own LOSO
        validation (rotation-only in, rotation-only out). Forward-inference
        callers that need to score a full roster -- bench players included
        -- should pass target_min_minutes=0 so the model actually sees
        low-minute outcomes during training instead of extrapolating a
        rotation-only mapping onto players it never learned a "low minutes"
        case for (see predict_2026_27_minutes.py)."""
    player_vectors = json.loads((PLAYER_VECTOR_DIR / f"player_vectors_{season}.json").read_text(encoding="utf-8"))
    stats = pd.read_csv(PLAYER_STATS_DIR / f"player_stats_{season}.csv", encoding="utf-8")
    stats = stats.groupby("PLAYER_ID", as_index=True)[["GP", "MIN"]].sum()  # traded mid-season -> combine rows

    by_team: dict[str, list[tuple[int, float, dict]]] = {}
    for player_id_str, entry in player_vectors.items():
        player_id = int(player_id_str)
        if player_id not in stats.index:
            continue
        minutes, gp = stats.loc[player_id, "MIN"], stats.loc[player_id, "GP"]
        if minutes <= 0 or gp <= 0:
            continue
        by_team.setdefault(entry["TEAM"], []).append((player_id, minutes, entry["VECTOR"]))

    rows = []
    for team, roster in by_team.items():
        context_roster = [p for p in roster if p[1] >= context_min_minutes]
        for player_id, minutes, vector in roster:
            if minutes < target_min_minutes:
                continue
            teammates = [(m, v) for pid, m, v in context_roster if pid != player_id]
            if not teammates:
                continue  # need at least one teammate for context features
            t_mins = np.array([m for m, _ in teammates], dtype=float)
            t_weights = t_mins / t_mins.sum()

            row = {"PLAYER_ID": player_id, "TEAM": team, "SEASON": season}
            for cat in VECTOR_CATEGORIES:
                row[f"OWN_{cat}"] = vector[cat]
            for cat in VECTOR_CATEGORIES:
                vals = np.array([v[cat] for _, v in teammates], dtype=float)
                w_mean = np.sum(t_weights * vals)
                w_std = np.sqrt(np.sum(t_weights * (vals - w_mean) ** 2))
                row[f"TEAMMATE_{cat}_WMEAN"] = w_mean
                row[f"TEAMMATE_{cat}_WSTD"] = w_std
            row["TEAMMATE_COUNT"] = len(teammates)
            row["TEAMMATE_TOTAL_MIN"] = float(t_mins.sum())
            gp = stats.loc[player_id, "GP"]
            row["TARGET_MPG"] = float(minutes) / float(gp)
            rows.append(row)

    return pd.DataFrame(rows)


def build_dataset(
    seasons: list[str] = STANDARD_SEASONS,
    target_min_minutes: float = MIN_TOTAL_MINUTES,
    context_min_minutes: float = MIN_TOTAL_MINUTES,
) -> pd.DataFrame:
    return pd.concat(
        [load_player_rows(s, target_min_minutes, context_min_minutes) for s in seasons],
        ignore_index=True,
    )


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("PLAYER_ID", "TEAM", "SEASON", "TARGET_MPG")]


def loso_predict(df: pd.DataFrame, seasons: list[str] = STANDARD_SEASONS) -> pd.DataFrame:
    """Out-of-fold predictions for every player-season, one row per player
    holding out its own season's model."""
    feat_cols = feature_columns(df)
    out = []
    for held_out in seasons:
        train = df[df["SEASON"] != held_out]
        test = df[df["SEASON"] == held_out]
        if train.empty or test.empty:
            continue

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(train[feat_cols], train["TARGET_MPG"])
        preds = model.predict(test[feat_cols])

        naive_pred = train["TARGET_MPG"].mean()

        fold = test[["PLAYER_ID", "TEAM", "SEASON", "TARGET_MPG"]].copy()
        fold["PRED_MPG"] = preds
        fold["NAIVE_PRED_MPG"] = naive_pred
        out.append(fold)

    return pd.concat(out, ignore_index=True)


def attach_names(preds: pd.DataFrame) -> pd.DataFrame:
    names = {}
    for season in preds["SEASON"].unique():
        vectors = json.loads((PLAYER_VECTOR_DIR / f"player_vectors_{season}.json").read_text(encoding="utf-8"))
        for pid_str, entry in vectors.items():
            names[(int(pid_str), season)] = entry["PLAYER_NAME"]
    preds = preds.copy()
    preds["PLAYER_NAME"] = [names.get((pid, s), "") for pid, s in zip(preds["PLAYER_ID"], preds["SEASON"])]
    return preds


def summarize(preds: pd.DataFrame) -> None:
    err = preds["PRED_MPG"] - preds["TARGET_MPG"]
    naive_err = preds["NAIVE_PRED_MPG"] - preds["TARGET_MPG"]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    naive_rmse = float(np.sqrt(np.mean(naive_err ** 2)))
    naive_mae = float(np.mean(np.abs(naive_err)))

    print(f"Built {len(preds)} player-season rows across {preds['SEASON'].nunique()} seasons.")
    print(f"\nPooled LOSO RMSE/MAE (minutes per game):")
    print(f"  Naive baseline: RMSE {naive_rmse:.2f}  MAE {naive_mae:.2f}")
    print(f"  XGBoost:        RMSE {rmse:.2f}  MAE {mae:.2f}  "
          f"({100 * (1 - rmse / naive_rmse):+.1f}% RMSE vs. naive)")

    print("\nPer-season RMSE:")
    for season in STANDARD_SEASONS:
        sub = preds[preds["SEASON"] == season]
        if sub.empty:
            continue
        season_rmse = float(np.sqrt(np.mean((sub["PRED_MPG"] - sub["TARGET_MPG"]) ** 2)))
        print(f"  {season}: {season_rmse:.2f}  (n={len(sub)})")


def plot_predicted_vs_actual(preds: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    lo, hi = 0, max(preds["TARGET_MPG"].max(), preds["PRED_MPG"].max()) + 1
    ax.plot([lo, hi], [lo, hi], color="#9AA5B1", linewidth=1, linestyle="--", zorder=1, label="Perfect prediction")
    ax.scatter(
        preds["TARGET_MPG"], preds["PRED_MPG"],
        s=14, alpha=0.35, color="#3B82F6", edgecolors="none", zorder=2,
    )

    err = (preds["PRED_MPG"] - preds["TARGET_MPG"]).abs()
    top_outliers = preds.loc[err.sort_values(ascending=False).index[:8]]
    for _, r in top_outliers.iterrows():
        label = r["PLAYER_NAME"] or str(r["PLAYER_ID"])
        ax.annotate(
            f"{label} ({r['SEASON']})",
            (r["TARGET_MPG"], r["PRED_MPG"]),
            fontsize=7, color="#4B5563",
            xytext=(4, 4), textcoords="offset points",
        )

    rmse = float(np.sqrt(np.mean((preds["PRED_MPG"] - preds["TARGET_MPG"]) ** 2)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual minutes per game")
    ax.set_ylabel("Predicted minutes per game")
    ax.set_title(f"Predicted vs. actual MPG (own vector + teammate context)\nLOSO CV, pooled across "
                 f"{preds['SEASON'].nunique()} seasons -- RMSE {rmse:.2f}")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"\nSaved plot to {output_path}")


def main():
    df = build_dataset()
    preds = loso_predict(df)
    preds = attach_names(preds)
    summarize(preds)
    plot_predicted_vs_actual(preds, OUTPUT_DIR / "predicted_vs_actual_mpg.png")


if __name__ == "__main__":
    main()
