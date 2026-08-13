"""
Predicts minutes per game for the 2026-27 season using the actual 2026-27
rosters (data/rosters/rosters_2026-27.csv, pulled via pull_rosters.py)
instead of a held-out historical season -- genuine forward inference, not a
back-test.

The 2026-27 season hasn't happened yet, so there's no 2026-27 skill vector
or teammate-minutes data to build features from. Instead, each player is
assigned their most recent available skill vector (data/player_vectors_json/,
newest STANDARD_SEASON they appear in) and, for teammate weighting, their
most recent available season-total minutes (data/player_stats/) as a proxy
for "how much this player tends to play" -- the same role TEAMMATE_*_WMEAN/
WSTD's minutes-weighting plays in training, just sourced from last season
instead of the (unknown) current one. Players with no historical vector at
all (rookies, first NBA season) are assigned a replacement-level vector
(build_replacement_vector(): the average skill vector across real deep-bench
player-seasons, REPLACEMENT_MIN_RANGE minutes) instead of being dropped --
they still enter each team's 240 MPG budget and get a real (if generic)
prediction, flagged via REASON. Dropping them instead left their minutes
unaccounted for, which inflated every teammate's share of the fixed 240
budget on rookie-heavy rosters. A team that still has zero rotation-caliber
historical players at all (context is empty) is the one case still reported
as genuinely unscorable.

The production model here is trained on ALL 8 STANDARD_SEASONS at once
(unlike predict_minutes_from_roster_context.py's leave-one-season-out CV,
which exists to measure accuracy on already-known outcomes) -- there's
nothing to hold out when predicting a season with no ground truth yet, so
using every available season maximizes signal for the real prediction.

Unlike predict_minutes_from_roster_context.py's own reported model (targets
AND context both restricted to players with >=1000 season minutes), this
script trains on EVERY player-season as a target (target_min_minutes=0) --
scoring full 2026-27 rosters means predicting for bench/two-way players too,
and a model that only ever saw rotation-caliber targets during training has
no low-minutes examples to anchor on, so it extrapolates a rotation-level
floor onto everyone regardless of skill. Teammate context still stays
rotation-only (context_min_minutes defaults to MIN_TOTAL_MINUTES), both
during training and when scoring 2026-27 rosters (build_2026_27_rows()) --
a bench player's context should reflect the real rotation players around
them, not other bench players.

After the model's raw predictions are quantile-mapped onto the real
historical MPG distribution's shape (realistic starter/rotation/bench
tiers), each team's calibrated predictions are rescaled proportionally to
sum to exactly 240 min/game (5 players x 48 min) -- summing a full ~17-man
scored roster's individual-season MPG naturally runs well above 240 (mean
~329 pre-rescale) since a full roster's reserves all contribute some
minutes, same as any real team's would. PRED_MPG_UNSCALED keeps the
pre-rescale numbers visible alongside the rescaled PRED_MPG.

Output: data/2026_27_projections/multi_year_minutes_predictions.csv, one row per 2026-27 roster player: TEAM,
PLAYER_NAME, PLAYER_ID, PRED_MPG, PRED_MPG_UNSCALED (both blank if no
historical vector), and the season each player's vector/minutes-weight was
sourced from.

Usage:
    python scripts/predict_2026_27_minutes.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from predict_minutes_from_roster_context import (
    VECTOR_CATEGORIES, XGB_PARAMS, PLAYER_VECTOR_DIR, PLAYER_STATS_DIR,
    STANDARD_SEASONS, MIN_TOTAL_MINUTES, build_dataset, feature_columns,
)

ROSTER_PATH = Path("data/rosters/rosters_2026-27.csv")
OUTPUT_PATH = Path("data/2026_27_projections/multi_year_minutes_predictions.csv")
TEAM_TOTAL_MPG = 240.0  # 5 players x 48 minutes/game (ignores OT)
# Season-minutes band used to define "replacement level" for rookies: high
# enough to exclude single garbage-time cameos (noisy, unrepresentative
# vectors), low enough to stay clear of MIN_TOTAL_MINUTES's rotation
# cutoff (1000) -- this is meant to capture genuine deep-bench/two-way
# roles, not fringe rotation players.
REPLACEMENT_MIN_RANGE = (200.0, 800.0)
# Exponent for rescale_to_team_total()'s power-law reallocation: > 1 gives
# higher-calibrated-MPG players a bigger share of the 240 budget, < 1 would
# flatten toward an even split. 1.6 was picked empirically (see the module's
# rescale_to_team_total() docstring) as the highest value that keeps every
# 2026-27 roster's single-highest rescaled player under ~40 MPG -- a real,
# if rare, single-season iron-man ceiling -- while still meaningfully
# lifting star minutes over a flat proportional split (league-wide top-5
# average 22.7 -> 26.8 MPG). A sigmoid-shaped alternative was tried and
# reverted at the user's request -- it clustered a team's top few players
# too close together near a shared ceiling (e.g. Denver's Murray/Jokic/
# Johnson/Gordon all landing within ~2.6 MPG of each other, 29.3-26.7,
# despite a real ~10 MPG gap in their calibrated values) instead of letting
# a genuine top scorer separate from "very good" teammates.
REALLOCATION_EXPONENT = 1.6


def rescale_to_team_total(
    df: pd.DataFrame, value_col: str, total: float = TEAM_TOTAL_MPG, exponent: float = REALLOCATION_EXPONENT,
) -> pd.Series:
    """Rescales `value_col` within each TEAM so the team sums to `total`,
    via a power-law reallocation (weight = value ** exponent, then
    normalized to sum to `total`) instead of a flat proportional split.

    A flat proportional rescale (every player multiplied by the same
    team-level factor) discounts a star at the same rate as a deep bench
    player -- e.g. Jokic's calibrated ~35.8 MPG got cut to ~23.1 to make
    Denver's full 17-man roster fit 240, which is backwards: in a real deep
    rotation it's bench minutes that get squeezed to make room, not a
    franchise player's. exponent > 1 gives higher-value players a
    disproportionately larger share of the fixed budget, so stars keep most
    of their realistic calibrated minutes while the bench absorbs most of
    the compression. An earlier attempt at a hard "protect the top 7, rescale
    only the rest" version created an ugly cliff instead -- Denver's 8th
    player dropped from a calibrated ~23.5 to ~2.4 MPG, since a deep team's
    top 7 alone can already sum close to 240, leaving almost nothing for
    the remaining ~10 players. The power law degrades smoothly instead: no
    single rank boundary where everyone below it collapses."""
    weights = df.groupby("TEAM")[value_col].transform(lambda s: s.clip(lower=0.1) ** exponent)
    weight_totals = df.groupby("TEAM")[value_col].transform(
        lambda s: (s.clip(lower=0.1) ** exponent).sum()
    )
    return weights / weight_totals * total


def quantile_map_to_target(raw_preds: np.ndarray, target_distribution: np.ndarray) -> np.ndarray:
    """Rank-based quantile mapping: the k-th smallest of `raw_preds` becomes
    the k-th percentile of `target_distribution` -- forces the *scored*
    population's shape to match the real one exactly by construction.

    XGBoost (like most tree ensembles trained on a few thousand rows) is
    under-dispersed on its own -- its predictions cluster toward the middle
    more than real MPG outcomes do -- but a first attempt at fixing that via
    a fixed value-interpolation curve (built from out-of-fold LOSO
    predictions on historical rows) under-corrected: the raw model's output
    on 2026-27 rosters runs ~1.4 MPG higher on average than its raw output
    on the historical training rows the curve was built from (mean 20.8 vs.
    19.4, verified by comparing model.predict() on both), since the 508
    scored 2026-27 players are already pre-filtered to established,
    non-rookie talent -- so a fixed curve built on a different population's
    raw-score range let too many of those shifted-up values land in the
    curve's high tail. Ranking within the population actually being scored,
    rather than matching against an absolute raw-value curve, is immune to
    that shift: whatever the model's raw scale/shift happens to be for a
    given roster, the calibrated output's proportions above/below any
    threshold are pinned to the real distribution's proportions."""
    ranks = pd.Series(raw_preds).rank(method="average", pct=True).to_numpy()
    return np.quantile(target_distribution, ranks)


def build_latest_vectors(seasons: list[str] = STANDARD_SEASONS) -> dict[int, tuple[dict, str]]:
    """player_id -> (VECTOR, season) using each player's most recent
    STANDARD_SEASONS appearance (later seasons overwrite earlier ones)."""
    latest: dict[int, tuple[dict, str]] = {}
    for season in seasons:  # ascending order -> later writes win
        vectors = json.loads((PLAYER_VECTOR_DIR / f"player_vectors_{season}.json").read_text(encoding="utf-8"))
        for pid_str, entry in vectors.items():
            latest[int(pid_str)] = (entry["VECTOR"], season)
    return latest


def build_latest_minutes(seasons: list[str] = STANDARD_SEASONS) -> dict[int, tuple[float, str]]:
    """player_id -> (total MIN, season) using each player's most recent
    STANDARD_SEASONS appearance -- the proxy weight for teammate context
    since 2026-27 minutes don't exist yet."""
    latest: dict[int, tuple[float, str]] = {}
    for season in seasons:
        stats = pd.read_csv(PLAYER_STATS_DIR / f"player_stats_{season}.csv", encoding="utf-8")
        totals = stats.groupby("PLAYER_ID")["MIN"].sum()
        for pid, minutes in totals.items():
            if minutes and minutes > 0:
                latest[int(pid)] = (float(minutes), season)
    return latest


def build_replacement_vector(seasons: list[str] = STANDARD_SEASONS) -> dict[str, float]:
    """Average skill vector across real deep-bench player-seasons
    (REPLACEMENT_MIN_RANGE season minutes) -- the stand-in vector assigned
    to rookies (no historical vector of their own) so they still enter the
    model and each team's 240 MPG budget instead of silently vanishing
    from it (see module docstring)."""
    lo, hi = REPLACEMENT_MIN_RANGE
    sums = {cat: 0.0 for cat in VECTOR_CATEGORIES}
    count = 0
    for season in seasons:
        stats = pd.read_csv(PLAYER_STATS_DIR / f"player_stats_{season}.csv", encoding="utf-8")
        totals = stats.groupby("PLAYER_ID")["MIN"].sum()
        band_ids = set(totals[(totals >= lo) & (totals <= hi)].index.astype(int))
        vectors = json.loads((PLAYER_VECTOR_DIR / f"player_vectors_{season}.json").read_text(encoding="utf-8"))
        for pid_str, entry in vectors.items():
            if int(pid_str) in band_ids:
                for cat in VECTOR_CATEGORIES:
                    sums[cat] += entry["VECTOR"][cat]
                count += 1
    if count == 0:
        raise ValueError(f"no player-seasons found in REPLACEMENT_MIN_RANGE={REPLACEMENT_MIN_RANGE}")
    return {cat: sums[cat] / count for cat in VECTOR_CATEGORIES}


def build_2026_27_rows(
    latest_vectors: dict, latest_minutes: dict, replacement_vector: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (scorable_rows, unscorable_rows). Rookies (no historical
    vector on file) are scored using replacement_vector rather than being
    dropped -- flagged via REASON on the returned scorable row. A team left
    with zero rotation-caliber historical players at all is the one case
    still reported as genuinely unscorable."""
    rosters = pd.read_csv(ROSTER_PATH, encoding="utf-8")

    by_team: dict[str, list[dict]] = {}
    for _, r in rosters.iterrows():
        by_team.setdefault(r["TEAM"], []).append({
            "PLAYER_ID": int(r["PLAYER_ID"]),
            "PLAYER_NAME": r["PLAYER_NAME"],
        })

    scorable, unscorable = [], []
    for team, roster in by_team.items():
        known = []
        for p in roster:
            pid = p["PLAYER_ID"]
            if pid in latest_vectors:
                vector, v_season = latest_vectors[pid]
                minutes, m_season = latest_minutes.get(pid, (0.0, None))
                known.append({**p, "VECTOR": vector, "VECTOR_SEASON": v_season,
                              "MIN": max(minutes, 1.0), "MIN_SEASON": m_season, "REASON": ""})
            else:
                # rookie/no NBA data: replacement-level vector, and MIN stays
                # below MIN_TOTAL_MINUTES so they're never used as someone
                # else's rotation-caliber teammate context.
                known.append({**p, "VECTOR": replacement_vector, "VECTOR_SEASON": "replacement-level",
                              "MIN": 1.0, "MIN_SEASON": None,
                              "REASON": "rookie/no NBA data (replacement-level vector)"})

        # teammate context is restricted to rotation-caliber players (>= MIN_TOTAL_MINUTES
        # last known season), matching how the production model's context features were
        # trained (predict_minutes_from_roster_context.load_player_rows' context_min_minutes)
        # -- otherwise a bench player's own weak context would look artificially strong just
        # from averaging in other bench players, which the model never saw during training.
        context = [q for q in known if q["MIN"] >= MIN_TOTAL_MINUTES]
        if not context:
            for p in known:
                unscorable.append({**p, "TEAM": team, "REASON": "team has no rotation-caliber historical players"})
            continue

        for p in known:
            teammates = [q for q in context if q["PLAYER_ID"] != p["PLAYER_ID"]]
            if not teammates:
                unscorable.append({**p, "TEAM": team, "REASON": "no rotation-caliber teammates with historical data"})
                continue
            t_mins = np.array([q["MIN"] for q in teammates], dtype=float)
            t_weights = t_mins / t_mins.sum()

            row = {"PLAYER_ID": p["PLAYER_ID"], "PLAYER_NAME": p["PLAYER_NAME"], "TEAM": team,
                   "VECTOR_SEASON": p["VECTOR_SEASON"], "MIN_SEASON": p["MIN_SEASON"], "REASON": p["REASON"]}
            for cat in VECTOR_CATEGORIES:
                row[f"OWN_{cat}"] = p["VECTOR"][cat]
            for cat in VECTOR_CATEGORIES:
                vals = np.array([q["VECTOR"][cat] for q in teammates], dtype=float)
                w_mean = np.sum(t_weights * vals)
                w_std = np.sqrt(np.sum(t_weights * (vals - w_mean) ** 2))
                row[f"TEAMMATE_{cat}_WMEAN"] = w_mean
                row[f"TEAMMATE_{cat}_WSTD"] = w_std
            row["TEAMMATE_COUNT"] = len(teammates)
            row["TEAMMATE_TOTAL_MIN"] = float(t_mins.sum())
            scorable.append(row)

    return pd.DataFrame(scorable), pd.DataFrame(unscorable)


def main():
    print("Building training set (all 8 STANDARD_SEASONS, no holdout -- see module docstring)...")
    # target_min_minutes=0: unlike predict_minutes_from_roster_context.py's own LOSO
    # validation (rotation-only in, rotation-only out), this production model has to
    # score entire 587-player rosters, bench players included. Training only on
    # rotation-caliber targets left it with no low-minutes examples to anchor on, so it
    # extrapolated a rotation-level floor (~13 MPG) onto every bench/two-way player
    # regardless of how weak their skill vector was. context_min_minutes stays at the
    # default (rotation-only) so teammate context is still built from real rotation
    # players either way.
    train_df = build_dataset(target_min_minutes=0)
    feat_cols = feature_columns(train_df)
    print(f"  {len(train_df)} training rows (target_min_minutes=0, context_min_minutes={MIN_TOTAL_MINUTES})")

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train_df[feat_cols], train_df["TARGET_MPG"])

    print("Scoring 2026-27 rosters...")
    latest_vectors = build_latest_vectors()
    latest_minutes = build_latest_minutes()
    replacement_vector = build_replacement_vector()
    scorable, unscorable = build_2026_27_rows(latest_vectors, latest_minutes, replacement_vector)

    print("Quantile-mapping raw predictions onto the real TARGET_MPG distribution's shape...")
    raw_pred = model.predict(scorable[feat_cols])
    scorable["PRED_MPG_UNSCALED"] = quantile_map_to_target(raw_pred, train_df["TARGET_MPG"].to_numpy()).round(1)

    # Rescale each team's calibrated predictions so they sum to 240
    # min/game. The calibration step above fixes the *shape* of the
    # distribution (realistic starter/rotation/bench tiers, matching real
    # per-player MPG outcomes) but says nothing about a team's *total* --
    # summing every scored roster player's individual-season MPG naturally
    # exceeds 240 for a full ~17-man roster (mean ~329 here), same as it
    # would for any real team's full roster. rescale_to_team_total()'s
    # power-law reallocation shifts more of the 240 budget toward each
    # team's best players and less toward its deep bench, rather than
    # discounting stars and scrubs at the same flat rate. PRED_MPG_UNSCALED
    # is kept alongside PRED_MPG so the pre-rescale numbers are still visible.
    scorable["PRED_MPG"] = rescale_to_team_total(scorable, "PRED_MPG_UNSCALED").round(1)
    scorable = scorable.sort_values(["TEAM", "PRED_MPG"], ascending=[True, False])

    out_cols = ["TEAM", "PLAYER_NAME", "PLAYER_ID", "PRED_MPG", "PRED_MPG_UNSCALED",
                "VECTOR_SEASON", "MIN_SEASON", "REASON"]
    if not unscorable.empty:
        unscorable["PRED_MPG"] = np.nan
        unscorable["PRED_MPG_UNSCALED"] = np.nan
        unscorable["VECTOR_SEASON"] = ""
        unscorable["MIN_SEASON"] = ""
    out = pd.concat([scorable[out_cols], unscorable.reindex(columns=out_cols)], ignore_index=True)
    out.to_csv(OUTPUT_PATH, index=False)

    n_rookies = (scorable["REASON"] != "").sum()
    print(f"\nScored {len(scorable)} players ({n_rookies} rookies via replacement-level vector), "
          f"{len(unscorable)} genuinely unscorable (no rotation-caliber historical teammates).")
    team_check = scorable.groupby("TEAM")["PRED_MPG"].sum()
    print(f"Team-total check: min {team_check.min():.1f}, max {team_check.max():.1f} (should both be {TEAM_TOTAL_MPG:.0f})")
    print(f"Wrote {OUTPUT_PATH}")

    stale = scorable[scorable["VECTOR_SEASON"] != "2025-26"]
    print(f"\n{len(stale)} scored players' vector is older than 2025-26 (missed last season / traded-off-radar):")
    if not stale.empty:
        text = stale[["TEAM", "PLAYER_NAME", "VECTOR_SEASON"]].to_string(index=False)
        print(text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
