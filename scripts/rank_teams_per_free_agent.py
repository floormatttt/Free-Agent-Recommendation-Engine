"""
Flips exhaustive_free_agent_combos_need.py's team-centric question ("which
free agents fit team X best") into a player-centric one: for each of the
47 usable free agents, rank all 30 teams 1-30 by the same composite %
need-reduction score (need_before -> need_after via the linearized PLS
NEED_VECTOR model), evaluated as if that ONE player were that team's only
offseason addition on top of its real PRE_FA roster.

This is a single-player (k=1) evaluation for every (player, team) pair --
under a second once the model and contribution matrices are built. Only
teams with at least one real July-2026 signing are scored (28 of 30 --
GSW and WAS signed nobody this cycle; there's no hard-cap/apron flag in
this repo's data to confirm *why*, so rather than assert a cap situation
we don't have evidence for, a team that added zero free agents in reality
is treated as not evaluable as a hypothetical destination either).

Composite score is weighted by category (see exhaustive_free_agent_combos_
need.py's load_win_contribution_weights): |Ridge coefficient| from the
real NEED_VECTOR -> win% model, normalized to sum to 1, so a category
that actually predicts winning more (PERIMETER_DEFENSE, by a wide margin)
counts for more than one that barely matters (PUT_BACKS). Same
DENOM_FLOOR-capped percentage formula as the combinatorial script, to
keep a near-1 baseline need in one category from dominating the score.

Also emits TEAM_Z_SCORE: WEIGHTED_PCT_NEED_INCREASE standardized within
each team's own 47-player distribution ((score - team_mean) / team_std).
Raw % answers "how many wins-equivalent of need does this fill" -- which
is legitimately larger for a bad team with more room to improve, real
signal, not a bug. But it means ranking a team's OWN free-agent options
by raw % just reproduces "is this team bad" (measured: corr(team's real,
already-played-season-independent NEED_BEFORE_WEIGHTED_COMPOSITE, avg raw
% across all 47 players) = 0.80 across the 28 evaluable teams) rather than
telling you which player is actually the standout option for that
specific team. TEAM_Z_SCORE decouples that (same correlation drops to
0.10, not significant) by asking "how many standard deviations above this
team's own typical option is this player" instead of raw swing size --
it's the fairer lens for "which role players should team X specifically
value," and is what the artifact's "By team" view sorts by. It does NOT
change a single team's own ranking order versus raw % (z-score is a
positive linear rescaling of each team's own column) -- it only changes
which comparisons are meaningful ACROSS teams.

Also emits PLAYER_Z_SCORE: the mirror-image standardization, within each
PLAYER's own 28-team distribution instead of each team's 47-player one.
Unlike TEAM_Z_SCORE, this isn't correcting a confound -- the raw %
"which team gains the most from signing me" question is legitimately
dominated by weak teams having more room to improve, and that's real, not
bias. PLAYER_Z_SCORE instead answers "how much of a standout is this
particular team for me, relative to my own typical team" -- useful for
comparing that standout-ness across different players on equal footing
(one player's raw scores might swing 3x tighter across teams than
another's), and for display consistency with the by-team view. Like
TEAM_Z_SCORE, it's a positive linear rescaling of each player's own row,
so it does NOT change which team ranks best for a given player -- RANK is
identical whether computed from the raw % or from PLAYER_Z_SCORE.

Output: data/free_agency/free_agent_team_fit_rankings.csv, long format:
    PLAYER_NAME, TEAM, RANK (1 = best fit by raw %), WEIGHTED_PCT_NEED_INCREASE,
    TEAM_Z_SCORE, PLAYER_Z_SCORE

Usage:
    python scripts/rank_teams_per_free_agent.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from exhaustive_free_agent_combos_need import (
    fit_need_pls, fa_pool_contribution_matrices, team_pre_fa_moments,
    team_feat_to_need, load_win_contribution_weights, FA_SIGNED_PATTERN, DENOM_FLOOR,
)
from predict_2026_27_team_profile import TEAM_TOTAL_MIN_TARGET
from predict_2026_27_minutes import build_latest_vectors, build_latest_minutes, ROSTER_PATH

ROOT = Path(__file__).resolve().parent.parent


def main():
    rosters = pd.read_csv(ROSTER_PATH, encoding="utf-8")
    is_fa = rosters["HOW_ACQUIRED"].apply(lambda s: bool(FA_SIGNED_PATTERN.match(str(s))))
    fa_pool_df = rosters[is_fa].copy()
    pre_fa_rosters = rosters[~is_fa].copy()

    print("Fitting roster-vectors -> NEED_VECTOR (PLS, linearized)...")
    W, intercept, feat_cols = fit_need_pls()
    weights = load_win_contribution_weights()

    latest_vectors = build_latest_vectors()
    latest_minutes = build_latest_minutes()

    print("Building free-agent pool contribution matrices...")
    C1, C2, Cmin, names, fa_real_teams = fa_pool_contribution_matrices(fa_pool_df, latest_vectors, latest_minutes)
    print(f"  {len(names)} free agents with usable data")

    teams = sorted(fa_pool_df["TEAM"].unique())
    excluded = sorted(set(rosters["TEAM"].unique()) - set(teams))
    print(f"  Evaluating {len(teams)} teams with a real July-2026 signing "
          f"(excluded, zero signings: {', '.join(excluded)})")
    team_need_before = {}
    for team in teams:
        pre_S1, pre_S2, pre_min = team_pre_fa_moments(pre_fa_rosters, team, latest_vectors, latest_minutes)
        pre_mean = pre_S1 / pre_min
        pre_var = pre_S2 / pre_min - pre_mean ** 2
        pre_std = np.sqrt(np.clip(pre_var, 0, None))
        pre_feat = np.empty(25)
        pre_feat[0] = TEAM_TOTAL_MIN_TARGET
        pre_feat[1::2] = pre_mean
        pre_feat[2::2] = pre_std
        need_before = team_feat_to_need(pre_feat[None, :], W, intercept)[0]
        team_need_before[team] = (pre_S1, pre_S2, pre_min, need_before)

    rows = []
    for p_idx, player in enumerate(names):
        p_min, p_S1, p_S2 = Cmin[p_idx], C1[p_idx], C2[p_idx]
        for team in teams:
            pre_S1, pre_S2, pre_min, need_before = team_need_before[team]
            total_min = pre_min + p_min
            mean = (pre_S1 + p_S1) / total_min
            var = (pre_S2 + p_S2) / total_min - mean ** 2
            std = np.sqrt(np.clip(var, 0, None))
            feat = np.empty(25)
            feat[0] = TEAM_TOTAL_MIN_TARGET
            feat[1::2] = mean
            feat[2::2] = std
            need_after = team_feat_to_need(feat[None, :], W, intercept)[0]
            denom = np.maximum(1 - need_before, DENOM_FLOOR)
            pct = float(((need_before - need_after) / denom * 100) @ weights)
            rows.append({"PLAYER_NAME": player, "TEAM": team, "WEIGHTED_PCT_NEED_INCREASE": round(pct, 3)})

    df = pd.DataFrame(rows)
    df["RANK"] = df.groupby("PLAYER_NAME")["WEIGHTED_PCT_NEED_INCREASE"].rank(ascending=False, method="first").astype(int)
    df["TEAM_Z_SCORE"] = df.groupby("TEAM")["WEIGHTED_PCT_NEED_INCREASE"].transform(
        lambda s: (s - s.mean()) / s.std()).round(3)
    df["PLAYER_Z_SCORE"] = df.groupby("PLAYER_NAME")["WEIGHTED_PCT_NEED_INCREASE"].transform(
        lambda s: (s - s.mean()) / s.std()).round(3)
    df = df.sort_values(["PLAYER_NAME", "RANK"])[
        ["PLAYER_NAME", "TEAM", "RANK", "WEIGHTED_PCT_NEED_INCREASE", "TEAM_Z_SCORE", "PLAYER_Z_SCORE"]]
    df.to_csv(ROOT / "data" / "free_agency" / "free_agent_team_fit_rankings.csv", index=False, encoding="utf-8")
    print(f"\nWrote data/free_agency/free_agent_team_fit_rankings.csv ({len(df)} rows, "
          f"{df['PLAYER_NAME'].nunique()} players x {len(teams)} teams)")


if __name__ == "__main__":
    main()
