"""
Builds the team NEED-vector component of the model described in team.md: the
same 12 categories as build_player_vectors.py (9 playtype scores + rim
defense, perimeter defense, rebounding), computed directly from team-level
stats in data/team_stats/team_stats_<season>.csv -- NOT aggregated from
player vectors.

Each category is a strength score internally per team-season (higher =
better, same shape as the player vector, and still z-scored against just
that season's 30 teams -- see below on why no CLT correction is needed for
that step). The NEED conversion is different: for a given season, it pools
that season plus its ROLLING_WINDOW-1 trailing seasons (a 3-season rolling
window by default, see rolling_window()) and standardizes + sigmoid-squashes
each category over that rolling pool (see need_score()), so 0 = approaching
best-ever *for the league as it's playing right now* (not just this
season's 30-team cohort, but not a static decade-old blend either) and 1 =
approaching worst-ever, judged against that same rolling window. This is
deliberately a rolling window rather than the full STANDARD_SEASONS history:
the league goes through real playstyle shifts (e.g. 3pt-volume growth), so
what counts as "elite" or "nonexistent" in a category should track a few
recent seasons of context, not get averaged against a playstyle era that no
longer applies.

Because these are already team-level numbers (one row per team, not an
average of ~15 noisy player samples), z-scoring them directly across the 30
teams does not suffer the central-limit-theorem compression that motivated
build_team_vectors.py's earlier player-aggregation design -- no extra
re-anchoring step is needed here.

No shrinkage and no multi-season decay are applied to the underlying raw
stats (both used at the player level for different reasons -- see
defense.md / build_playtype_vectors.py): a team's season DEF_RATING is
already a full-sample outcome, not a noisy few-possession on/off split, so
shrinkage doesn't apply; and unlike an individual player's career, a team's
roster/scheme can turn over completely year to year, so blending the raw
per-team-season stats across seasons would mix incomparable rosters rather
than smooth out noise. Pooling for the NEED normalization is a different
thing from that -- it doesn't blend a team's stats across seasons, it just
widens the reference frame those (still single-season) stats are judged
against, and only over a short recent window rather than the whole history.

Usage:
    python scripts/build_team_vectors.py --season 2025-26   # write just that season
    python scripts/build_team_vectors.py --season all        # write every standard season
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as static_teams

DATA_DIR = Path("data")
TEAM_STATS_DIR = DATA_DIR / "team_stats"
TEAM_VECTOR_DIR = DATA_DIR / "team_vectors_json"

TEAM_ABBREV_BY_ID = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}

# Matches the player-vector pipeline's standard 8-season window
# (2018-19 through 2025-26 -- see vector_process.md's Changelog for why
# 2016-17/2017-18 were dropped from that standard set).
STANDARD_SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

PLAY_TYPES = [
    "ISOLATION",
    "PNR_HANDLER",
    "PNR_ROLLMAN",
    "POST_UP",
    "SPOT_UP",
    "HANDOFF",
    "CUT",
    "OFF_SCREEN",
    "PUT_BACKS",
]

NEED_CATEGORY_COLUMNS = [f"{pt}_SCORE" for pt in PLAY_TYPES] + [
    "RIM_BLOCK", "PERIMETER_D_BLOCK", "REB_BLOCK",
]

# Need-score normalization pools each season against itself plus this many
# trailing seasons (see rolling_window()), rather than the full
# STANDARD_SEASONS history -- so the reference frame tracks how the league
# is actually playing now instead of averaging across playstyle eras (e.g.
# the 3pt-volume shift). Matches the 3-season trailing window
# build_playtype_vectors.py already uses for player decay.
ROLLING_WINDOW = 3

# Exponent on ppp_badness in the playtype-score formula (see build_vectors()):
# how strongly good effectiveness forgives a high-frequency team. 1 = linear
# ("half as bad at it" forgives "half as much" of the frequency penalty); >1
# is more generous (good-but-not-elite effectiveness forgives more than
# proportionally); a team that is literally #1 in effectiveness always gets
# a full pass regardless of this value, since ppp_badness=0 there. Set to 1
# (linear) at the user's request after comparing k=1 through k=4 in the
# scatter/heatmap artifacts.
EFFECTIVENESS_GATE = 1


def zero_anchored_z(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return series / std if std else series * 0.0


def need_score(strength: pd.Series) -> pd.Series:
    """Standardize a strength score (z-score across the pool) and squash it
    through a logistic sigmoid, inverted so higher strength -> lower need:

        need = 1 / (1 + exp(z)),  z = (strength - mean) / std

    An average team-season (z=0) scores 0.5. Need approaches 0 for a
    dominant team and 1 for a nonexistent one, but -- unlike min-max --
    never actually touches either endpoint, so the scale isn't pinned to
    whichever single team-season happens to be the pool's literal best/worst
    (an outlier that would otherwise stretch/compress everyone else's
    score). A category with no spread across the pool (std == 0) carries no
    signal either way -- scored as a neutral 0.5 rather than dividing by
    zero."""
    std = strength.std(ddof=0)
    if not std:
        return pd.Series(0.5, index=strength.index)
    z = (strength - strength.mean()) / std
    return 1 / (1 + np.exp(z))


def build_vectors(season: str) -> pd.DataFrame:
    path = TEAM_STATS_DIR / f"team_stats_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run pull_nba_data.py --season {season} first")
    df = pd.read_csv(path, encoding="utf-8")
    df["TEAM"] = df["TEAM_ID"].map(TEAM_ABBREV_BY_ID)
    gp = df["GP"].clip(lower=1)

    # ---- Playtype scores ----
    # A team leaning heavily on a playtype it's *bad* at is the actual
    # weakness (high need); rarely using a playtype it happens to be
    # effective at isn't a real hole (low need). Unlike the player vector --
    # where frequency and effectiveness both point the same direction, since
    # a notable *skill* is one a player leans on and is good at -- team
    # *need* is about exposure: heavy use of a weak playtype is the worst
    # quadrant, not the best.
    #
    # This is a gated interaction, not a weighted geometric mean of two
    # z-scores (an earlier version of this formula): frequency only becomes
    # a problem in proportion to how bad the effectiveness is, so elite
    # effectiveness can fully neutralize even an extreme frequency outlier,
    # not just dampen it. A weighted geometric mean (`freq_inv^a * ppp^b`)
    # can't do this -- any exponent on a near-zero frequency term still
    # drags the product down noticeably, so a team that happens to be the
    # single most-frequent user of a playtype in the league gets pinned near
    # max need almost regardless of how effective it is (confirmed
    # empirically: even at a 0.25/0.75 split favoring effectiveness, a team
    # hypothetically both #1 in frequency AND #1 in effectiveness still
    # ranked dead last under that formula). Percentile ranks (bounded in
    # (0, 1], unlike raw z-scores which can run arbitrarily large) make
    # `EFFECTIVENESS_GATE` actually work as a dial: raising it lets good
    # effectiveness forgive a high-frequency team more readily.
    for pt in PLAY_TYPES:
        freq_pct_rank = df[f"{pt}_FREQ_PCT"].rank(pct=True)  # 1.0 = most frequent in the league
        ppp_pct_rank = df[f"{pt}_PPP"].rank(pct=True)        # 1.0 = most effective in the league
        ppp_badness = 1 - ppp_pct_rank
        need_raw = freq_pct_rank * ppp_badness**EFFECTIVENESS_GATE
        df[f"{pt}_SCORE"] = 1 - need_raw

    # ---- Rate-adjust counting stats (season totals -> per game) ----
    # STL_PG and BLK_PG stand in for the player-level PCT_STEALS/PCT_BLOCKS
    # rate stats -- team_stats has no opponent-shot-normalized steal/block
    # rate on file, so per-game (already the convention used for the other
    # counting stats here) is the closest available substitute.
    df["CONTESTED_3PT_PG"] = df["CONTESTED_SHOTS_3PT"] / gp
    df["DEFLECTIONS_PG"] = df["DEFLECTIONS"] / gp
    df["STL_PG"] = df["STL"] / gp
    df["BLK_PG"] = df["BLK"] / gp
    df["RIM_DFGA_PG"] = df["RIM_DFGA"] / gp
    df["BOXOUTS_PG"] = df["DEF_BOXOUTS"] / gp

    # ---- Perimeter defense block ----
    contested_3pt_z = zero_anchored_z(df["CONTESTED_3PT_PG"])
    deflections_z = zero_anchored_z(df["DEFLECTIONS_PG"])
    stl_z = zero_anchored_z(df["STL_PG"])
    activity_score = contested_3pt_z**0.35 * deflections_z**0.35 * stl_z**0.30

    drtg = df["DEF_RATING"]
    # No empirical-Bayes shrinkage here -- unlike a player's on/off split,
    # a team's season DEF_RATING is already the full-sample outcome, not a
    # noisy low-minute proxy that needs stabilizing.
    adjusted_drtg = drtg - (0.20 * activity_score * drtg.std(ddof=0))
    league_avg_drtg = adjusted_drtg.mean()
    adj_std = adjusted_drtg.std(ddof=0)
    adjusted_drtg_z = (league_avg_drtg - adjusted_drtg) / adj_std if adj_std else adjusted_drtg * 0.0
    perimeter_d_block = adjusted_drtg_z - adjusted_drtg_z.min()

    df["PERIMETER_D_BLOCK"] = perimeter_d_block

    # ---- Rim protection block ----
    blk_z = zero_anchored_z(df["BLK_PG"])
    dfg_z = zero_anchored_z(df["RIM_DFG_PCT"])
    diff_z = zero_anchored_z(df["RIM_DIFF_PCT"])
    dfga_z = zero_anchored_z(df["RIM_DFGA_PG"])

    dfg_z_inv = dfg_z.max() - dfg_z
    diff_z_inv = diff_z.max() - diff_z

    df["RIM_BLOCK"] = blk_z**0.25 * dfg_z_inv**0.30 * diff_z_inv**0.35 * dfga_z**0.10

    # ---- Rebounding block ----
    dreb_z = zero_anchored_z(df["DREB_PCT"])
    oreb_z = zero_anchored_z(df["OREB_PCT"])
    boxouts_z = zero_anchored_z(df["BOXOUTS_PG"])
    df["REB_BLOCK"] = (dreb_z * oreb_z * boxouts_z) ** (1 / 3)

    # defense.md's player-level pipeline rescales PERIMETER_D_BLOCK to match
    # RIM_BLOCK/REB_BLOCK's *mean* (their std's were already comparable at
    # 500-player sample size, so a mean-only fix was sufficient there). At
    # 30-team sample size that doesn't hold: PERIMETER_D_BLOCK's raw std is
    # already in line with the other two blocks, but its raw mean is tiny by
    # comparison (zero-anchored z-score means are far less stable with only
    # 30 samples -- REB_BLOCK's mean alone can run ~5x RIM_BLOCK's). Forcing
    # the mean to match via a uniform multiplier would drag the std along
    # with it, blowing PERIMETER_DEFENSE's spread out to several times every
    # other category's -- confirmed empirically (std ~4.7 vs ~1.0-1.2
    # elsewhere) before this fix. Matching *std* instead keeps every
    # category's spread comparable, which is what matters for any downstream
    # z-score-based distance/regression use, and still preserves the
    # zero-anchor (0 stays 0 under a positive multiply).
    reference_std = (df["RIM_BLOCK"].std(ddof=0) + df["REB_BLOCK"].std(ddof=0)) / 2
    perimeter_std = df["PERIMETER_D_BLOCK"].std(ddof=0)
    if perimeter_std > 0:
        df["PERIMETER_D_BLOCK"] = df["PERIMETER_D_BLOCK"] * (reference_std / perimeter_std)

    df["SEASON"] = season
    return df


def rolling_window(season: str, seasons: list[str] = STANDARD_SEASONS, window: int = ROLLING_WINDOW) -> list[str]:
    """`season` plus up to `window - 1` seasons immediately before it in
    `seasons` (chronological order). Early seasons with fewer than
    `window - 1` predecessors on file just get a shorter window -- no
    reaching outside `seasons` to pad it out, same "no imputation" stance
    the player pipeline takes for its own decay window."""
    idx = seasons.index(season)
    start = max(0, idx - window + 1)
    return seasons[start:idx + 1]


def build(output_seasons: list[str], all_seasons: list[str] = STANDARD_SEASONS) -> dict:
    """Team need vectors for every season in `output_seasons`, keyed
    `{season: {team: {...}}}`. Each season's NEED_CATEGORY_COLUMNS are
    z-scored and sigmoid-squashed (see need_score()) against a pool of just
    that season plus its `ROLLING_WINDOW - 1` predecessors (see
    rolling_window()) -- not the full `all_seasons` history -- so the
    normalization reference tracks how the league actually plays *right
    now*, not a static blend of a decade-old league alongside the current
    one."""
    raw_by_season = {season: build_vectors(season) for season in all_seasons}

    by_season: dict = {}
    for season in output_seasons:
        pool_seasons = rolling_window(season, all_seasons)
        pool = pd.concat([raw_by_season[s] for s in pool_seasons], ignore_index=True)
        for col in NEED_CATEGORY_COLUMNS:
            pool[col] = need_score(pool[col])

        season_rows = pool[pool["SEASON"] == season]
        teams = {}
        for _, row in season_rows.iterrows():
            need_vector = {pt: row[f"{pt}_SCORE"] for pt in PLAY_TYPES}
            need_vector["RIM_DEFENSE"] = row["RIM_BLOCK"]
            need_vector["PERIMETER_DEFENSE"] = row["PERIMETER_D_BLOCK"]
            need_vector["REBOUNDING"] = row["REB_BLOCK"]
            teams[row["TEAM"]] = {
                "TEAM": row["TEAM"],
                "SEASON": row["SEASON"],
                "NEED_VECTOR": need_vector,
            }
        by_season[season] = teams
    return by_season


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season", required=True,
        help="e.g. 2025-26, or 'all' to write every season in STANDARD_SEASONS",
    )
    parser.add_argument("--out-dir", default=str(TEAM_VECTOR_DIR))
    args = parser.parse_args()

    output_seasons = STANDARD_SEASONS if args.season == "all" else [args.season]
    by_season = build(output_seasons)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for season in output_seasons:
        out_path = out_dir / f"team_vectors_{season}.json"
        out_path.write_text(json.dumps(by_season[season], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(by_season[season])} team vectors to {out_path}")


if __name__ == "__main__":
    main()
