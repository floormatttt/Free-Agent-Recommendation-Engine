"""
Builds the defense/rebounding component of the player-vector model,
implementing the exact pipeline in defense.md: a perimeter-defense block, a
rim-protection block, and a rebounding block.

For each player, per-season counting stats are rate-adjusted (per game)
where needed, then up to 3 seasons are blended using the same
experience-adjusted time decay as build_playtype_vectors.py.

Before that decay blend, on-court DEF_RATING (the input to the perimeter
defense block) is run through empirical-Bayes shrinkage toward the
player's own team's season DEF_RATING, weighted by on-court minutes vs.
K_SHRINKAGE pseudo-minutes of prior belief -- see load_season(). This
tames the wild single-season swings a low-minute player's on/off split can
show; it does not by itself separate individual defense from team scheme
(the prior is the team's own rating), which is a known limitation, not a
bug.

All three blocks use "zero-anchored" z-scores (value/std, no mean
subtraction -- see build_playtype_vectors.py's docstring for why). Because
every raw input here is a naturally non-negative rate/count stat (or, for
DFG%/Diff% at the rim, is made non-negative via a max-shift inversion
before being multiplied in), none of the composite formulas need the
sign-preserving power trick the first draft of this script used --
ordinary exponentiation is safe throughout.

Usage:
    python build_defense_vectors.py --season 2025-26
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as static_teams

DATA_DIR = Path("data")
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
TEAM_STATS_DIR = DATA_DIR / "team_stats"
OUT_DIR = DATA_DIR / "player_vectors_full"

TEAM_ABBREV_BY_ID = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}

# Empirical-Bayes shrinkage for on-court DEF_RATING: a player's raw on/off
# sample is noisy at low minutes (a few possessions can swing it wildly), so
# it's blended with their team's season DEF_RATING as a prior, weighted by
# on-court minutes vs. this many "pseudo-minutes" of prior belief. A player
# with far fewer than K_SHRINKAGE on-court minutes ends up close to their
# team's average; a full-season starter is barely pulled off their own mark.
K_SHRINKAGE = 500.0

# raw column name -> internal name used while blending. The three "_PG"
# entries are computed from a counting stat divided by GP at load time.
# ON_COURT_DEF_RATING is handled separately (see load_season) since it needs
# shrinkage, not a plain rename.
RATE_COLUMNS = {
    "PCT_STEALS": "STL_PCT",
    "PCT_BLOCKS": "BLK_PCT",
    "RIM_DFG_PCT": "DFG_PCT",
    "RIM_DIFF_PCT": "DIFF_PCT",
    "RIM_DFGA": "DFGA",
    "DREB_PCT": "DREB_PCT",
    "OREB_PCT": "OREB_PCT",
}
PER_GAME_COLUMNS = {
    "CONTESTED_SHOTS_3PT": "CONTESTED_3PT_PG",
    "DEFLECTIONS": "DEFLECTIONS_PG",
    "DEF_BOXOUTS": "BOXOUTS_PG",
}
ALL_STAT_COLS = list(RATE_COLUMNS.values()) + list(PER_GAME_COLUMNS.values()) + ["ON_DRTG"]

# Career-stage decay -- same shape as build_playtype_vectors.py: recency
# bias is strongest at the start/end of a career, weakest at the prime.
PRIME_CENTER = 7.0
PRIME_SPREAD = 5.0
DECAY_MIN = 0.15
DECAY_MAX = 1.2


def decay_strength(experience: float) -> float:
    t = min(1.0, ((experience - PRIME_CENTER) / PRIME_SPREAD) ** 2)
    return DECAY_MIN + (DECAY_MAX - DECAY_MIN) * t


def season_offset(season: str, back: int) -> str:
    start = int(season.split("-")[0]) - back
    end_yy = (start + 1) % 100
    return f"{start}-{end_yy:02d}"


def load_team_drtg(season: str):
    """Each team's season DEF_RATING, indexed by team abbreviation -- the
    shrinkage prior for on-court DEF_RATING."""
    path = TEAM_STATS_DIR / f"team_stats_{season}.csv"
    if not path.exists():
        return None
    team_df = pd.read_csv(path, encoding="utf-8")
    team_df["TEAM"] = team_df["TEAM_ID"].map(TEAM_ABBREV_BY_ID)
    return team_df.set_index("TEAM")["DEF_RATING"]


def load_season(season: str):
    path = PLAYER_STATS_DIR / f"player_stats_{season}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns=RATE_COLUMNS)
    gp = df["GP"].clip(lower=1)
    for raw_col, pg_col in PER_GAME_COLUMNS.items():
        df[pg_col] = df[raw_col] / gp

    team_drtg = load_team_drtg(season)
    team_prior = df["TEAM"].map(team_drtg) if team_drtg is not None else pd.Series(np.nan, index=df.index)
    # a team's own abbreviation is always in its own season's team_stats, so
    # this only falls back to the league mean if that lookup is somehow
    # missing (e.g. a historical abbreviation mismatch).
    team_prior = team_prior.fillna(team_prior.mean())

    on_court_min = df["ON_COURT_MIN"].fillna(0.0)
    on_drtg_raw = df["ON_COURT_DEF_RATING"].fillna(team_prior)
    df["ON_DRTG"] = (on_court_min * on_drtg_raw + K_SHRINKAGE * team_prior) / (
        on_court_min + K_SHRINKAGE
    )

    return df


def experience_for(bio_row, season_start_year: int, fallback_years_seen: int) -> float:
    draft_year = bio_row.get("DRAFT_YEAR")
    try:
        return max(0.0, season_start_year - float(draft_year))
    except (TypeError, ValueError):
        return float(fallback_years_seen)


def zero_anchored_z(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return series / std if std else series * 0.0


def build_vectors(season: str) -> pd.DataFrame:
    season_ids = [season, season_offset(season, 1), season_offset(season, 2)]
    current = load_season(season_ids[0])
    if current is None:
        raise FileNotFoundError(f"no player_stats file for {season}")

    prior_frames = [load_season(s) for s in season_ids[1:]]

    # No baseline rule is defined for defense (unlike playtypes' "not
    # enough of a playstyle"), so a missing raw stat is filled with that
    # season's population mean -- a neutral (zero-anchored-z-of-average)
    # stand-in rather than penalizing/rewarding missing data.
    season_means = {}
    for f in [current] + prior_frames:
        if f is None:
            continue
        season_means[id(f)] = {col: f[col].mean(skipna=True) for col in ALL_STAT_COLS}

    prior_by_id = [(f.set_index("PLAYER_ID") if f is not None else None) for f in prior_frames]

    season_start_year = int(season.split("-")[0])
    rows = []

    for _, bio in current.iterrows():
        pid = bio["PLAYER_ID"]

        available = [0]
        for i, pf in enumerate(prior_by_id, start=1):
            if pf is not None and pid in pf.index:
                available.append(i)

        experience = experience_for(bio, season_start_year, len(available))
        d = decay_strength(experience)
        raw_w = {i: np.exp(-d * i) for i in available}
        wsum = sum(raw_w.values())
        weights = {i: w / wsum for i, w in raw_w.items()}

        out = {
            "PLAYER_ID": pid,
            "PLAYER_NAME": bio["PLAYER_NAME"],
            "TEAM": bio["TEAM"],
            "EXPERIENCE_YRS": experience,
        }

        decayed = {c: 0.0 for c in ALL_STAT_COLS}
        for i in available:
            if i == 0:
                row, frame = bio, current
            else:
                frame = prior_frames[i - 1]
                row = prior_by_id[i - 1].loc[pid] if pid in prior_by_id[i - 1].index else None

            for col in ALL_STAT_COLS:
                val = row.get(col) if row is not None else None
                if row is None or pd.isna(val):
                    val = season_means[id(frame)][col]
                decayed[col] += weights[i] * val

        for col in ALL_STAT_COLS:
            out[f"{col}_DECAYED"] = decayed[col]

        rows.append(out)

    df = pd.DataFrame(rows)

    # ---- Perimeter defense block ----
    contested_3pt_z = zero_anchored_z(df["CONTESTED_3PT_PG_DECAYED"])
    deflections_z = zero_anchored_z(df["DEFLECTIONS_PG_DECAYED"])
    stl_pct_z = zero_anchored_z(df["STL_PCT_DECAYED"])
    activity_score = contested_3pt_z**0.35 * deflections_z**0.35 * stl_pct_z**0.30

    on_drtg = df["ON_DRTG_DECAYED"]
    adjusted_drtg = on_drtg - (0.20 * activity_score * on_drtg.std(ddof=0))
    league_avg_drtg = adjusted_drtg.mean()
    adj_std = adjusted_drtg.std(ddof=0)
    adjusted_drtg_z = (league_avg_drtg - adjusted_drtg) / adj_std if adj_std else adjusted_drtg * 0.0
    perimeter_d_block = adjusted_drtg_z - adjusted_drtg_z.min()

    df["CONTESTED_3PT_Z"] = contested_3pt_z
    df["DEFLECTIONS_Z"] = deflections_z
    df["STL_PCT_Z"] = stl_pct_z
    df["ACTIVITY_SCORE"] = activity_score
    df["ADJUSTED_DRTG"] = adjusted_drtg
    df["PERIMETER_D_BLOCK"] = perimeter_d_block

    # ---- Rim protection block ----
    blk_z = zero_anchored_z(df["BLK_PCT_DECAYED"])
    dfg_z = zero_anchored_z(df["DFG_PCT_DECAYED"])
    diff_z = zero_anchored_z(df["DIFF_PCT_DECAYED"])
    dfga_z = zero_anchored_z(df["DFGA_DECAYED"])

    dfg_z_inv = dfg_z.max() - dfg_z
    diff_z_inv = diff_z.max() - diff_z

    df["BLK_PCT_Z"] = blk_z
    df["DFG_PCT_Z_INV"] = dfg_z_inv
    df["DIFF_PCT_Z_INV"] = diff_z_inv
    df["DFGA_Z"] = dfga_z

    df["RIM_BLOCK"] = blk_z**0.25 * dfg_z_inv**0.30 * diff_z_inv**0.35 * dfga_z**0.10

    # ---- Rebounding block ----
    dreb_z = zero_anchored_z(df["DREB_PCT_DECAYED"])
    oreb_z = zero_anchored_z(df["OREB_PCT_DECAYED"])
    boxouts_z = zero_anchored_z(df["BOXOUTS_PG_DECAYED"])

    df["DREB_PCT_Z"] = dreb_z
    df["OREB_PCT_Z"] = oreb_z
    df["BOXOUTS_PG_Z"] = boxouts_z
    df["REB_BLOCK"] = (dreb_z * oreb_z * boxouts_z) ** (1 / 3)

    # PERIMETER_D_BLOCK's min-shift step (worst player -> 0) systematically
    # inflates its mean relative to RIM_BLOCK/REB_BLOCK, which have no such
    # shift -- confirmed empirically: perimeter defense ran ~0.5-1.4 points
    # above the other categories' average across every season checked.
    # Rescale it to match the other two blocks' typical magnitude, which
    # preserves the "worst player = 0, non-negative" design since scaling a
    # value already anchored at 0 by a positive constant leaves it at 0.
    reference_mean = (df["RIM_BLOCK"].mean() + df["REB_BLOCK"].mean()) / 2
    perimeter_mean = df["PERIMETER_D_BLOCK"].mean()
    if perimeter_mean > 0:
        df["PERIMETER_D_BLOCK"] = df["PERIMETER_D_BLOCK"] * (reference_mean / perimeter_mean)

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="e.g. 2025-26")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    df = build_vectors(args.season)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"player_defense_vectors_{args.season}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} player vectors to {out_path}")


if __name__ == "__main__":
    main()
