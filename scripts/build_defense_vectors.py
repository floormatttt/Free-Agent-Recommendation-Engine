"""
Builds the defense/rebounding component of the player-vector model,
implementing the exact pipeline in defense.md: a perimeter-defense block, a
rim-protection block, and a rebounding block.

For each player, per-season counting stats are rate-adjusted (per game)
where needed, then up to 3 seasons are blended using the same
experience-adjusted time decay as build_playtype_vectors.py.

The perimeter-defense block's outcome stat is Defensive Win Shares (Dean
Oliver's formula, self-computed by scripts/compute_dws.py rather than taken
from NBA.com's own DEF_WS field -- see that script's docstring for why),
read as a per-game rate (DWS_PER_GAME, already computed and saved on
data/player_stats/*.csv by compute_dws.py) at load time -- see
load_season(). This replaces an earlier on/off DEF_RATING approach that
leaked team scheme into individual ratings (a bench player on an elite
defense shrank toward a flattering number regardless of their own play).
DWS isn't fully team-independent either (it's still partly box-score/team
derived), but it doesn't have that specific team-prior leak -- see Known
Limitations in defense.md.

All three blocks use "zero-anchored" z-scores (value/std, no mean
subtraction -- see build_playtype_vectors.py's docstring for why). Because
every raw input here is a naturally non-negative rate/count stat (or, for
DFG%/Diff% at the rim, is made non-negative via a max-shift inversion
before being multiplied in), none of the composite formulas need the
sign-preserving power trick the first draft of this script used --
ordinary exponentiation is safe throughout.

Usage:
    python scripts/build_defense_vectors.py --season 2025-26
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
OUT_DIR = DATA_DIR / "player_vectors_full"
UNDRAFTED_FIRST_YEAR_PATH = DATA_DIR / "undrafted_first_year.csv"

# raw column name -> internal name used while blending. The three "_PG"
# entries are computed from a counting stat divided by GP at load time.
# DWS_PER_GAME is already a rate (computed by compute_dws.py), so it's
# included directly in ALL_STAT_COLS without needing an entry here.
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
ALL_STAT_COLS = list(RATE_COLUMNS.values()) + list(PER_GAME_COLUMNS.values()) + ["DWS_PER_GAME"]

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


def load_season(season: str):
    path = PLAYER_STATS_DIR / f"player_stats_{season}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns=RATE_COLUMNS)
    gp = df["GP"].clip(lower=1)
    for raw_col, pg_col in PER_GAME_COLUMNS.items():
        df[pg_col] = df[raw_col] / gp

    # DWS_PER_GAME is computed and saved by compute_dws.py (Dean Oliver's
    # formula), not derived here. If a season's player_stats file predates
    # that script (or the underlying DEF_RATING/team inputs were
    # unavailable for that season), the column is simply absent; leaving it
    # NaN lets the decay blend below renormalize across whichever of a
    # player's available seasons do have it.
    if "DWS_PER_GAME" not in df.columns:
        df["DWS_PER_GAME"] = np.nan

    return df


def load_undrafted_first_years() -> dict:
    """PLAYER_ID -> real NBA debut season (FROM_YEAR) for undrafted players,
    backfilled via nba_api's CommonPlayerInfo (scripts/backfill_undrafted_first_year.py)
    since these players have no DRAFT_YEAR to derive experience from."""
    if not UNDRAFTED_FIRST_YEAR_PATH.exists():
        return {}
    df = pd.read_csv(UNDRAFTED_FIRST_YEAR_PATH, encoding="utf-8")
    df = df.dropna(subset=["FROM_YEAR"])
    return {int(pid): int(year) for pid, year in zip(df["PLAYER_ID"], df["FROM_YEAR"])}


def experience_for(bio_row, season_start_year: int, fallback_years_seen: int, first_year_lookup: dict) -> float:
    """Years since being drafted, as a proxy for career stage. Undrafted
    players (no DRAFT_YEAR) use their real NBA debut season (FROM_YEAR) from
    first_year_lookup instead; only players unresolved even there fall back
    to the number of seasons we actually have on file."""
    draft_year = bio_row.get("DRAFT_YEAR")
    try:
        return max(0.0, season_start_year - float(draft_year))
    except (TypeError, ValueError):
        pass
    first_year = first_year_lookup.get(int(bio_row["PLAYER_ID"]))
    if first_year is not None:
        return max(0.0, season_start_year - first_year)
    return float(fallback_years_seen)


def zero_anchored_z(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return series / std if std else series * 0.0


def build_vectors(season: str) -> pd.DataFrame:
    season_ids = [season, season_offset(season, 1), season_offset(season, 2)]
    current = load_season(season_ids[0])
    if current is None:
        raise FileNotFoundError(f"no player_stats file for {season}")

    first_year_lookup = load_undrafted_first_years()

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

        experience = experience_for(bio, season_start_year, len(available), first_year_lookup)
        d = decay_strength(experience)
        raw_w = {i: np.exp(-d * i) for i in available}

        out = {
            "PLAYER_ID": pid,
            "PLAYER_NAME": bio["PLAYER_NAME"],
            "TEAM": bio["TEAM"],
            "EXPERIENCE_YRS": experience,
        }

        # season -> row/frame, looked up once per player rather than per stat.
        season_lookup = {}
        for i in available:
            if i == 0:
                season_lookup[i] = (bio, current)
            else:
                frame = prior_frames[i - 1]
                row = prior_by_id[i - 1].loc[pid] if pid in prior_by_id[i - 1].index else None
                season_lookup[i] = (row, frame)

        # Per-stat weighted blend, renormalized over only the seasons where
        # that stat is actually available -- most stats are available in
        # every season, but a handful of historical seasons have no DEF_WS
        # at all (see load_season), and those seasons should drop out of
        # DEF_WS_PER48's blend rather than pull it toward 0/NaN.
        for col in ALL_STAT_COLS:
            vals = {}
            for i in available:
                row, frame = season_lookup[i]
                val = row.get(col) if row is not None else None
                if row is None or pd.isna(val):
                    val = season_means[id(frame)][col]
                if pd.notna(val):
                    vals[i] = val

            if not vals:
                out[f"{col}_DECAYED"] = np.nan
                continue

            w = {i: raw_w[i] for i in vals}
            wsum = sum(w.values())
            out[f"{col}_DECAYED"] = sum((w[i] / wsum) * vals[i] for i in vals)

        rows.append(out)

    df = pd.DataFrame(rows)

    # ---- Perimeter defense block ----
    contested_3pt_z = zero_anchored_z(df["CONTESTED_3PT_PG_DECAYED"])
    deflections_z = zero_anchored_z(df["DEFLECTIONS_PG_DECAYED"])
    stl_pct_z = zero_anchored_z(df["STL_PCT_DECAYED"])
    activity_score = contested_3pt_z**0.35 * deflections_z**0.35 * stl_pct_z**0.30

    dws_per_game = df["DWS_PER_GAME_DECAYED"]
    adjusted_dws = dws_per_game + (0.20 * activity_score * dws_per_game.std(ddof=0))
    league_avg_dws = adjusted_dws.mean()
    adj_std = adjusted_dws.std(ddof=0)
    adjusted_dws_z = (adjusted_dws - league_avg_dws) / adj_std if adj_std else adjusted_dws * 0.0
    perimeter_d_block = adjusted_dws_z - adjusted_dws_z.min()

    df["CONTESTED_3PT_Z"] = contested_3pt_z
    df["DEFLECTIONS_Z"] = deflections_z
    df["STL_PCT_Z"] = stl_pct_z
    df["ACTIVITY_SCORE"] = activity_score
    df["ADJUSTED_DWS_PER_GAME"] = adjusted_dws
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
