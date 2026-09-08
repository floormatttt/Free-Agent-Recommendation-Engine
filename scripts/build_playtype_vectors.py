"""
Builds the playtype component of the player-vector model described in
player.md.

For each player, blends up to 3 seasons of playtype Freq%/PPP data using an
experience-adjusted time decay (recency bias is strongest at the start and
end of a career, and "bottoms out" -- i.e. weights spread more evenly across
the 3 seasons -- at the prime), then converts the blended Freq%/PPP into a
scale-free "playtype score" per play type via z-score-style normalization
and a geometric mean.

Only the playtype portion of player.md's model is implemented here --
defense and rebounding are out of scope for this pass, so the output vector
covers 9 of the 12 categories.

Usage:
    python scripts/build_playtype_vectors.py --season 2025-26
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
OUT_DIR = DATA_DIR / "player_vectors_full"
UNDRAFTED_FIRST_YEAR_PATH = DATA_DIR / "undrafted_first_year.csv"

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

# "Not enough of a playstyle" -> assume this baseline instead of a noisy
# small-sample (or missing) value.
BASELINE_FREQ = 0.005
BASELINE_PPP = 0.9

# Career-stage decay: how strongly the current season is favored over the
# two prior seasons. U-shaped in experience (years since draft) -- high at
# the start and end of a career (recent form matters most), low ("bottoms
# out") at the prime (a longer, more stable baseline is fine).
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
    return pd.read_csv(path, encoding="utf-8")


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


def build_vectors(season: str) -> pd.DataFrame:
    season_ids = [season, season_offset(season, 1), season_offset(season, 2)]
    current = load_season(season_ids[0])
    if current is None:
        raise FileNotFoundError(f"no player_stats file for {season}")

    first_year_lookup = load_undrafted_first_years()

    prior_by_id = [
        (f.set_index("PLAYER_ID") if f is not None else None)
        for f in (load_season(s) for s in season_ids[1:])
    ]

    season_start_year = int(season.split("-")[0])
    rows = []

    for _, bio in current.iterrows():
        pid = bio["PLAYER_ID"]

        # which of the (up to) 3 trailing seasons we actually have data for;
        # a missing season (rookie, or a gap -- e.g. injury, "ignore it for
        # now") is simply left out of the weighting rather than imputed.
        available = [0]
        for i, pf in enumerate(prior_by_id, start=1):
            if pf is not None and pid in pf.index:
                available.append(i)

        experience = experience_for(bio, season_start_year, len(available), first_year_lookup)
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

        for pt in PLAY_TYPES:
            freq_col, ppp_col = f"{pt}_FREQ_PCT", f"{pt}_PPP"
            decayed_freq = 0.0
            decayed_ppp = 0.0
            for i in available:
                if i == 0:
                    row = bio
                else:
                    pf = prior_by_id[i - 1]
                    row = pf.loc[pid] if pid in pf.index else None

                freq_val = row.get(freq_col) if row is not None else None
                ppp_val = row.get(ppp_col) if row is not None else None
                if row is None or pd.isna(freq_val) or pd.isna(ppp_val):
                    freq_val, ppp_val = BASELINE_FREQ, BASELINE_PPP

                decayed_freq += weights[i] * freq_val
                decayed_ppp += weights[i] * ppp_val

            out[f"{pt}_FREQ_DECAYED"] = decayed_freq
            out[f"{pt}_PPP_DECAYED"] = decayed_ppp

        rows.append(out)

    df = pd.DataFrame(rows)

    # z-score-style normalization: player.md asks for
    # z(value) - z(0) = (value - mean)/std - (0 - mean)/std = value/std,
    # so a player who never runs a given play type scores exactly 0 on it
    # instead of a large negative z-score.
    for pt in PLAY_TYPES:
        freq_col, ppp_col = f"{pt}_FREQ_DECAYED", f"{pt}_PPP_DECAYED"
        freq_std = df[freq_col].std(ddof=0)
        ppp_std = df[ppp_col].std(ddof=0)
        freq_score = df[freq_col] / freq_std if freq_std else df[freq_col] * 0.0
        ppp_score = df[ppp_col] / ppp_std if ppp_std else df[ppp_col] * 0.0
        df[f"{pt}_FREQ_SCORE"] = freq_score
        df[f"{pt}_PPP_SCORE"] = ppp_score
        df[f"{pt}_SCORE"] = np.sqrt(np.clip(freq_score, 0, None) * np.clip(ppp_score, 0, None))

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="e.g. 2025-26")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    df = build_vectors(args.season)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"player_playtype_vectors_{args.season}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} player vectors to {out_path}")


if __name__ == "__main__":
    main()
