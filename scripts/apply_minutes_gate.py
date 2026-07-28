"""
Post-processes any of this repo's contract-prediction output CSVs to fix a
pattern the user flagged directly: several models overpredict players who
barely played -- deep-bench/two-way guys whose 12-category vector looks
statistically "average" in small samples (or leans on player.md's baseline
substitution for play types they never ran), producing a plausible-looking
prediction for someone who's actually a replacement-level roster afterthought.

Computes each player-season's trailing 3-year total minutes (that season
plus up to 2 prior, matching the same trailing-window convention
build_playtype_vectors.py already uses for experience-decay -- no imputation
for seasons outside data/player_stats/'s range). Below MINUTES_THRESHOLD
(2,400 -- roughly a full healthy season, since this is a 3-year *total*),
the prediction is scaled toward 0 by sqrt(total_min / threshold), a smooth
curve rather than a hard floor: a player at the threshold keeps their full
prediction, a player at 1/4 of the threshold keeps half of it (sqrt(0.25) =
0.5), and a player with ~0 minutes is scaled to ~0. The square root shape
(rather than linear) means the shrinkage is gentle near the threshold and
only gets severe for players with very little playing time at all -- e.g.
sqrt(0.75) = 0.87 (barely touched) vs sqrt(0.05) = 0.22 (nearly untouched).

Works on either of this repo's prediction-file shapes:
  - one SEASON column: minutes are computed for that same SEASON.
  - PREV_SEASON/TARGET_SEASON columns (build_contract_model_v3.py's shape):
    minutes are computed for PREV_SEASON, since that's the season whose
    vector/usage actually feeds the prediction.

build_contract_model_v3.py imports three_year_total_minutes() directly and
applies the gate as a standard final step -- this CLI is for applying it to
some other prediction file on its own.

Usage:
    python scripts/apply_minutes_gate.py --input data/salary_prediction/contract_model_v3_predictions.csv --season-col PREV_SEASON
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
MINUTES_THRESHOLD = 2400


def season_offset(season: str, n: int) -> str:
    start, end = season.split("-")
    new_start = int(start) - n
    new_end = int(end) - n
    return f"{new_start}-{new_end:02d}"


_minutes_cache: dict[str, pd.Series] = {}


def minutes_for_season(season: str) -> pd.Series:
    if season not in _minutes_cache:
        path = PLAYER_STATS_DIR / f"player_stats_{season}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8")
            _minutes_cache[season] = df.set_index("PLAYER_ID")["MIN"]
        else:
            _minutes_cache[season] = pd.Series(dtype=float)
    return _minutes_cache[season]


def three_year_total_minutes(player_ids: pd.Series, seasons: pd.Series) -> np.ndarray:
    totals = np.zeros(len(player_ids))
    for i in range(3):  # current season + up to 2 trailing
        offset_seasons = seasons.apply(lambda s: season_offset(s, i))
        for season in offset_seasons.unique():
            mask = offset_seasons == season
            mins = minutes_for_season(season)
            ids = player_ids[mask]
            totals[mask.to_numpy()] += ids.map(mins).fillna(0.0).to_numpy()
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--season-col", default="SEASON", help="which column names the season to pull minutes for")
    parser.add_argument("--threshold", type=float, default=MINUTES_THRESHOLD)
    parser.add_argument("--output", default=None, help="defaults to <input>_minutesgated.csv")
    args = parser.parse_args()

    in_path = Path(args.input)
    df = pd.read_csv(in_path)

    df["TOTAL_MIN_3YR"] = three_year_total_minutes(df["PLAYER_ID"], df[args.season_col])
    fraction = (df["TOTAL_MIN_3YR"] / args.threshold).clip(upper=1.0).clip(lower=0.0)
    shrink = np.sqrt(fraction)
    below = df["TOTAL_MIN_3YR"] < args.threshold

    n_gated = int(below.sum())
    print(f"{n_gated} of {len(df)} rows ({100*n_gated/len(df):.1f}%) below {args.threshold:.0f} trailing-3yr minutes -- "
          f"shrinking their prediction toward 0 by sqrt(total_min/{args.threshold:.0f}).")

    pred_col = "PREDICTED_SALARY_PCT_OF_CAP"
    df["SHRINK_FACTOR"] = shrink
    df[pred_col] = df[pred_col] * shrink
    if "RESIDUAL_PP" in df.columns:
        actual_col = "SALARY_PCT_OF_CAP" if "SALARY_PCT_OF_CAP" in df.columns else "ACTUAL_SALARY_PCT_OF_CAP"
        df["RESIDUAL_PP"] = df[actual_col] - df[pred_col]

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_minutesgated.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
