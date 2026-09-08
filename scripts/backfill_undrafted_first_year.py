"""
One-off backfill: for every undrafted player appearing anywhere in
data/player_stats/*.csv, looks up their real NBA debut season (FROM_YEAR)
via nba_api's CommonPlayerInfo endpoint, since DRAFT_YEAR is "Undrafted" for
these players and build_defense_vectors.py / build_playtype_vectors.py
otherwise fall back to a data-availability proxy (number of seasons on
file) rather than true career experience.

Writes data/undrafted_first_year.csv: PLAYER_ID, PLAYER_NAME, FROM_YEAR.
Resumable -- skips PLAYER_IDs already present in the output file, so a
rate-limit failure partway through doesn't lose prior progress.
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

DATA_DIR = Path("data")
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
OUT_PATH = DATA_DIR / "undrafted_first_year.csv"

REQUEST_DELAY = 0.6
MAX_RETRIES = 3


def collect_undrafted_players() -> pd.DataFrame:
    rows = {}
    for f in sorted(PLAYER_STATS_DIR.glob("player_stats_*.csv")):
        df = pd.read_csv(f, encoding="utf-8")
        if "DRAFT_YEAR" not in df.columns:
            continue
        mask = df["DRAFT_YEAR"].isna() | (
            df["DRAFT_YEAR"].astype(str).str.strip().str.lower().isin(["", "undrafted", "nan"])
        )
        for _, r in df[mask].iterrows():
            rows[int(r["PLAYER_ID"])] = r["PLAYER_NAME"]
    out = pd.DataFrame(
        [{"PLAYER_ID": pid, "PLAYER_NAME": name} for pid, name in rows.items()]
    ).sort_values("PLAYER_ID")
    return out


def fetch_from_year(player_id: int) -> int | None:
    for attempt in range(MAX_RETRIES):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30)
            df = info.get_data_frames()[0]
            if df.empty:
                return None
            val = df.iloc[0]["FROM_YEAR"]
            return int(val) if pd.notna(val) and str(val).strip() != "" else None
        except Exception as e:
            print(f"  player_id={player_id} attempt {attempt + 1} failed: {e}")
            time.sleep(2.0)
    return None


def main():
    undrafted = collect_undrafted_players()
    print(f"{len(undrafted)} unique undrafted players found across all seasons")

    if OUT_PATH.exists():
        done = pd.read_csv(OUT_PATH)
        done_ids = set(done["PLAYER_ID"])
        results = done.to_dict("records")
        print(f"resuming: {len(done_ids)} already fetched")
    else:
        done_ids = set()
        results = []

    remaining = undrafted[~undrafted["PLAYER_ID"].isin(done_ids)]
    print(f"{len(remaining)} remaining to fetch")

    for i, (_, row) in enumerate(remaining.iterrows()):
        pid, name = int(row["PLAYER_ID"]), row["PLAYER_NAME"]
        from_year = fetch_from_year(pid)
        results.append({"PLAYER_ID": pid, "PLAYER_NAME": name, "FROM_YEAR": from_year})
        status = from_year if from_year is not None else "NOT FOUND"
        print(f"[{i + 1}/{len(remaining)}] {name} ({pid}): FROM_YEAR={status}")

        if (i + 1) % 25 == 0:
            pd.DataFrame(results).sort_values("PLAYER_ID").to_csv(OUT_PATH, index=False)

        time.sleep(REQUEST_DELAY)

    pd.DataFrame(results).sort_values("PLAYER_ID").to_csv(OUT_PATH, index=False)
    n_missing = pd.DataFrame(results)["FROM_YEAR"].isna().sum()
    print(f"done: {len(results)} players written to {OUT_PATH} ({n_missing} unresolved)")


if __name__ == "__main__":
    main()
