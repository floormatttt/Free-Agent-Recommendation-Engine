"""
Pulls full team rosters (CommonTeamRoster) for every team for a given
season -- unlike the salary-ranking .txt dumps (only ~432 players with
meaningful guaranteed money), this includes two-way/camp/non-guaranteed
bodies, so it's the more complete source for "who is actually on this
team's roster."

Output: data/rosters/rosters_<season>.csv, one row per player with columns
TEAM, PLAYER_NAME, PLAYER_ID, POSITION, HEIGHT, WEIGHT, AGE, EXP, SCHOOL,
HOW_ACQUIRED.

Usage:
    python scripts/pull_rosters.py --season 2026-27
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

REQUEST_DELAY = 0.6  # seconds between calls, be polite to stats.nba.com

COLUMNS = [
    "TEAM", "PLAYER_NAME", "PLAYER_ID", "POSITION", "HEIGHT", "WEIGHT",
    "AGE", "EXP", "SCHOOL", "HOW_ACQUIRED",
]


def fetch_roster(team_abbr: str, team_id: int, season: str) -> pd.DataFrame:
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    roster = roster.rename(columns={"PLAYER": "PLAYER_NAME"})
    roster["TEAM"] = team_abbr
    return roster[COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27", help="e.g. 2026-27")
    args = parser.parse_args()

    all_teams = sorted(teams.get_teams(), key=lambda t: t["abbreviation"])

    frames = []
    for i, team in enumerate(all_teams):
        print(f"[{i + 1}/{len(all_teams)}] {team['abbreviation']}...")
        frames.append(fetch_roster(team["abbreviation"], team["id"], args.season))
        if i < len(all_teams) - 1:
            time.sleep(REQUEST_DELAY)

    df = pd.concat(frames, ignore_index=True)

    output_dir = Path("data/rosters")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rosters_{args.season}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nWrote {len(df)} players across {df['TEAM'].nunique()} teams to {output_path}")


if __name__ == "__main__":
    main()
