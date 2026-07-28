"""
One-off backfill: adds the raw inputs Dean Oliver's Defensive Win Shares
formula needs (see defense.md) to existing data/player_stats/*.csv and
data/team_stats/*.csv files, without re-running the full pull_nba_data.py
for every already-pulled historical season.

Player side: DEF_RATING from LeagueDashPlayerStats(measure_type="Advanced")
-- NBA.com's individual defensive-rating estimate, the DRtg input to the
formula. Team side: POSS, PACE from the same Advanced measure type, and PTS,
MIN from measure_type="Base".

Usage:
    python scripts/backfill_dws_inputs.py
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

PLAYER_STATS_DIR = Path("data") / "player_stats"
TEAM_STATS_DIR = Path("data") / "team_stats"
REQUEST_DELAY = 0.6
SEASON_TYPE = "Regular Season"


def fetch_player_def_rating(season: str) -> pd.DataFrame | None:
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, season_type_all_star=SEASON_TYPE, measure_type_detailed_defense="Advanced"
    )
    df = resp.get_data_frames()[0]
    if df.empty:
        return None
    return df[["PLAYER_ID", "DEF_RATING"]]


def fetch_team_dws_inputs(season: str) -> pd.DataFrame | None:
    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season, season_type_all_star=SEASON_TYPE, measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    base = leaguedashteamstats.LeagueDashTeamStats(
        season=season, season_type_all_star=SEASON_TYPE, measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    if adv.empty or base.empty:
        return None
    adv = adv[["TEAM_ID", "POSS", "PACE"]]
    base = base[["TEAM_ID", "PTS", "MIN"]]
    return adv.merge(base, on="TEAM_ID", how="inner")


def backfill_players():
    for path in sorted(PLAYER_STATS_DIR.glob("player_stats_*.csv")):
        season = path.stem.replace("player_stats_", "")
        df = pd.read_csv(path, encoding="utf-8")
        if "DEF_RATING" in df.columns:
            print(f"player {season}: already has DEF_RATING, skipping")
            continue

        result = None
        for attempt in range(3):
            result = fetch_player_def_rating(season)
            if result is not None:
                break
            print(f"player {season}: empty response on attempt {attempt + 1}, retrying...")
            time.sleep(2.0)

        if result is None:
            print(f"player {season}: DEF_RATING unavailable after retries")
            continue

        df = df.merge(result, on="PLAYER_ID", how="left")
        df.to_csv(path, index=False)
        print(f"player {season}: wrote DEF_RATING for {result['DEF_RATING'].notna().sum()}/{len(df)} players")
        time.sleep(REQUEST_DELAY)


def backfill_teams():
    for path in sorted(TEAM_STATS_DIR.glob("team_stats_*.csv")):
        season = path.stem.replace("team_stats_", "")
        df = pd.read_csv(path, encoding="utf-8")
        if {"POSS", "PACE", "PTS"}.issubset(df.columns):
            print(f"team {season}: already has POSS/PACE/PTS, skipping")
            continue

        result = None
        for attempt in range(3):
            result = fetch_team_dws_inputs(season)
            if result is not None:
                break
            print(f"team {season}: empty response on attempt {attempt + 1}, retrying...")
            time.sleep(2.0)

        if result is None:
            print(f"team {season}: POSS/PACE/PTS/MIN unavailable after retries")
            continue

        df = df.merge(result, on="TEAM_ID", how="left")
        df.to_csv(path, index=False)
        print(f"team {season}: wrote POSS/PACE/PTS/MIN for {len(df)} teams")
        time.sleep(REQUEST_DELAY)


def main():
    backfill_players()
    backfill_teams()


if __name__ == "__main__":
    main()
