"""
Pulls the player- and team-level dataset described in Data.txt using nba_api.

Output:
    data/player_stats_<season>.csv
    data/team_stats_<season>.csv

Usage:
    python pull_nba_data.py --season 2023-24 --season-type "Regular Season"
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from nba_api.stats.endpoints import (
    leaguedashplayerbiostats,
    leaguedashplayerstats,
    leaguedashptdefend,
    leaguedashptteamdefend,
    leaguedashteamstats,
    leaguehustlestatsplayer,
    leaguehustlestatsteam,
    synergyplaytypes,
    teamplayeronoffsummary,
)
from nba_api.stats.static import teams as static_teams

REQUEST_DELAY = 0.6  # seconds between calls, be polite to stats.nba.com

# (PlayType value used by the API, short label used in output columns)
PLAY_TYPES = [
    ("Isolation", "ISOLATION"),
    ("PRBallHandler", "PNR_HANDLER"),
    ("PRRollman", "PNR_ROLLMAN"),
    ("Postup", "POST_UP"),
    ("Spotup", "SPOT_UP"),
    ("Handoff", "HANDOFF"),
    ("Cut", "CUT"),
    ("OffScreen", "OFF_SCREEN"),
    ("OffRebound", "PUT_BACKS"),
]


def _combine_stints(group: pd.DataFrame) -> pd.Series:
    """Collapse multiple team-stint rows (e.g. a player traded mid-season)
    into one possession-weighted row."""
    total_poss = group["POSS"].sum()
    if total_poss == 0:
        return pd.Series({"POSS_PCT": 0.0, "PPP": 0.0})
    freq = (group["POSS_PCT"] * group["POSS"]).sum() / total_poss
    ppp = (group["PPP"] * group["POSS"]).sum() / total_poss
    return pd.Series({"POSS_PCT": freq, "PPP": ppp})


def dedupe_rim_defense(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """LeagueDashPtDefend/LeagueDashPtTeamDefend split traded players/teams
    into one row per team stint; collapse those into a single attempts-
    weighted row so the id column stays unique."""

    def combine(group: pd.DataFrame) -> pd.Series:
        fgm = group["FGM_LT_06"].sum()
        fga = group["FGA_LT_06"].sum()
        dfg_pct = fgm / fga if fga else 0.0
        if fga:
            opp_fg_pct = (group["NS_LT_06_PCT"] * group["FGA_LT_06"]).sum() / fga
            diff_pct = (group["PLUSMINUS"] * group["FGA_LT_06"]).sum() / fga
        else:
            opp_fg_pct = 0.0
            diff_pct = 0.0
        return pd.Series(
            {
                "FGM_LT_06": fgm,
                "FGA_LT_06": fga,
                "LT_06_PCT": dfg_pct,
                "NS_LT_06_PCT": opp_fg_pct,
                "PLUSMINUS": diff_pct,
            }
        )

    return df.groupby(id_col, as_index=False).apply(combine, include_groups=False)


def fetch_playtypes(player_or_team: str, season: str, season_type: str) -> pd.DataFrame:
    """player_or_team: 'P' for player, 'T' for team."""
    id_col = "PLAYER_ID" if player_or_team == "P" else "TEAM_ID"
    merged = None
    for play_type, label in PLAY_TYPES:
        resp = synergyplaytypes.SynergyPlayTypes(
            player_or_team_abbreviation=player_or_team,
            play_type_nullable=play_type,
            type_grouping_nullable="offensive",
            season=season,
            season_type_all_star=season_type,
        )
        df = resp.get_data_frames()[0]
        df = df[[id_col, "POSS_PCT", "PPP", "POSS"]]
        # Traded players get one row per team stint; collapse to a single
        # possession-weighted row so the id column stays unique.
        df = df.groupby(id_col, as_index=False).apply(
            lambda g: _combine_stints(g), include_groups=False
        )
        df = df.rename(columns={"POSS_PCT": f"{label}_FREQ_PCT", "PPP": f"{label}_PPP"})
        merged = df if merged is None else merged.merge(df, on=id_col, how="outer")
        time.sleep(REQUEST_DELAY)
    return merged


def fetch_on_court_def_rating(season: str, season_type: str) -> pd.DataFrame:
    """On-court DEF_RATING per player, for the perimeter-defense block in
    defense.md. TeamPlayerOnOffSummary is a per-team endpoint, so this loops
    over all 30 teams; a traded player ends up with one row per team stint,
    which are combined into a single minutes-weighted average."""
    rows = []
    for team in static_teams.get_teams():
        resp = teamplayeronoffsummary.TeamPlayerOnOffSummary(
            team_id=team["id"],
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
        )
        on_court = resp.get_data_frames()[1]
        rows.append(on_court[["VS_PLAYER_ID", "MIN", "DEF_RATING"]])
        time.sleep(REQUEST_DELAY)

    combined = pd.concat(rows, ignore_index=True).rename(columns={"VS_PLAYER_ID": "PLAYER_ID"})

    def weighted_avg(g: pd.DataFrame) -> pd.Series:
        total_min = g["MIN"].sum()
        if total_min == 0:
            return pd.Series({"ON_COURT_DEF_RATING": g["DEF_RATING"].mean(), "ON_COURT_MIN": 0.0})
        return pd.Series(
            {
                "ON_COURT_DEF_RATING": (g["DEF_RATING"] * g["MIN"]).sum() / total_min,
                "ON_COURT_MIN": total_min,
            }
        )

    return combined.groupby("PLAYER_ID", as_index=False).apply(weighted_avg, include_groups=False)


def fetch_player_data(season: str, season_type: str) -> pd.DataFrame:
    print("Fetching player bio stats...")
    bios = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
        season=season, season_type_all_star=season_type
    ).get_data_frames()[0]
    bios = bios[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "AGE",
            "PLAYER_HEIGHT",
            "DRAFT_YEAR",
            "DRAFT_ROUND",
            "DRAFT_NUMBER",
            "GP",
        ]
    ].rename(columns={"DRAFT_NUMBER": "DRAFT_PICK", "TEAM_ABBREVIATION": "TEAM"})
    time.sleep(REQUEST_DELAY)

    print("Fetching player base stats (minutes)...")
    base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, season_type_all_star=season_type, measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    base = base[["PLAYER_ID", "MIN"]]
    time.sleep(REQUEST_DELAY)

    print("Fetching player advanced stats (rebounding %)...")
    adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, season_type_all_star=season_type, measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    adv = adv[["PLAYER_ID", "OREB_PCT", "DREB_PCT"]]
    time.sleep(REQUEST_DELAY)

    print("Fetching player usage stats (% steals, % blocks)...")
    usage = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, season_type_all_star=season_type, measure_type_detailed_defense="Usage"
    ).get_data_frames()[0]
    usage = usage[["PLAYER_ID", "PCT_STL", "PCT_BLK"]].rename(
        columns={"PCT_STL": "PCT_STEALS", "PCT_BLK": "PCT_BLOCKS"}
    )
    time.sleep(REQUEST_DELAY)

    print("Fetching player hustle stats (deflections, contested 3s, def box outs)...")
    hustle = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
        season=season, season_type_all_star=season_type
    ).get_data_frames()[0]
    hustle = hustle[["PLAYER_ID", "DEFLECTIONS", "CONTESTED_SHOTS_3PT", "DEF_BOXOUTS"]]
    time.sleep(REQUEST_DELAY)

    print("Fetching player rim protection stats (< 6ft)...")
    rim = leaguedashptdefend.LeagueDashPtDefend(
        season=season, season_type_all_star=season_type, defense_category="Less Than 6Ft"
    ).get_data_frames()[0]
    rim = rim.rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"})
    rim = dedupe_rim_defense(rim, "PLAYER_ID")
    rim = rim[
        ["PLAYER_ID", "FGM_LT_06", "FGA_LT_06", "LT_06_PCT", "NS_LT_06_PCT", "PLUSMINUS"]
    ].rename(
        columns={
            "FGM_LT_06": "RIM_DFGM",
            "FGA_LT_06": "RIM_DFGA",
            "LT_06_PCT": "RIM_DFG_PCT",
            "NS_LT_06_PCT": "RIM_OPP_FG_PCT",
            "PLUSMINUS": "RIM_DIFF_PCT",
        }
    )
    time.sleep(REQUEST_DELAY)

    print("Fetching player play type stats (this takes a while, 9 calls)...")
    playtypes = fetch_playtypes("P", season, season_type)

    print("Fetching on-court DEF_RATING (this takes a while, 30 calls)...")
    on_off = fetch_on_court_def_rating(season, season_type)

    df = bios
    for other in (base, adv, usage, hustle, rim, playtypes, on_off):
        df = df.merge(other, on="PLAYER_ID", how="left")
    return df


def fetch_team_data(season: str, season_type: str) -> pd.DataFrame:
    print("Fetching team base stats (blocks)...")
    base = leaguedashteamstats.LeagueDashTeamStats(
        season=season, season_type_all_star=season_type, measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    base = base[["TEAM_ID", "TEAM_NAME", "BLK"]]
    time.sleep(REQUEST_DELAY)

    print("Fetching team advanced stats (def rating, rebounding %)...")
    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season, season_type_all_star=season_type, measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    adv = adv[["TEAM_ID", "DEF_RATING", "DREB_PCT"]]
    time.sleep(REQUEST_DELAY)

    print("Fetching team hustle stats (deflections, contested 3s, def box outs)...")
    hustle = leaguehustlestatsteam.LeagueHustleStatsTeam(
        season=season, season_type_all_star=season_type
    ).get_data_frames()[0]
    hustle = hustle[["TEAM_ID", "DEFLECTIONS", "CONTESTED_SHOTS_3PT", "DEF_BOXOUTS"]]
    time.sleep(REQUEST_DELAY)

    print("Fetching team rim protection stats (< 6ft)...")
    rim = leaguedashptteamdefend.LeagueDashPtTeamDefend(
        season=season, season_type_all_star=season_type, defense_category="Less Than 6Ft"
    ).get_data_frames()[0]
    rim = dedupe_rim_defense(rim, "TEAM_ID")
    rim = rim[
        ["TEAM_ID", "FGM_LT_06", "FGA_LT_06", "LT_06_PCT", "NS_LT_06_PCT", "PLUSMINUS"]
    ].rename(
        columns={
            "FGM_LT_06": "RIM_DFGM",
            "FGA_LT_06": "RIM_DFGA",
            "LT_06_PCT": "RIM_DFG_PCT",
            "NS_LT_06_PCT": "RIM_OPP_FG_PCT",
            "PLUSMINUS": "RIM_DIFF_PCT",
        }
    )
    time.sleep(REQUEST_DELAY)

    print("Fetching team play type stats (this takes a while, 9 calls)...")
    playtypes = fetch_playtypes("T", season, season_type)

    df = base
    for other in (adv, hustle, rim, playtypes):
        df = df.merge(other, on="TEAM_ID", how="left")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2023-24", help="e.g. 2023-24")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        choices=["Regular Season", "Playoffs", "Pre Season"],
    )
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    player_dir = out_dir / "player_stats"
    team_dir = out_dir / "team_stats"
    player_dir.mkdir(parents=True, exist_ok=True)
    team_dir.mkdir(parents=True, exist_ok=True)

    player_df = fetch_player_data(args.season, args.season_type)
    player_df["MIN"] = player_df["MIN"].round(2)
    player_df = player_df.drop_duplicates()
    player_path = player_dir / f"player_stats_{args.season}.csv"
    player_df.to_csv(player_path, index=False)
    print(f"Wrote {len(player_df)} players to {player_path}")

    team_df = fetch_team_data(args.season, args.season_type)
    team_df = team_df.drop_duplicates()
    team_path = team_dir / f"team_stats_{args.season}.csv"
    team_df.to_csv(team_path, index=False)
    print(f"Wrote {len(team_df)} teams to {team_path}")


if __name__ == "__main__":
    main()
