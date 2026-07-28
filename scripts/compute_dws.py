"""
Computes Defensive Win Shares per player-season using Dean Oliver's formula
(Basketball-Reference's exact methodology), rather than depending on
NBA.com's own DEF_WS field -- which turned out to be entirely absent for 5
of 12 historical seasons on file (see defense.md's Known Limitations).
Writes DWS_RAW and DWS_PER_GAME back into each data/player_stats/*.csv.

Formula (all quantities are season totals for that player/team/league):

    marginal_defense = (MP_player / MP_team) * Poss_def_team
                        * (1.08 * PPP_league - DRtg_player / 100)
    marginal_pts_per_win = 0.32 * PPG_league * (Pace_team / Pace_league)
    DWS = marginal_defense / marginal_pts_per_win

Inputs, all already on file after backfill_dws_inputs.py / pull_nba_data.py:
    Player:  MIN (MP_player), GP, DEF_RATING (DRtg_player -- NBA.com's own
             individual defensive-rating estimate, an Oliver-style formula
             using team defense + the player's steals/blocks/rebounds/fouls,
             not an on-court/off-court split).
    Team:    POSS (used as Poss_def_team -- a team's defensive possessions
             are approximated by its own tempo-free possession count, the
             standard simplification here since possessions for/against
             balance out over a full season), PACE, PTS, GP. MP_team is
             NOT taken from the team_stats MIN column -- that column is the
             team's own single-unit game minutes (~48/game), not the
             roster's combined 5-on-court minutes Oliver's formula expects.
             MP_team is instead the sum of that team's own players' MIN
             from player_stats, computed directly below.
    League:  PPP_league = sum(team PTS) / sum(team POSS)
             PPG_league = sum(team PTS) / sum(team GP)
             Pace_league = GP-weighted mean of team PACE

DWS_PER_GAME = DWS_RAW / player's own GP -- requested as a per-game rate so
it's comparable across players with different games-played totals, the same
role MIN played in the earlier per-48 approach this replaces.

A player's TEAM column reflects only their most recent team for a season
(the same simplification already used elsewhere in this pipeline, e.g. the
old team-DRTG shrinkage prior lookup) -- a traded player's MP_team/Poss_def
etc. come from that one team, not a stint-weighted blend across teams.

Usage:
    python scripts/compute_dws.py                  # all seasons on file
    python scripts/compute_dws.py --season 2025-26  # just one season
"""

import argparse
from pathlib import Path

import pandas as pd
from nba_api.stats.static import teams as static_teams

PLAYER_STATS_DIR = Path("data") / "player_stats"
TEAM_STATS_DIR = Path("data") / "team_stats"

TEAM_ABBREV_BY_ID = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}


def compute_season(season: str) -> bool:
    player_path = PLAYER_STATS_DIR / f"player_stats_{season}.csv"
    team_path = TEAM_STATS_DIR / f"team_stats_{season}.csv"
    if not player_path.exists() or not team_path.exists():
        print(f"{season}: missing player_stats or team_stats file, skipping")
        return False

    players = pd.read_csv(player_path, encoding="utf-8")
    teams = pd.read_csv(team_path, encoding="utf-8")

    required_player_cols = {"MIN", "GP", "DEF_RATING", "TEAM"}
    required_team_cols = {"POSS", "PACE", "PTS", "GP"}
    if not required_player_cols.issubset(players.columns):
        print(f"{season}: player_stats missing {required_player_cols - set(players.columns)}, skipping")
        return False
    if not required_team_cols.issubset(teams.columns):
        print(f"{season}: team_stats missing {required_team_cols - set(teams.columns)}, skipping")
        return False

    teams = teams.copy()
    teams["TEAM"] = teams["TEAM_ID"].map(TEAM_ABBREV_BY_ID)

    pts_league = teams["PTS"].sum()
    poss_league = teams["POSS"].sum()
    gp_league_total = teams["GP"].sum()
    ppp_league = pts_league / poss_league
    ppg_league = pts_league / gp_league_total
    pace_league = (teams["PACE"] * teams["GP"]).sum() / gp_league_total

    team_lookup = teams.set_index("TEAM")[["POSS", "PACE"]].rename(
        columns={"POSS": "POSS_DEF_TEAM", "PACE": "PACE_TEAM"}
    )
    team_lookup["MP_TEAM"] = players.groupby("TEAM")["MIN"].sum()

    df = players.merge(team_lookup, on="TEAM", how="left")
    missing_team = df["MP_TEAM"].isna()
    if missing_team.any():
        print(f"{season}: {missing_team.sum()} players have no matching team row, DWS left NaN for them")

    mp_player = df["MIN"]
    drtg_player = df["DEF_RATING"]
    marginal_defense = (mp_player / df["MP_TEAM"]) * df["POSS_DEF_TEAM"] * (
        1.08 * ppp_league - drtg_player / 100.0
    )
    marginal_pts_per_win = 0.32 * ppg_league * (df["PACE_TEAM"] / pace_league)

    df["DWS_RAW"] = marginal_defense / marginal_pts_per_win
    df["DWS_PER_GAME"] = df["DWS_RAW"] / df["GP"].clip(lower=1)

    players["DWS_RAW"] = df["DWS_RAW"]
    players["DWS_PER_GAME"] = df["DWS_PER_GAME"]
    players.to_csv(player_path, index=False)

    print(
        f"{season}: wrote DWS_RAW/DWS_PER_GAME for {players['DWS_PER_GAME'].notna().sum()}/{len(players)} players "
        f"(PPP_league={ppp_league:.3f}, PPG_league={ppg_league:.2f}, Pace_league={pace_league:.2f})"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default=None, help="e.g. 2025-26; omit to process every season on file")
    args = parser.parse_args()

    if args.season:
        compute_season(args.season)
        return

    for path in sorted(PLAYER_STATS_DIR.glob("player_stats_*.csv")):
        season = path.stem.replace("player_stats_", "")
        compute_season(season)


if __name__ == "__main__":
    main()
