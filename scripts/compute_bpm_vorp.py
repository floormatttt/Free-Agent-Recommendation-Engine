"""
Reimplementation of Daniel Myers' BPM 2.0 / VORP (basketball-reference.com/
about/bpm2.html) from nba_api box score data, since basketball-reference
itself blocks automated fetches (403) and the exact BPM 2.0 regression
coefficients aren't reproduced anywhere else scrapeable -- so the published
article text (supplied by the user) is the only source, transcribed here.

Purpose: an external, non-circular "actual impact" ground truth for the
Surplus/Regret free-agency eval design (see docs/free_agency_2026.md) --
BPM/VORP never touches this repo's own roster-vector/need/win% models, so
using it to score free-agent signings doesn't grade the model against
itself.

Audited (see the published BPM/VORP artifact): an earlier version of this
script omitted the "points adjusted for team shooting context" term
entirely, which was traced as the dominant cause of a ~2x magnitude
inflation (e.g. computed 2024-25 BPM leaders exceeded the real all-time
record). The article describes the adjustment step (compare each team's
points-per-adjusted-shot-attempt to "the baseline ... used by the
regression") but never publishes that baseline constant.

A first attempt used each season's own league-wide average points-per-TSA as
the reference point -- defensible in spirit (a league-average-context team
gets zero adjustment) but empirically far too weak: it only moved 2024-25's
computed BPM leaders by ~1 point, while the article's own worked example
(2017 LeBron James, PTS/100 34.9 adjusted to 30.4, an 11.5% cut) implies an
adjustment about 3x larger than a single season's team-vs-league-average gap
can produce. Pulling real 2016-17 Cleveland/league totals and solving that
worked example backward gives an implied baseline of ~0.998 points per TSA
(TSA = FGA + 0.44*FTA) -- well below the 2016-17 league average of 1.105,
confirming the true baseline is not any single season's league average but a
lower, fixed constant (plausibly reflecting the regression's much longer,
lower-efficiency historical basis window -- box scores are described as
going back to 1974). BASELINE_PTS_PER_TSA below uses that calibrated value,
fixed across every season rather than recomputed per season. This is the
best available anchor without access to basketball-reference's actual fit,
but it rests on a single worked example, not a from-source constant --
treat it as calibrated, not verified.

Documented simplifications still open vs. the real BPM 2.0 (all because the
required inputs aren't available from bulk nba_api endpoints):
  - Season-level only (no game-level summation).
  - No score-effects (average-lead) correction to the team-adjustment anchor.
  - Position/offensive-role estimates use the season-total regression only
    (skips the "blend with 50 minutes at listed position/role 4.0" prior),
    since bulk endpoints don't expose a per-player listed primary position.
    Audited and confirmed as a secondary, still-unfixed inflation source: a
    ball-dominant guard with an unusually high share of their team's blocks
    (e.g. 2024-25 Shai Gilgeous-Alexander) gets pulled toward a "big man"
    position estimate, inflating their assist coefficient.
  - Team-adjustment normalization (position/role mean -> 3.0, raw BPM mean ->
    team NET_RATING) is done as a single non-recursive shift-and-reclip, not
    the iterative solve the original describes.
These will produce numbers close to, but not identical to, basketball-
reference's published BPM/VORP -- treat this as an approximation built to
the published methodology, not a reproduction of their exact output.

Output: data/player_value/bpm_vorp_<season>.csv

Usage:
    python scripts/compute_bpm_vorp.py --season 2024-25
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

ROOT = Path(__file__).resolve().parent.parent
REQUEST_DELAY = 0.6

# ---- BPM (overall) regression: coefficients at position 1 (PG) and 5 (C) ----
BPM_POS = {
    "AST": (0.580, 1.034),
    "OREB": (0.613, 0.181),
    "DREB": (0.116, 0.181),
    "STL": (1.369, 1.008),
    "BLK": (1.327, 0.703),
}
BPM_CONST = {"PTS_ADJ": 0.860, "FG3M": 0.389, "TOV": -0.964, "PF": -0.367}
BPM_ROLE = {"FGA": (-0.560, -0.780), "FTA": (-0.246, -0.343)}
BPM_POS_ADJ = -0.818   # at position 1, linear to 0 at position >= 3
BPM_ROLE_ADJ = -2.774  # at role 1 (creator), linear to +2.774 at role 5

# ---- OBPM regression ----
OBPM_POS = {
    "TOV": (-0.579, -0.882),
    "OREB": (0.606, 0.422),
    "DREB": (-0.112, 0.103),
    "STL": (0.177, 0.294),
    "BLK": (0.725, 0.097),
}
OBPM_CONST = {"PTS_ADJ": 0.605, "FG3M": 0.477, "AST": 0.476, "PF": -0.439}
OBPM_ROLE = {"FGA": (-0.330, -0.472), "FTA": (-0.145, -0.208)}
OBPM_POS_ADJ = -1.698
OBPM_ROLE_ADJ = -0.860

REPLACEMENT_LEVEL = -2.0

# Calibrated from the article's 2017 LeBron James worked example (see module
# docstring) -- not a from-source constant, but the best available anchor.
BASELINE_PTS_PER_TSA = 0.9982


def interp(lo_hi, x):
    lo, hi = lo_hi
    return lo + (np.clip(x, 1, 5) - 1) / 4 * (hi - lo)


def fetch(season: str):
    kwargs = dict(season=season, season_type_all_star="Regular Season")
    p_tot = leaguedashplayerstats.LeagueDashPlayerStats(
        measure_type_detailed_defense="Base", per_mode_detailed="Totals", **kwargs
    ).get_data_frames()[0]
    time.sleep(REQUEST_DELAY)
    p_100 = leaguedashplayerstats.LeagueDashPlayerStats(
        measure_type_detailed_defense="Base", per_mode_detailed="Per100Possessions", **kwargs
    ).get_data_frames()[0]
    time.sleep(REQUEST_DELAY)
    t_tot = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense="Base", per_mode_detailed="Totals", **kwargs
    ).get_data_frames()[0]
    time.sleep(REQUEST_DELAY)
    t_adv = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense="Advanced", per_mode_detailed="Totals", **kwargs
    ).get_data_frames()[0]
    time.sleep(REQUEST_DELAY)
    return p_tot, p_100, t_tot, t_adv


def team_pts_per_tsa(t_tot: pd.DataFrame) -> pd.Series:
    """Each team's season PTS per true-shot-attempt (TSA = FGA + 0.44*FTA),
    compared against BASELINE_PTS_PER_TSA (see module docstring) rather than
    a per-season league average."""
    tsa = t_tot["FGA"] + 0.44 * t_tot["FTA"]
    ppt = t_tot["PTS"] / tsa.replace(0, np.nan)
    return pd.Series(ppt.values, index=t_tot["TEAM_ID"])


def compute_position_and_role(p_tot: pd.DataFrame, t_tot: pd.DataFrame) -> pd.DataFrame:
    team_cols = t_tot.set_index("TEAM_ID")[["REB", "STL", "PF", "AST", "BLK", "PTS", "FGA", "FTA", "MIN"]]
    df = p_tot.merge(team_cols, on="TEAM_ID", suffixes=("", "_TEAM"))

    pct_trb = df["REB"] / df["REB_TEAM"].replace(0, np.nan)
    pct_stl = df["STL"] / df["STL_TEAM"].replace(0, np.nan)
    pct_pf = df["PF"] / df["PF_TEAM"].replace(0, np.nan)
    pct_ast = df["AST"] / df["AST_TEAM"].replace(0, np.nan)
    pct_blk = df["BLK"] / df["BLK_TEAM"].replace(0, np.nan)

    position = (
        2.130 + 8.668 * pct_trb.fillna(0) - 2.486 * pct_stl.fillna(0)
        + 0.992 * pct_pf.fillna(0) - 3.536 * pct_ast.fillna(0) + 1.667 * pct_blk.fillna(0)
    )
    df["POSITION_RAW"] = position.clip(1, 5)

    tsa = df["FGA"] + 0.44 * df["FTA"]
    tsa_team = df["FGA_TEAM"] + 0.44 * df["FTA_TEAM"]
    team_pts_per_tsa = df["PTS_TEAM"] / tsa_team.replace(0, np.nan)
    threshold_rate = team_pts_per_tsa - 0.33
    thresh_pts = df["PTS"] - threshold_rate * tsa
    thresh_pts_team = df.groupby("TEAM_ID")["PTS"].transform(
        lambda s: (df.loc[s.index, "PTS"] - threshold_rate.loc[s.index] * tsa.loc[s.index]).sum()
    )
    pct_thresh = (thresh_pts / thresh_pts_team.replace(0, np.nan)).fillna(0)

    role = 6.00 - 6.642 * pct_ast.fillna(0) - 8.544 * pct_thresh
    df["ROLE_RAW"] = role.clip(1, 5)

    for col in ("POSITION_RAW", "ROLE_RAW"):
        w = df["MIN"]
        team_mean = df.groupby("TEAM_ID").apply(
            lambda g: np.average(g[col], weights=g["MIN"].clip(lower=1e-6))
        )
        df[col] = df.apply(lambda r: r[col] + (3.0 - team_mean.loc[r["TEAM_ID"]]), axis=1)
        df[col] = df[col].clip(1, 5)

    return df[["PLAYER_ID", "POSITION_RAW", "ROLE_RAW"]].rename(
        columns={"POSITION_RAW": "POSITION_EST", "ROLE_RAW": "ROLE_EST"})


def raw_component_sum(p100: pd.DataFrame, position: pd.Series, role: pd.Series,
                       pos_coefs: dict, const_coefs: dict, role_coefs: dict) -> pd.Series:
    total = pd.Series(0.0, index=p100.index)
    for stat, lohi in pos_coefs.items():
        total += interp(lohi, position) * p100[stat]
    for stat, coef in const_coefs.items():
        total += coef * p100[stat]
    for stat, lohi in role_coefs.items():
        total += interp(lohi, role) * p100[stat]
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2024-25")
    args = ap.parse_args()

    print(f"Fetching nba_api data for {args.season}...")
    p_tot, p_100, t_tot, t_adv = fetch(args.season)

    pr = compute_position_and_role(p_tot, t_tot)
    df = p_100.merge(pr, on="PLAYER_ID").merge(
        p_tot[["PLAYER_ID", "MIN", "GP", "TEAM_ID"]], on=["PLAYER_ID", "TEAM_ID"], suffixes=("", "_TOT"))
    df = df.rename(columns={"MIN_TOT": "MIN_TOTAL"})

    # Adjust each player's per-100-poss points for their team's shooting
    # context: teams whose real points-per-TSA ran above the league average
    # get their players' points scaled back down toward it, and vice versa
    # (see team_pts_per_tsa() / module docstring for why league average is
    # used as the reference point).
    team_ppt = team_pts_per_tsa(t_tot)
    df["TSA_100"] = df["FGA"] + 0.44 * df["FTA"]
    df["PTS_ADJ"] = df["PTS"] - (df["TEAM_ID"].map(team_ppt) - BASELINE_PTS_PER_TSA) * df["TSA_100"]

    position, role = df["POSITION_EST"], df["ROLE_EST"]

    raw_bpm = raw_component_sum(df, position, role, BPM_POS, BPM_CONST, BPM_ROLE)
    pos_adj = np.where(position >= 3, 0.0, (3 - position) / 2 * BPM_POS_ADJ)
    role_adj = (3 - role) / 2 * BPM_ROLE_ADJ
    df["RAW_BPM"] = raw_bpm + pos_adj + role_adj

    raw_obpm = raw_component_sum(df, position, role, OBPM_POS, OBPM_CONST, OBPM_ROLE)
    pos_adj_o = np.where(position >= 3, 0.0, (3 - position) / 2 * OBPM_POS_ADJ)
    role_adj_o = (3 - role) / 2 * OBPM_ROLE_ADJ
    df["RAW_OBPM"] = raw_obpm + pos_adj_o + role_adj_o

    t_adv = t_adv.copy()
    t_tot_min = t_tot.set_index("TEAM_ID")["MIN"]
    t_adv["NET_RATING"] = t_adv["OFF_RATING"] - t_adv["DEF_RATING"]
    league_avg_off = np.average(t_adv["OFF_RATING"], weights=t_adv["POSS"])
    t_adv["OFF_ABOVE_AVG"] = t_adv["OFF_RATING"] - league_avg_off
    t_adv = t_adv.set_index("TEAM_ID")

    def team_shift(group, raw_col, anchor_series):
        w = group["MIN_TOTAL"].clip(lower=1e-6)
        team_avg = np.average(group[raw_col], weights=w)
        anchor = anchor_series.loc[group.name]
        return group[raw_col] + (anchor - team_avg)

    df["BPM"] = df.groupby("TEAM_ID", group_keys=False).apply(
        lambda g: team_shift(g, "RAW_BPM", t_adv["NET_RATING"]))
    df["OBPM"] = df.groupby("TEAM_ID", group_keys=False).apply(
        lambda g: team_shift(g, "RAW_OBPM", t_adv["OFF_ABOVE_AVG"]))
    df["DBPM"] = df["BPM"] - df["OBPM"]

    team_gp = t_tot.set_index("TEAM_ID")["GP"]
    team_min = t_tot.set_index("TEAM_ID")["MIN"]
    df["PCT_POSS"] = df["MIN_TOTAL"] / df["TEAM_ID"].map(team_min)
    df["GAMES_FACTOR"] = df["TEAM_ID"].map(team_gp) / 82.0
    df["VORP"] = (df["BPM"] - REPLACEMENT_LEVEL) * df["PCT_POSS"] * df["GAMES_FACTOR"]
    df["VORP_PER_GAME"] = df["VORP"] / df["GP"].replace(0, np.nan)

    out = df[[
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GP", "MIN_TOTAL",
        "POSITION_EST", "ROLE_EST", "PTS", "PTS_ADJ", "BPM", "OBPM", "DBPM", "VORP", "VORP_PER_GAME",
    ]].round(4).sort_values("VORP", ascending=False)

    out_dir = ROOT / "data" / "player_value"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bpm_vorp_{args.season}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} players to {out_path}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
