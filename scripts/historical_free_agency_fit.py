"""
Retroactive version of rank_teams_per_free_agent.py: runs the same
win%-weighted NEED_VECTOR fit-score analysis for every past offseason
cycle the repo has two consecutive seasons of real data for (7 cycles,
Summer 2019 through Summer 2025), instead of just the current 2026-27
free-agent class.

Methodology, per explicit instruction -- "use previous season's rosters
and then run based on the pending free agents that were on the team":

  - The free-agent pool for the season-S -> season-S_next transition is
    the REAL free-agent class (data/free_agents/<season>.txt, parsed by
    load_free_agent_class.py) -- every player who actually signed a new
    contract as a free agent that summer, whether they re-signed with
    their own team or moved. This is a real transactional category, not
    an inference.
  - PRE_FA(team) = that team's actual season-S roster MINUS anyone in the
    free-agent pool (matched by name, accent/punctuation-normalized).
    Everyone else -- including free agents who re-signed with their own
    team -- is removed from PRE_FA and re-added via the pool below.
  - Real destination = wherever that player's name shows up on a
    season-S_next roster (data/player_vectors_json). Free agents with no
    season-S_next appearance (retired, left the league, or a name that
    didn't match) are dropped -- no destination to validate against.
  - Skill vectors/minutes for both PRE_FA and pool players are season-S's
    (their profile entering that offseason), matching the 2026-27 page's
    "most recent known" convention.

This supersedes an earlier version of this script that inferred "movers"
by diffing two seasons' roster snapshots (team changed between S and
S_next). That approach had two real gaps this real free-agent list fixes
outright: it couldn't see same-team re-signings (identical to a player
who was simply never a free agent), and it couldn't distinguish a genuine
offseason move from a player traded MID-season during S_next itself (a
separate script, find_midseason_trades.py, pulled real transaction dates
to patch that -- no longer needed here, since a traded player keeps their
existing contract and never appears in a free-agent signing list).

One remaining real limitation, still shared with the 2026-27 page: a
free agent who left the league entirely, or a name that fails to match
between the free-agent list and the roster snapshots, silently drops out
of the pool -- there's no destination team to score them against either
way. Match rate is logged per cycle below.

Also, league-wide free-agent classes (~130-175 players/year, per the
parsed .txt files) are still far larger than the 2026 class (62), so a
real team's number of departures is often too large for the 62.9M-
combination-scale exhaustive search used on the 2026-27 page to stay
tractable. This script only does the O(pool x 30 teams) single-player
marginal fit (identical math to rank_teams_per_free_agent.py) -- no
combinatorial best/worst-k-sized-combo search for historical cycles.
"Team summary" here is each team's best/worst SINGLE free agent, not a
real-k-sized combo.

Output: data/free_agency/historical_fit_data.json, one entry per cycle
with the same shape as the 2026-27 artifact's data (weights excluded --
identical across cycles, already in the base artifact data).

Usage:
    python scripts/historical_free_agency_fit.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as static_teams

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import NEED_CATEGORIES
from exhaustive_free_agent_combos_need import fit_need_pls, load_win_contribution_weights, DENOM_FLOOR
from load_free_agent_class import load_free_agent_names, normalize_name

ROOT = Path(__file__).resolve().parent.parent
TEAM_NAME = {t["abbreviation"]: t["full_name"] for t in static_teams.get_teams()}
TEAM_TOTAL_MIN_TARGET = 240 * 82


def load_season_roster(season: str) -> dict:
    """PLAYER_ID -> {name, team, vector (12 NEED_CATEGORIES), minutes}."""
    vectors = json.loads((ROOT / "data" / "player_vectors_json" / f"player_vectors_{season}.json").read_text(encoding="utf-8"))
    stats = pd.read_csv(ROOT / "data" / "player_stats" / f"player_stats_{season}.csv")
    min_by_id = stats.set_index("PLAYER_ID")["MIN"].to_dict()

    roster = {}
    for pid_str, v in vectors.items():
        pid = int(pid_str)
        minutes = min_by_id.get(pid)
        if not minutes or minutes <= 0:
            continue
        roster[pid] = {
            "name": v["PLAYER_NAME"], "team": v["TEAM"],
            "vector": np.array([v["VECTOR"][c] for c in NEED_CATEGORIES]),
            "minutes": float(minutes),
        }
    return roster


def team_moments(roster: dict, players: list) -> tuple:
    S1 = np.zeros(12)
    S2 = np.zeros(12)
    total_min = 0.0
    for pid in players:
        p = roster[pid]
        S1 += p["minutes"] * p["vector"]
        S2 += p["minutes"] * p["vector"] ** 2
        total_min += p["minutes"]
    return S1, S2, total_min


def feat_from_moments(S1, S2, total_min):
    mean = S1 / total_min
    var = S2 / total_min - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))
    feat = np.empty(25)
    feat[0] = TEAM_TOTAL_MIN_TARGET
    feat[1::2] = mean
    feat[2::2] = std
    return feat


def run_cycle(season: str, season_next: str, W, intercept, weights) -> dict:
    roster_s = load_season_roster(season)
    roster_next = load_season_roster(season_next)
    team_of_next = {pid: r["team"] for pid, r in roster_next.items()}

    fa_names_raw = load_free_agent_names(season)
    fa_keys = {normalize_name(n) for n in fa_names_raw}
    by_key_s = {}
    for pid, p in roster_s.items():
        by_key_s.setdefault(normalize_name(p["name"]), []).append(pid)

    # every roster_s player matching the real free-agent list, regardless of whether they later
    # matched a destination -- ALL of these must come OUT of PRE_FA (they were free agents, full
    # stop); "movers" below is the scoreable SUBSET that also has a real destination to evaluate
    all_fa_pids = {pid for key in fa_keys for pid in by_key_s.get(key, [])}
    movers = [pid for pid in all_fa_pids if pid in team_of_next]
    n_matched, n_unmatched = len(movers), len(all_fa_pids) - len(movers)

    by_team_pre = {}
    for pid, p in roster_s.items():
        if pid in all_fa_pids:
            continue
        by_team_pre.setdefault(p["team"], []).append(pid)

    teams = sorted(t for t in by_team_pre if len(by_team_pre[t]) >= 2)
    print(f"  {season} -> {season_next}: {len(fa_keys)} real free agents, {n_matched} matched to a "
          f"season-S roster spot + real destination ({n_unmatched} unmatched/no destination), "
          f"{len(teams)} teams with a stable PRE_FA core")

    # need_before per team, and cache each team's PRE_FA moments (S1, S2, total_min) once
    need_before = {}
    team_pre_moments = {}
    for team in teams:
        S1, S2, total_min = team_moments(roster_s, by_team_pre[team])
        team_pre_moments[team] = (S1, S2, total_min)
        feat = feat_from_moments(S1, S2, total_min)
        need_before[team] = np.clip(feat @ W + intercept, 0.01, 0.99)

    mover_ids = movers
    mover_names = [roster_s[pid]["name"] for pid in mover_ids]
    mover_real_dest = [team_of_next[pid] for pid in mover_ids]
    mover_origin = [roster_s[pid]["team"] for pid in mover_ids]

    rows = []
    for pid, name, dest, origin in zip(mover_ids, mover_names, mover_real_dest, mover_origin):
        p = roster_s[pid]
        for team in teams:
            S1, S2, total_min = team_pre_moments[team]
            total_min2 = total_min + p["minutes"]
            S1_2 = S1 + p["minutes"] * p["vector"]
            S2_2 = S2 + p["minutes"] * p["vector"] ** 2
            feat = feat_from_moments(S1_2, S2_2, total_min2)
            need_after = np.clip(feat @ W + intercept, 0.01, 0.99)
            denom = np.maximum(1 - need_before[team], DENOM_FLOOR)
            pct = float(((need_before[team] - need_after) / denom * 100) @ weights)
            rows.append({"PLAYER_NAME": name, "PLAYER_ID": pid, "ORIGIN": origin, "REAL_DEST": dest,
                         "TEAM": team, "PCT": round(pct, 3)})

    df = pd.DataFrame(rows)
    df["TEAM_Z"] = df.groupby("TEAM")["PCT"].transform(lambda s: (s - s.mean()) / s.std()).round(2)
    df["PCT"] = df["PCT"].round(1)

    # Compact encoding: two index arrays (deduped player/team names) + one flat
    # [player_idx, team_idx, pct, team_z, is_real] row list, instead of two
    # redundant nested dicts -- cuts embedded JSON size roughly in half across
    # 7 cycles x ~180 movers x 30 teams.
    player_names = sorted(df["PLAYER_NAME"].unique())
    player_idx = {n: i for i, n in enumerate(player_names)}
    team_codes = sorted(df["TEAM"].unique())
    team_idx = {t: i for i, t in enumerate(team_codes)}
    dest_by_player = df.drop_duplicates("PLAYER_NAME").set_index("PLAYER_NAME")["REAL_DEST"].to_dict()

    flat = [
        [player_idx[r.PLAYER_NAME], team_idx[r.TEAM], r.PCT, r.TEAM_Z, int(r.REAL_DEST == r.TEAM)]
        for r in df.itertuples()
    ]

    team_summary_rows = []
    for team, grp in df.groupby("TEAM"):
        sub = grp.sort_values("TEAM_Z", ascending=False)
        best, worst = sub.iloc[0], sub.iloc[-1]
        team_summary_rows.append({
            "TEAM": team, "team_name": TEAM_NAME.get(team, team),
            "BEST_COMBO": best.PLAYER_NAME, "BEST_WEIGHTED_PCT_INCREASE": round(float(best.PCT), 1),
            "WORST_COMBO": worst.PLAYER_NAME, "WORST_WEIGHTED_PCT_INCREASE": round(float(worst.PCT), 1),
            "N_COMBOS": len(mover_ids),
        })

    return {
        "season": season, "season_next": season_next, "n_movers": len(mover_ids), "n_teams": len(teams),
        "team_summary": team_summary_rows,
        "player_names": player_names,
        "team_codes": team_codes,
        "team_names": {t: TEAM_NAME.get(t, t) for t in team_codes},
        "destinations": [dest_by_player.get(n) for n in player_names],
        "rows": flat,
    }


def main():
    print("Fitting roster-vectors -> NEED_VECTOR (PLS, linearized, same model as the 2026-27 page)...")
    W, intercept, feat_cols = fit_need_pls()
    weights = load_win_contribution_weights()

    cycles = list(zip(STANDARD_SEASONS[:-1], STANDARD_SEASONS[1:]))
    out = {}
    for season, season_next in cycles:
        result = run_cycle(season, season_next, W, intercept, weights)
        out[season_next] = result

    out_path = ROOT / "data" / "free_agency" / "historical_fit_data.json"
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
