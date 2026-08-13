"""
Backtests the whole roster -> NEED_VECTOR -> win% pipeline against real
outcomes, for every historical cycle historical_free_agency_fit.py covers
(Summer 2019 - Summer 2025): for each team, reconstruct the roster as it
was actually built heading into season_next -- PRE_FA core (players not
in that summer's real free-agent class, per load_free_agent_class.py)
plus the REAL free agents who actually landed on that team -- using each
player's season-S (pre-offseason) skill vector/minutes, same convention
as the rest of this pipeline -- then:

  1. Run that roster through the linearized PLS model to get a PROJECTED
     NEED_VECTOR, and compare it to the REAL, team-stats-derived
     NEED_VECTOR observed for that team in season_next
     (data/team_vectors_json/team_vectors_<season_next>.json) -- how
     accurate is the reconstruction itself?
  2. Feed the PROJECTED NEED_VECTOR through a fresh production RidgeCV
     (real NEED_VECTOR -> W_PCT, fit on all 8 seasons) to get a
     PROJECTED win%, and compare to the REAL win% the team posted in
     season_next (data/team_wins/team_wins_<season_next>.csv) -- does
     the projected roster construction actually predict how the team did?

This is a genuinely "before the season" backtest (unlike predict_win_pct_
from_need.py's LOSO validation, which uses each team's real, hindsight
NEED_VECTOR): every input here is knowable before season_next tips off.

Output: data/free_agency/historical_need_projection_eval.csv, one row per
team-season with projected/real NEED_VECTOR (12 categories each),
projected/real win%, and per-row need RMSE.

Usage:
    python scripts/evaluate_historical_need_projections.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import NEED_CATEGORIES
from predict_win_pct_from_need import RIDGE_ALPHAS, build_dataset as build_winpct_dataset
from exhaustive_free_agent_combos_need import fit_need_pls
from historical_free_agency_fit import load_season_roster, team_moments, feat_from_moments
from load_free_agent_class import load_free_agent_names, normalize_name

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("Fitting roster-vectors -> NEED_VECTOR (PLS, linearized)...")
    W, intercept, feat_cols = fit_need_pls()

    print("Fitting real NEED_VECTOR -> win% (RidgeCV, production fit on all 8 seasons)...")
    winpct_df = build_winpct_dataset()
    win_ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    win_ridge.fit(winpct_df[NEED_CATEGORIES], winpct_df["W_PCT"])

    cycles = list(zip(STANDARD_SEASONS[:-1], STANDARD_SEASONS[1:]))
    rows = []
    for season, season_next in cycles:
        roster_s = load_season_roster(season)
        roster_next = load_season_roster(season_next)
        team_of_next = {pid: r["team"] for pid, r in roster_next.items()}

        fa_keys = {normalize_name(n) for n in load_free_agent_names(season)}
        by_key_s = {}
        for pid, p in roster_s.items():
            by_key_s.setdefault(normalize_name(p["name"]), []).append(pid)
        # ALL real free agents come out of PRE_FA, even ones with no season_next appearance to
        # score against (movers is the scoreable subset with a real destination)
        all_fa_pids = {pid for key in fa_keys for pid in by_key_s.get(key, [])}
        movers = [pid for pid in all_fa_pids if pid in team_of_next]

        by_team_pre = {}
        for pid, p in roster_s.items():
            if pid in all_fa_pids:
                continue
            by_team_pre.setdefault(p["team"], []).append(pid)
        by_team_arrivals = {}
        for pid in movers:
            dest = team_of_next[pid]
            by_team_arrivals.setdefault(dest, []).append(pid)

        real_need = json.loads((ROOT / "data" / "team_vectors_json" / f"team_vectors_{season_next}.json").read_text(encoding="utf-8"))
        real_wins = pd.read_csv(ROOT / "data" / "team_wins" / f"team_wins_{season_next}.csv").set_index("TEAM")

        teams = sorted(t for t in by_team_pre if len(by_team_pre[t]) >= 2)
        for team in teams:
            if team not in real_need or team not in real_wins.index:
                continue
            roster_ids = by_team_pre[team] + by_team_arrivals.get(team, [])
            S1, S2, total_min = team_moments(roster_s, roster_ids)
            feat = feat_from_moments(S1, S2, total_min)
            projected_need = np.clip(feat @ W + intercept, 0.01, 0.99)
            projected_wpct = float(np.clip(win_ridge.predict(projected_need.reshape(1, -1))[0], 0.0, 1.0))

            real_need_vec = np.array([real_need[team]["NEED_VECTOR"][c] for c in NEED_CATEGORIES])
            real_wpct = float(real_wins.loc[team, "W_PCT"])

            need_rmse = float(np.sqrt(np.mean((projected_need - real_need_vec) ** 2)))
            row = {
                "SEASON": season, "SEASON_NEXT": season_next, "TEAM": team,
                "N_PRE_FA": len(by_team_pre[team]), "N_REAL_ARRIVALS": len(by_team_arrivals.get(team, [])),
                "PROJECTED_WPCT": round(projected_wpct, 4), "REAL_WPCT": round(real_wpct, 4),
                "NEED_RMSE": round(need_rmse, 4),
            }
            for i, c in enumerate(NEED_CATEGORIES):
                row[f"PROJ_{c}"] = round(float(projected_need[i]), 3)
                row[f"REAL_{c}"] = round(float(real_need_vec[i]), 3)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "free_agency" / "historical_need_projection_eval.csv", index=False)

    win_rmse = float(np.sqrt(np.mean((df["PROJECTED_WPCT"] - df["REAL_WPCT"]) ** 2)))
    win_corr = float(np.corrcoef(df["PROJECTED_WPCT"], df["REAL_WPCT"])[0, 1])
    print(f"\n{len(df)} team-seasons evaluated across {len(cycles)} cycles")
    print(f"Win%% RMSE (projected vs. real): {win_rmse:.4f}  ({win_rmse*82:.1f} wins)")
    print(f"Win%% correlation (projected vs. real): {win_corr:.3f}")
    print(f"NEED_VECTOR RMSE (projected vs. real, mean across categories): {df['NEED_RMSE'].mean():.4f}")

    print("\nPer-category NEED_VECTOR RMSE:")
    for c in NEED_CATEGORIES:
        cat_rmse = float(np.sqrt(np.mean((df[f"PROJ_{c}"] - df[f"REAL_{c}"]) ** 2)))
        print(f"  {c:20s} {cat_rmse:.4f}")

    print("\nPer-cycle win%% RMSE:")
    for season_next, grp in df.groupby("SEASON_NEXT"):
        rm = float(np.sqrt(np.mean((grp["PROJECTED_WPCT"] - grp["REAL_WPCT"]) ** 2)))
        print(f"  {season_next}: {rm:.4f}  ({rm*82:.1f} wins, n={len(grp)})")

    print(f"\nWrote data/free_agency/historical_need_projection_eval.csv")


if __name__ == "__main__":
    main()
