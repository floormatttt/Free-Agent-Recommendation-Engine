"""
Isolates the win impact of 2026 free agency using the rating-detour
pipeline (predict_rating_detour_win_pct.py, LOSO RMSE 0.0777 -- the best
model found so far).

"Free agent" here = a player whose data/rosters/rosters_2026-27.csv
HOW_ACQUIRED is "Signed on 07/DD/26" (the July 2026 free-agency signing
period; the roster data separately shows a clean gap with zero June 2026
signings, so July-2026 is an unambiguous cutoff). Sign-and-trade and
trade-deadline moves are left alone -- those are a different transaction
type with an existing contract attached, not a resettable "add this FA to
a different team" input.

Three things:
  1. PRE_FA rosters: each team's real 2026-27 roster with its July-2026
     free-agent signings removed. Win prediction on this vs. the real
     full roster isolates free agency's total win impact per team.
  2. Free agent pool: all 62 identified signings, with their real
     destination team.
  3. Swap scenarios: pick team pairs that added the same NUMBER of free
     agents this offseason, and swap which team actually got them --
     each team's roster after a swap still has exactly as many players as
     its real current roster (PRE_FA count + the same number of added
     free agents, just a different set of players), matching the
     constraint that a counterfactual roster must match today's real
     roster size.

Output: data/free_agency/free_agency_impact.csv (per-team, with/without FA),
        data/free_agency/free_agency_swap_scenarios.csv

Usage:
    python scripts/simulate_free_agency_impact.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from build_team_vectors import STANDARD_SEASONS
from predict_team_need_from_roster import (
    build_dataset as build_roster_dataset, feature_columns, NEED_CATEGORIES,
)
from predict_win_pct_from_need import RIDGE_ALPHAS
from evaluate_roster_need_win_pct import attach_win_pct, loso_predict
from analyze_player_vectors_vs_ratings import attach_ratings
from predict_2026_27_team_profile import GAMES_PER_SEASON, W_PCT_CLIP, TEAM_TOTAL_MIN_TARGET
from predict_2026_27_minutes import build_latest_vectors, build_latest_minutes, ROSTER_PATH

ROOT = Path(__file__).resolve().parent.parent
FA_SIGNED_PATTERN = re.compile(r"^Signed on 07/\d{2}/26$")


def build_team_rows(rosters: pd.DataFrame, latest_vectors: dict, latest_minutes: dict) -> pd.DataFrame:
    """Same aggregation as predict_2026_27_team_profile.build_2026_27_team_rows,
    but takes an arbitrary roster dataframe instead of re-reading the real
    2026-27 roster file, so it can be applied to PRE_FA or swapped rosters."""
    by_team: dict[str, list[tuple[float, dict]]] = {}
    for _, r in rosters.iterrows():
        pid = int(r["PLAYER_ID"])
        if pid not in latest_vectors:
            continue
        vector, _ = latest_vectors[pid]
        minutes, _ = latest_minutes.get(pid, (0.0, None))
        if not minutes or minutes <= 0:
            continue
        by_team.setdefault(r["TEAM"], []).append((minutes, vector))

    rows = []
    for team, roster in by_team.items():
        if len(roster) < 2:
            continue
        mins = np.array([m for m, _ in roster], dtype=float)
        weights = mins / mins.sum()
        row = {"TEAM": team, "TOTAL_MIN": TEAM_TOTAL_MIN_TARGET, "ROSTER_N": len(roster)}
        for cat in NEED_CATEGORIES:
            vals = np.array([v[cat] for _, v in roster], dtype=float)
            w_mean = np.sum(weights * vals)
            w_std = np.sqrt(np.sum(weights * (vals - w_mean) ** 2))
            row[f"{cat}_WMEAN"] = w_mean
            row[f"{cat}_WSTD"] = w_std
        rows.append(row)
    return pd.DataFrame(rows)


def fit_production_pipeline():
    """Same fit as predict_rating_detour_win_pct.py's production path:
    Ridge(roster-vectors -> OFF/DEF rating) then Ridge(25 roster features +
    LOSO-predicted OFF/DEF/NET rating -> win%), all fit on the 8 real
    historical seasons."""
    roster_df = build_roster_dataset()
    feat_cols = feature_columns(roster_df)
    df = attach_ratings(attach_win_pct(roster_df))

    d_off = df.drop(columns=["W_PCT"]).rename(columns={"OFF_RATING": "W_PCT"})
    d_def = df.drop(columns=["W_PCT"]).rename(columns={"DEF_RATING": "W_PCT"})
    off_loso, _ = loso_predict(d_off, feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    def_loso, _ = loso_predict(d_def, feat_cols, lambda: RidgeCV(alphas=RIDGE_ALPHAS))
    off_p = off_loso.rename(columns={"PRED": "PRED_OFF_RATING"})[["SEASON", "TEAM", "PRED_OFF_RATING"]]
    def_p = def_loso.rename(columns={"PRED": "PRED_DEF_RATING"})[["SEASON", "TEAM", "PRED_DEF_RATING"]]
    aug = df.merge(off_p, on=["SEASON", "TEAM"]).merge(def_p, on=["SEASON", "TEAM"])
    aug["PRED_NET_RATING"] = aug["PRED_OFF_RATING"] - aug["PRED_DEF_RATING"]
    aug_feat_cols = feat_cols + ["PRED_OFF_RATING", "PRED_DEF_RATING", "PRED_NET_RATING"]

    off_full = RidgeCV(alphas=RIDGE_ALPHAS).fit(df[feat_cols], df["OFF_RATING"])
    def_full = RidgeCV(alphas=RIDGE_ALPHAS).fit(df[feat_cols], df["DEF_RATING"])
    win_full = RidgeCV(alphas=RIDGE_ALPHAS).fit(aug[aug_feat_cols], aug["W_PCT"])
    return off_full, def_full, win_full, feat_cols, aug_feat_cols


def predict_wins(rosters: pd.DataFrame, latest_vectors, latest_minutes, off_full, def_full, win_full,
                  feat_cols, aug_feat_cols) -> pd.DataFrame:
    team_rows = build_team_rows(rosters, latest_vectors, latest_minutes)
    if team_rows.empty:
        return team_rows
    team_rows["PRED_OFF_RATING"] = off_full.predict(team_rows[feat_cols])
    team_rows["PRED_DEF_RATING"] = def_full.predict(team_rows[feat_cols])
    team_rows["PRED_NET_RATING"] = team_rows["PRED_OFF_RATING"] - team_rows["PRED_DEF_RATING"]
    pred = np.clip(win_full.predict(team_rows[aug_feat_cols]), *W_PCT_CLIP)
    team_rows["PRED_W_PCT"] = pred
    team_rows["PROJECTED_WINS"] = (pred * GAMES_PER_SEASON).round(1)
    return team_rows


def main():
    rosters = pd.read_csv(ROSTER_PATH, encoding="utf-8")
    is_fa = rosters["HOW_ACQUIRED"].apply(lambda s: bool(FA_SIGNED_PATTERN.match(str(s))))
    fa_pool = rosters[is_fa].copy()
    pre_fa_rosters = rosters[~is_fa].copy()
    print(f"Identified {len(fa_pool)} July-2026 free-agent signings across "
          f"{fa_pool['TEAM'].nunique()} teams (full roster {len(rosters)} players -> "
          f"pre-FA {len(pre_fa_rosters)} players)")

    print("Fitting production rating-detour pipeline on all 8 historical seasons...")
    off_full, def_full, win_full, feat_cols, aug_feat_cols = fit_production_pipeline()
    latest_vectors = build_latest_vectors()
    latest_minutes = build_latest_minutes()

    print("Predicting wins: real full roster vs. PRE_FA roster...")
    full_pred = predict_wins(rosters, latest_vectors, latest_minutes, off_full, def_full, win_full, feat_cols, aug_feat_cols)
    pre_fa_pred = predict_wins(pre_fa_rosters, latest_vectors, latest_minutes, off_full, def_full, win_full, feat_cols, aug_feat_cols)

    fa_counts = fa_pool["TEAM"].value_counts()
    impact = full_pred[["TEAM", "ROSTER_N", "PROJECTED_WINS"]].rename(
        columns={"ROSTER_N": "ROSTER_N_FULL", "PROJECTED_WINS": "WINS_FULL"})
    impact = impact.merge(
        pre_fa_pred[["TEAM", "ROSTER_N", "PROJECTED_WINS"]].rename(
            columns={"ROSTER_N": "ROSTER_N_PRE_FA", "PROJECTED_WINS": "WINS_PRE_FA"}),
        on="TEAM", how="outer")
    impact["NUM_FA_SIGNED"] = impact["TEAM"].map(fa_counts).fillna(0).astype(int)
    impact["FA_WIN_IMPACT"] = (impact["WINS_FULL"] - impact["WINS_PRE_FA"]).round(1)
    impact = impact.sort_values("FA_WIN_IMPACT", ascending=False)
    impact.to_csv(ROOT / "data" / "free_agency" / "free_agency_impact.csv", index=False)

    print("\n=== Free agency win impact by team (WINS_FULL - WINS_PRE_FA) ===")
    print(impact.to_string(index=False))

    # ---- swap scenarios: teams that added the same NUMBER of free agents ----
    print("\n=== Swap scenarios: same FA count, different players ===")
    same_count_pairs = []
    for n in sorted(fa_counts.unique()):
        teams_with_n = sorted(fa_counts[fa_counts == n].index.tolist())
        for i in range(0, len(teams_with_n) - 1, 2):
            same_count_pairs.append((teams_with_n[i], teams_with_n[i + 1], n))

    scenario_rows = []
    for team_a, team_b, n in same_count_pairs:
        fa_a = fa_pool[fa_pool["TEAM"] == team_a]
        fa_b = fa_pool[fa_pool["TEAM"] == team_b]
        pre_a = pre_fa_rosters[pre_fa_rosters["TEAM"] == team_a]
        pre_b = pre_fa_rosters[pre_fa_rosters["TEAM"] == team_b]

        # swap: team_a's pre-FA roster + team_b's real free agents (relabeled), and vice versa
        swapped_a = pd.concat([pre_a, fa_b.assign(TEAM=team_a)], ignore_index=True)
        swapped_b = pd.concat([pre_b, fa_a.assign(TEAM=team_b)], ignore_index=True)
        assert len(swapped_a) == len(pre_a) + n and len(swapped_b) == len(pre_b) + n

        swapped_rosters = pd.concat([
            rosters[~rosters["TEAM"].isin([team_a, team_b])], swapped_a, swapped_b,
        ], ignore_index=True)
        swapped_pred = predict_wins(swapped_rosters, latest_vectors, latest_minutes,
                                     off_full, def_full, win_full, feat_cols, aug_feat_cols)
        swapped_by_team = swapped_pred.set_index("TEAM")["PROJECTED_WINS"]

        real_a = full_pred.set_index("TEAM").loc[team_a, "PROJECTED_WINS"]
        real_b = full_pred.set_index("TEAM").loc[team_b, "PROJECTED_WINS"]
        scenario_rows.append({
            "team": team_a, "swapped_with": team_b, "n_fa_swapped": n,
            "real_wins": real_a, "swapped_wins": swapped_by_team.get(team_a, float("nan")),
        })
        scenario_rows.append({
            "team": team_b, "swapped_with": team_a, "n_fa_swapped": n,
            "real_wins": real_b, "swapped_wins": swapped_by_team.get(team_b, float("nan")),
        })

    scenarios = pd.DataFrame(scenario_rows)
    scenarios["DELTA"] = (scenarios["swapped_wins"] - scenarios["real_wins"]).round(1)
    scenarios.to_csv(ROOT / "data" / "free_agency" / "free_agency_swap_scenarios.csv", index=False)
    print(scenarios.to_string(index=False))

    fa_pool[["TEAM", "PLAYER_NAME", "PLAYER_ID", "HOW_ACQUIRED"]].to_csv(
        ROOT / "data" / "free_agency" / "free_agency_pool_2026.csv", index=False)
    print(f"\nWrote data/free_agency/free_agency_impact.csv, data/free_agency/free_agency_swap_scenarios.csv, "
          f"data/free_agency/free_agency_pool_2026.csv")


if __name__ == "__main__":
    main()
