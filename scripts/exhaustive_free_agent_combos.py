"""
For every team, exhaustively enumerates every possible combination of
free agents (from the full 62-player 2026 free-agent pool identified in
simulate_free_agency_impact.py) of size k = that team's REAL number of
July-2026 signings, and scores each combination with the same production
rating-detour pipeline (predict_rating_detour_win_pct.py).

This is not "what if team X had signed exactly its real free agents" --
it's "of every size-k subset of this year's entire free-agent class,
which ones would have helped team X most, and which individual players
show up disproportionately in the best combinations."

The naive approach (build a roster dataframe and rerun the full pandas
pipeline per combination) is far too slow at combinatorial scale --
Boston alone needs a 4-of-62 search (635,376 combinations) and the Lakers/
76ers each need a 7-of-62 search (245,866,078 combinations). Instead this
reduces the whole pipeline to closed-form linear algebra:

  - A team's 25 roster-aggregate features are minutes-weighted mean/std of
    12 skill categories. Weighted mean and the weighted second moment
    (E[X^2], used to get variance via Var = E[X^2] - E[X]^2) are BOTH
    linear in "sum of minutes*vector" and "sum of minutes*vector^2" across
    roster members -- so a team's PRE_FA roster contributes one fixed
    (S1, S2, total_minutes) triple, and any k-subset of free agents
    contributes another (S1, S2, total_minutes) triple that is a plain sum
    over the k chosen rows of a precomputed (62, 12) contribution matrix.
  - Both Ridge models downstream (roster-vectors -> OFF/DEF rating, and
    25 features + 3 predicted ratings -> win%) are linear, so the entire
    pipeline collapses to a handful of dot products per combination.

This turns each combination's cost into O(1) vector math instead of a
full pandas/sklearn pass, and lets combinations be scored in large
vectorized numpy batches (default 200,000 per batch) rather than a Python
loop -- the only thing that's still combinatorial is enumerating the
index tuples themselves (itertools.combinations), which is the actual
bottleneck at k=7.

For each team, reports:
  - The single best-scoring free-agent combination (and its projected wins).
  - The single worst-scoring combination.
  - Every free agent's average projected wins across all combinations that
    included them (a same-coalition-size marginal-contribution proxy --
    not a true Shapley value, but the closest cheap approximation without
    also varying combination size) -- this is the "most impactful players
    for this team" ranking.

Output: data/free_agency/exhaustive_fa_combos_summary.csv (one row per team: best/worst
combo and wins), data/free_agency/exhaustive_fa_combos_player_impact.csv (one row per
team x free agent: average projected wins across all combos containing
that player, for that team).

Usage:
    python scripts/exhaustive_free_agent_combos.py [--max-k N]

    --max-k caps k at N (skips teams whose real FA count exceeds N) --
    useful for a fast partial run; omit for the full exhaustive sweep
    (the k=7 teams alone take several minutes each).
"""
import argparse
import itertools
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

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
BATCH_SIZE = 200_000


def fit_production_pipeline():
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


def team_pre_fa_moments(rosters: pd.DataFrame, team: str, latest_vectors: dict, latest_minutes: dict):
    """(S1, S2, total_min, n_players) for a team's PRE_FA roster -- sum of
    minutes*vector and minutes*vector^2 across the 12 NEED_CATEGORIES, plus
    total minutes and player count, all additive so free agents can be
    layered on with a plain vector sum."""
    team_rows = rosters[rosters["TEAM"] == team]
    S1 = np.zeros(12)
    S2 = np.zeros(12)
    total_min = 0.0
    n = 0
    for _, r in team_rows.iterrows():
        pid = int(r["PLAYER_ID"])
        if pid not in latest_vectors:
            continue
        vector, _ = latest_vectors[pid]
        minutes, _ = latest_minutes.get(pid, (0.0, None))
        if not minutes or minutes <= 0:
            continue
        vec = np.array([vector[c] for c in NEED_CATEGORIES])
        S1 += minutes * vec
        S2 += minutes * vec ** 2
        total_min += minutes
        n += 1
    return S1, S2, total_min, n


def fa_pool_contribution_matrices(fa_pool: pd.DataFrame, latest_vectors: dict, latest_minutes: dict):
    """(C1, C2, Cmin, names) for the free-agent pool -- per-player
    contribution rows so any k-subset's total is a sum over chosen rows."""
    mins, vecs, names, teams = [], [], [], []
    for _, r in fa_pool.iterrows():
        pid = int(r["PLAYER_ID"])
        if pid not in latest_vectors:
            continue
        vector, _ = latest_vectors[pid]
        minutes, _ = latest_minutes.get(pid, (0.0, None))
        if not minutes or minutes <= 0:
            continue
        vecs.append([vector[c] for c in NEED_CATEGORIES])
        mins.append(minutes)
        names.append(r["PLAYER_NAME"])
        teams.append(r["TEAM"])
    vecs = np.array(vecs)          # (P, 12)
    mins = np.array(mins)          # (P,)
    C1 = mins[:, None] * vecs      # (P, 12)
    C2 = mins[:, None] * vecs ** 2  # (P, 12)
    return C1, C2, mins, names, teams


def score_batch(idx: np.ndarray, pre_S1, pre_S2, pre_min, C1, C2, Cmin,
                 off_coef, off_int, def_coef, def_int, win_coef, win_int) -> np.ndarray:
    """idx: (B, k) int array of free-agent-pool indices. Returns (B,) projected wins."""
    combo_min = Cmin[idx].sum(axis=1)                      # (B,)
    combo_S1 = C1[idx].sum(axis=1)                          # (B, 12)
    combo_S2 = C2[idx].sum(axis=1)                          # (B, 12)
    total_min = pre_min + combo_min                          # (B,)
    mean = (pre_S1 + combo_S1) / total_min[:, None]          # (B, 12)
    var = (pre_S2 + combo_S2) / total_min[:, None] - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))

    B = idx.shape[0]
    feat = np.empty((B, 1 + 24))
    feat[:, 0] = TEAM_TOTAL_MIN_TARGET
    feat[:, 1::2] = mean
    feat[:, 2::2] = std

    off_pred = feat @ off_coef + off_int
    def_pred = feat @ def_coef + def_int
    net_pred = off_pred - def_pred

    win_feat = np.concatenate([feat, off_pred[:, None], def_pred[:, None], net_pred[:, None]], axis=1)
    win_pred = win_feat @ win_coef + win_int
    clipped = np.clip(win_pred, *W_PCT_CLIP)
    return clipped * GAMES_PER_SEASON


def run_team(team: str, k: int, pre_S1, pre_S2, pre_min, C1, C2, Cmin, names, coefs) -> dict:
    P = len(names)
    total_combos = 1
    for i in range(k):
        total_combos = total_combos * (P - i) // (i + 1)
    print(f"  {team}: k={k}, {total_combos:,} combinations from a {P}-player pool")

    best_wins, best_combo = -np.inf, None
    worst_wins, worst_combo = np.inf, None
    player_sum = np.zeros(P)
    player_count = np.zeros(P)

    t0 = time.time()
    combos_iter = itertools.combinations(range(P), k)
    n_done = 0
    while True:
        batch = list(itertools.islice(combos_iter, BATCH_SIZE))
        if not batch:
            break
        idx = np.array(batch, dtype=np.int64)
        wins = score_batch(idx, pre_S1, pre_S2, pre_min, C1, C2, Cmin, *coefs)

        b_best = wins.argmax()
        if wins[b_best] > best_wins:
            best_wins, best_combo = float(wins[b_best]), idx[b_best].copy()
        b_worst = wins.argmin()
        if wins[b_worst] < worst_wins:
            worst_wins, worst_combo = float(wins[b_worst]), idx[b_worst].copy()

        flat_idx = idx.ravel()
        flat_wins = np.repeat(wins, k)
        player_sum += np.bincount(flat_idx, weights=flat_wins, minlength=P)
        player_count += np.bincount(flat_idx, minlength=P)

        n_done += len(batch)

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s ({n_done:,} combos, {n_done/max(elapsed,1e-9):,.0f}/s)")

    avg_impact = np.divide(player_sum, player_count, out=np.full(P, np.nan), where=player_count > 0)
    return {
        "team": team, "k": k, "n_combos": total_combos,
        "best_combo": [names[i] for i in best_combo], "best_wins": round(best_wins, 2),
        "worst_combo": [names[i] for i in worst_combo], "worst_wins": round(worst_wins, 2),
        "player_avg_wins": {names[i]: round(float(avg_impact[i]), 3) for i in range(P) if player_count[i] > 0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-k", type=int, default=None, help="skip teams whose real FA count exceeds this")
    args = ap.parse_args()

    rosters = pd.read_csv(ROSTER_PATH, encoding="utf-8")
    is_fa = rosters["HOW_ACQUIRED"].apply(lambda s: bool(FA_SIGNED_PATTERN.match(str(s))))
    fa_pool_df = rosters[is_fa].copy()
    pre_fa_rosters = rosters[~is_fa].copy()
    fa_counts = fa_pool_df["TEAM"].value_counts()

    print("Fitting production rating-detour pipeline...")
    off_full, def_full, win_full, feat_cols, aug_feat_cols = fit_production_pipeline()
    coefs = (off_full.coef_, off_full.intercept_, def_full.coef_, def_full.intercept_,
             win_full.coef_, win_full.intercept_)

    latest_vectors = build_latest_vectors()
    latest_minutes = build_latest_minutes()

    print("Building free-agent pool contribution matrices...")
    C1, C2, Cmin, names, fa_real_teams = fa_pool_contribution_matrices(fa_pool_df, latest_vectors, latest_minutes)
    print(f"  {len(names)} free agents with usable data")

    summary_rows = []
    impact_rows = []
    teams = sorted(rosters["TEAM"].unique())
    for team in teams:
        k = int(fa_counts.get(team, 0))
        if k == 0:
            continue
        if args.max_k is not None and k > args.max_k:
            print(f"  {team}: k={k} exceeds --max-k={args.max_k}, skipped")
            continue

        pre_S1, pre_S2, pre_min, pre_n = team_pre_fa_moments(pre_fa_rosters, team, latest_vectors, latest_minutes)
        result = run_team(team, k, pre_S1, pre_S2, pre_min, C1, C2, Cmin, names, coefs)

        summary_rows.append({
            "TEAM": team, "K": k,
            "BEST_COMBO": " + ".join(result["best_combo"]), "BEST_WINS": result["best_wins"],
            "WORST_COMBO": " + ".join(result["worst_combo"]), "WORST_WINS": result["worst_wins"],
            "N_COMBOS": result["n_combos"],
        })
        for player, avg_wins in sorted(result["player_avg_wins"].items(), key=lambda kv: -kv[1]):
            impact_rows.append({
                "TEAM": team, "K": k, "PLAYER_NAME": player,
                "AVG_PROJECTED_WINS_IN_COMBO": avg_wins,
                "REAL_2026_DESTINATION": fa_real_teams[names.index(player)] if player in names else None,
            })

        # flush incrementally so partial progress survives an interruption
        pd.DataFrame(summary_rows).to_csv(ROOT / "data" / "free_agency" / "exhaustive_fa_combos_summary.csv", index=False)
        pd.DataFrame(impact_rows).to_csv(ROOT / "data" / "free_agency" / "exhaustive_fa_combos_player_impact.csv", index=False)

    print("\nDone. Wrote data/free_agency/exhaustive_fa_combos_summary.csv, data/free_agency/exhaustive_fa_combos_player_impact.csv")


if __name__ == "__main__":
    main()
