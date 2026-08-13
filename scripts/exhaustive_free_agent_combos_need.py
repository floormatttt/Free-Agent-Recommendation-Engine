"""
Architecture change from exhaustive_free_agent_combos.py: instead of
scoring free-agent combinations by predicted win total (roster-vectors ->
OFF/DEF rating -> win%), this scores them by how much they lower the
team's 12-category NEED_VECTOR (0 = strength/no need, 1 = worst/as-if-
nonexistent -- the same scale used on the artifact's first page).

The improvement metric is a percentage change in "strength" (1 - need),
not a raw point difference, per category:

    pct_increase = (need_before - need_after) / (1 - need_before) * 100

e.g. need 0.75 -> 0.50 is strength 0.25 -> 0.50, a 100% increase -- going
from 25% filled to 50% filled is a doubling, not a 25-point improvement.

A team's composite score for a combination is a WEIGHTED average of this
percentage across the 12 categories, not a plain mean -- weighted by how
much each category actually predicts winning. Weights are the absolute
value of predict_win_pct_from_need.py's Ridge coefficients (real,
team-stats-derived NEED_VECTOR -> win%, averaged across its 8 LOSO folds,
stored in data/win_pct_model/win_pct_model_results.json), normalized to
sum to 1. PERIMETER_DEFENSE dominates (coefficient -0.355, more than
double the next-largest), so lowering that need matters several times
more to the composite score than, say, PUT_BACKS (-0.008). This is a
genuine architecture change from the unweighted version: a player who
only helps low-weight categories can score worse here even if they
"lowered more needs" by a flat count.

The percentage formula divides by (1 - need_before), which blows up for
any team-category with need_before very close to 1 (Washington's
PERIMETER_DEFENSE need was 0.986 pre-weighting, and produced a +358%
single-category swing from one modest real improvement -- see the
project's free-agency writeup). The denominator is floored at
DENOM_FLOOR to cap that blowup at a 1/DENOM_FLOOR-x multiplier instead of
being unbounded; this matters more now that PERIMETER_DEFENSE carries the
largest single weight, since a near-1 baseline there would otherwise
dominate the weighted composite even harder than it dominated the
unweighted one.

Model: roster-vectors -> NEED_VECTOR reconstruction, PLS only (not the
PLS+XGBoost ensemble predict_2026_27_team_profile.py uses for production)
-- fit on all 8 historical seasons. PLS-only was a deliberate choice: it
is an affine map (Y = X @ W + b once fit), so it can be "linearized" by
evaluating it at the origin and at each unit basis vector, giving an
exact (25, 12) coefficient matrix + (12,) intercept. That lets the same
closed-form, fully-vectorized combinatorial trick from
exhaustive_free_agent_combos.py apply here too -- XGBoost's tree splits
have no such linear form, so adding it back would make the 62.9M-
combination searches (LAL, PHI) computationally infeasible again.

Output: data/free_agency/exhaustive_fa_combos_need_summary.csv (best/worst combo per
team by composite % need-reduction), data/free_agency/exhaustive_fa_combos_need_player_impact.csv
(each free agent's average composite score across all combos containing
them, per team).

Usage:
    python scripts/exhaustive_free_agent_combos_need.py [--max-k N]
"""
import argparse
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from predict_team_need_from_roster import (
    build_dataset as build_roster_dataset, feature_columns, NEED_CATEGORIES,
    _select_pls_components, PLS_MAX_COMPONENTS,
)
from predict_2026_27_team_profile import TEAM_TOTAL_MIN_TARGET
from predict_2026_27_minutes import build_latest_vectors, build_latest_minutes, ROSTER_PATH

ROOT = Path(__file__).resolve().parent.parent
FA_SIGNED_PATTERN = re.compile(r"^Signed on 07/\d{2}/26$")
BATCH_SIZE = 200_000
NEED_CLIP = (0.01, 0.99)  # keeps 1 - need_before off zero and predictions off the unbounded PLS extrapolation tails
DENOM_FLOOR = 0.15  # floors (1 - need_before) so a near-1 baseline need can't blow up its % swing unboundedly


def load_win_contribution_weights() -> np.ndarray:
    """Per-category weights for the composite need-reduction score: |Ridge
    coefficient| from the real NEED_VECTOR -> win% model (predict_win_pct_
    from_need.py, averaged across its 8 LOSO folds), normalized to sum to
    1 -- categories that actually move win% more count for more here."""
    results = json.loads((ROOT / "data" / "win_pct_model" / "win_pct_model_results.json").read_text())
    coef = np.array([abs(results["ridge_coef_mean"][c]) for c in NEED_CATEGORIES])
    return coef / coef.sum()


def fit_need_pls():
    """Full-data PLS fit (25 roster features -> 12 NEED_CATEGORIES), then
    linearized via unit-vector evaluation: since PLSRegression.predict is
    exactly affine, predict(e_i) - predict(0) recovers column i of the
    equivalent (25, 12) weight matrix, and predict(0) is the intercept."""
    roster_df = build_roster_dataset()
    feat_cols = feature_columns(roster_df)
    target_cols = [f"TARGET_{c}" for c in NEED_CATEGORIES]
    X = roster_df[feat_cols].to_numpy()
    Y = roster_df[target_cols].to_numpy()

    n_components = _select_pls_components(X, Y, PLS_MAX_COMPONENTS)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)

    n_feat = len(feat_cols)
    zero = np.zeros((1, n_feat))
    intercept = pls.predict(zero).ravel()               # (12,)
    basis_pred = pls.predict(np.eye(n_feat))              # (25, 12)
    W = basis_pred - intercept                            # (25, 12)

    # sanity check: the linearization must be exact, not an approximation
    sample = X[:5]
    exact = pls.predict(sample)
    linear = sample @ W + intercept
    max_err = np.abs(exact - linear).max()
    assert max_err < 1e-6, f"PLS linearization mismatch: {max_err}"

    print(f"  PLS n_components={n_components}, linearization max error={max_err:.2e}")
    return W, intercept, feat_cols


def team_pre_fa_moments(rosters: pd.DataFrame, team: str, latest_vectors: dict, latest_minutes: dict):
    team_rows = rosters[rosters["TEAM"] == team]
    S1 = np.zeros(12)
    S2 = np.zeros(12)
    total_min = 0.0
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
    return S1, S2, total_min


def fa_pool_contribution_matrices(fa_pool: pd.DataFrame, latest_vectors: dict, latest_minutes: dict):
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
    vecs = np.array(vecs)
    mins = np.array(mins)
    C1 = mins[:, None] * vecs
    C2 = mins[:, None] * vecs ** 2
    return C1, C2, mins, names, teams


def team_feat_to_need(feat: np.ndarray, W: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.clip(feat @ W + intercept, *NEED_CLIP)


def score_batch(idx: np.ndarray, pre_S1, pre_S2, pre_min, need_before: np.ndarray,
                 C1, C2, Cmin, W, intercept, weights: np.ndarray) -> np.ndarray:
    """idx: (B, k). Returns (B,) composite % strength-increase scores,
    weighted per category by `weights` (see load_win_contribution_weights)."""
    combo_min = Cmin[idx].sum(axis=1)
    combo_S1 = C1[idx].sum(axis=1)
    combo_S2 = C2[idx].sum(axis=1)
    total_min = pre_min + combo_min
    mean = (pre_S1 + combo_S1) / total_min[:, None]
    var = (pre_S2 + combo_S2) / total_min[:, None] - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))

    B = idx.shape[0]
    feat = np.empty((B, 1 + 24))
    feat[:, 0] = TEAM_TOTAL_MIN_TARGET
    feat[:, 1::2] = mean
    feat[:, 2::2] = std

    need_after = team_feat_to_need(feat, W, intercept)            # (B, 12)
    denom = np.maximum(1 - need_before[None, :], DENOM_FLOOR)
    pct_increase = (need_before[None, :] - need_after) / denom * 100
    return pct_increase @ weights


def run_team(team: str, k: int, pre_S1, pre_S2, pre_min, need_before, C1, C2, Cmin, names, W, intercept,
             weights: np.ndarray) -> dict:
    P = len(names)
    total_combos = 1
    for i in range(k):
        total_combos = total_combos * (P - i) // (i + 1)
    print(f"  {team}: k={k}, {total_combos:,} combinations from a {P}-player pool")

    best_score, best_combo = -np.inf, None
    worst_score, worst_combo = np.inf, None
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
        scores = score_batch(idx, pre_S1, pre_S2, pre_min, need_before, C1, C2, Cmin, W, intercept, weights)

        b_best = scores.argmax()
        if scores[b_best] > best_score:
            best_score, best_combo = float(scores[b_best]), idx[b_best].copy()
        b_worst = scores.argmin()
        if scores[b_worst] < worst_score:
            worst_score, worst_combo = float(scores[b_worst]), idx[b_worst].copy()

        flat_idx = idx.ravel()
        flat_scores = np.repeat(scores, k)
        player_sum += np.bincount(flat_idx, weights=flat_scores, minlength=P)
        player_count += np.bincount(flat_idx, minlength=P)

        n_done += len(batch)

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s ({n_done:,} combos, {n_done/max(elapsed,1e-9):,.0f}/s)")

    avg_impact = np.divide(player_sum, player_count, out=np.full(P, np.nan), where=player_count > 0)
    return {
        "team": team, "k": k, "n_combos": total_combos,
        "best_combo": [names[i] for i in best_combo], "best_score": round(best_score, 2),
        "worst_combo": [names[i] for i in worst_combo], "worst_score": round(worst_score, 2),
        "player_avg_score": {names[i]: round(float(avg_impact[i]), 3) for i in range(P) if player_count[i] > 0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-k", type=int, default=None)
    args = ap.parse_args()

    rosters = pd.read_csv(ROSTER_PATH, encoding="utf-8")
    is_fa = rosters["HOW_ACQUIRED"].apply(lambda s: bool(FA_SIGNED_PATTERN.match(str(s))))
    fa_pool_df = rosters[is_fa].copy()
    pre_fa_rosters = rosters[~is_fa].copy()
    fa_counts = fa_pool_df["TEAM"].value_counts()

    print("Fitting roster-vectors -> NEED_VECTOR (PLS, linearized)...")
    W, intercept, feat_cols = fit_need_pls()

    weights = load_win_contribution_weights()
    print("Category weights (|win% Ridge coef|, normalized):")
    for c, w in zip(NEED_CATEGORIES, weights):
        print(f"    {c:20s} {w:.4f}")

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

        pre_S1, pre_S2, pre_min = team_pre_fa_moments(pre_fa_rosters, team, latest_vectors, latest_minutes)
        pre_feat = np.empty(25)
        pre_feat[0] = TEAM_TOTAL_MIN_TARGET
        pre_mean = pre_S1 / pre_min
        pre_var = pre_S2 / pre_min - pre_mean ** 2
        pre_std = np.sqrt(np.clip(pre_var, 0, None))
        pre_feat[1::2] = pre_mean
        pre_feat[2::2] = pre_std
        need_before = team_feat_to_need(pre_feat[None, :], W, intercept)[0]

        result = run_team(team, k, pre_S1, pre_S2, pre_min, need_before, C1, C2, Cmin, names, W, intercept, weights)

        summary_rows.append({
            "TEAM": team, "K": k,
            "NEED_BEFORE_WEIGHTED_COMPOSITE": round(float(need_before @ weights), 3),
            "BEST_COMBO": " + ".join(result["best_combo"]), "BEST_WEIGHTED_PCT_INCREASE": result["best_score"],
            "WORST_COMBO": " + ".join(result["worst_combo"]), "WORST_WEIGHTED_PCT_INCREASE": result["worst_score"],
            "N_COMBOS": result["n_combos"],
        })
        for player, avg_score in sorted(result["player_avg_score"].items(), key=lambda kv: -kv[1]):
            impact_rows.append({
                "TEAM": team, "K": k, "PLAYER_NAME": player,
                "AVG_WEIGHTED_PCT_INCREASE_IN_COMBO": avg_score,
                "REAL_2026_DESTINATION": fa_real_teams[names.index(player)] if player in names else None,
            })

        pd.DataFrame(summary_rows).to_csv(ROOT / "data" / "free_agency" / "exhaustive_fa_combos_need_summary.csv", index=False)
        pd.DataFrame(impact_rows).to_csv(ROOT / "data" / "free_agency" / "exhaustive_fa_combos_need_player_impact.csv", index=False)

    print("\nDone. Wrote data/free_agency/exhaustive_fa_combos_need_summary.csv, "
          "data/free_agency/exhaustive_fa_combos_need_player_impact.csv")


if __name__ == "__main__":
    main()
