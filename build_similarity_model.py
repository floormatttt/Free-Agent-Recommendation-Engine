"""
Builds the Similar Player Model described in similaritymodel.md, across
every player-season vector we have on file (one player who appears in
multiple seasons contributes one row per season).

For each player-season, finds the 10 nearest neighbors -- drawn from the
*entire* cross-season pool, not just that player's own season -- by an
anisotropic Mahalanobis-style distance over the 12-category player vector,
then reports those neighbors' contracts (in whichever season the neighbor
itself belongs to) and a distance-weighted average contract. A neighbor
can never be the same underlying player in a different season -- that
would just be comparing someone to themselves.

Contracts are expressed as SALARY_PCT_OF_CAP (from
data/salaries/salaries_<season>.csv), not raw dollars -- since neighbors
can come from any season across a decade of cap growth, a raw dollar
salary from 2018-19 isn't comparable to one from 2025-26, but the same
salary's share of that season's cap is.

Output is one JSON per season (data/similarity/similarity_<season>.json),
containing that season's players as the query set, but each player's top
10 comps may be drawn from any season in the pool.

The "parallel axis" is the first principal component (PC1) of the
standardized cross-season player-vector population -- empirically, the
dominant shared direction of variation, which in practice reads as general
"quality" (players who are good tend to be good across many categories at
once, correlating together along this axis). The "perpendicular axes" are
everything orthogonal to it, capturing archetype/style differences
independent of that shared quality direction.

Given a parallel:perpendicular axis-length ratio R, the distance between
two player-seasons is:

    delta = z_i - z_j                          (in standardized space)
    delta_parallel = delta . pc1                (signed scalar)
    delta_perp_sq  = ||delta||^2 - delta_parallel^2
    distance = sqrt((delta_parallel / R)^2 + delta_perp_sq)

A larger R makes the metric more forgiving of differences along the
quality axis and relatively more sensitive to differences in shape/style
-- the intent per similaritymodel.md is to find players who play a similar
*style*, not just players of similar overall quality.

Standardization (z-scoring each of the 12 categories to mean 0, std 1) is
computed once across the *pooled* cross-season data, not per season --
otherwise a z-score from 2018-19 and one from 2025-26 wouldn't be on the
same scale and cross-season comparisons would be meaningless.

In addition to the top-10 display, PREDICTED_SALARY_PCT_OF_CAP is computed
via Nadaraya-Watson kernel regression over the *entire* pool (every
player-season with a known salary, not just the top 10), per
contractprediction.md. This exists because averaging only 10 neighbors'
contracts caps the estimate at the highest of those 10 specific salaries
-- for a true outlier (Jokić, Curry), even the closest real comps sit at a
meaningfully large distance, so the top-10 average systematically
underestimates them. Kernel regression instead lets every player in the
pool contribute, weighted by a smoothly decaying function of distance, so
a wider tier of moderately-similar high-quality players still nudges the
estimate for an outlier instead of being ignored outright.

Usage:
    python build_similarity_model.py --ratio 4 --bandwidth auto --kernel gaussian
"""

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
VECTOR_DIR = DATA_DIR / "player_vectors_json"
SALARY_DIR = DATA_DIR / "salaries"
OUT_DIR = DATA_DIR / "similarity"

N_NEIGHBORS = 10

KERNELS = {
    "gaussian": lambda u: np.exp(-0.5 * u**2),
    "inverse_distance": lambda u: 1.0 / np.maximum(u, 1e-9),
}


def normalize_name(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return stripped.strip().lower()


def load_all_seasons():
    paths = sorted(VECTOR_DIR.glob("player_vectors_*.json"))
    if not paths:
        raise FileNotFoundError(f"no player_vectors_*.json found in {VECTOR_DIR}")

    rows = []
    categories = None
    for path in paths:
        season = path.stem.replace("player_vectors_", "")
        data = json.load(path.open(encoding="utf-8"))
        if categories is None:
            categories = list(next(iter(data.values()))["VECTOR"].keys())
        for pid, p in data.items():
            rows.append(
                {
                    "PLAYER_ID": pid,
                    "SEASON": season,
                    "PLAYER_NAME": p["PLAYER_NAME"],
                    "TEAM": p["TEAM"],
                    "VECTOR": [p["VECTOR"][c] for c in categories],
                }
            )
    return rows, categories


_salary_cache = {}


def get_salary_pct(season: str, player_name: str):
    if season not in _salary_cache:
        path = SALARY_DIR / f"salaries_{season}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8")
            _salary_cache[season] = {
                normalize_name(row["PLAYER_NAME"]): row["SALARY_PCT_OF_CAP"]
                for _, row in df.iterrows()
            }
        else:
            _salary_cache[season] = {}
    return _salary_cache[season].get(normalize_name(player_name))


def build_model(ratio: float, bandwidth: float | None = None, kernel: str = "gaussian") -> dict:
    rows, _ = load_all_seasons()
    X = np.array([r["VECTOR"] for r in rows])
    player_ids = [r["PLAYER_ID"] for r in rows]
    n = len(rows)

    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std

    cov = np.cov(Z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, np.argmax(eigvals)]
    pc1 = pc1 / np.linalg.norm(pc1)

    # dense n x n distance matrix -- n is the full cross-season player-season
    # count (a few thousand rows), still cheap for numpy.
    parallel_coord = Z @ pc1
    delta_parallel = parallel_coord[:, None] - parallel_coord[None, :]

    sq_norms = np.sum(Z**2, axis=1)
    gram = Z @ Z.T
    total_sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    delta_perp_sq = np.clip(total_sq_dist - delta_parallel**2, 0, None)

    distance = np.sqrt((delta_parallel / ratio) ** 2 + delta_perp_sq)

    if bandwidth is None:
        # median of the unique off-diagonal pairwise distances, per
        # contractprediction.md's default
        bandwidth = float(np.median(distance[np.triu_indices(n, k=1)]))
    kernel_fn = KERNELS[kernel]

    # salary_pct per row, for the kernel regression pass -- NaN where unknown
    salary_pct_all = np.array(
        [get_salary_pct(r["SEASON"], r["PLAYER_NAME"]) for r in rows], dtype=float
    )
    known_mask = ~np.isnan(salary_pct_all)
    player_id_arr = np.array(player_ids)

    results_by_season = {}
    for i, r in enumerate(rows):
        order = np.argsort(distance[i])
        # a neighbor can't be the same underlying player in another season
        neighbor_idx = [j for j in order if player_ids[j] != r["PLAYER_ID"]][:N_NEIGHBORS]

        neighbors = []
        weight_sum = 0.0
        weighted_pct_sum = 0.0
        for j in neighbor_idx:
            nb = rows[j]
            dist = float(distance[i, j])
            salary_pct = get_salary_pct(nb["SEASON"], nb["PLAYER_NAME"])
            neighbors.append(
                {
                    "PLAYER_NAME": nb["PLAYER_NAME"],
                    "TEAM": nb["TEAM"],
                    "SEASON": nb["SEASON"],
                    "DISTANCE": dist,
                    "SALARY_PCT_OF_CAP": salary_pct,
                }
            )
            if salary_pct is not None:
                weight = 1.0 / max(dist, 1e-9)
                weight_sum += weight
                weighted_pct_sum += weight * salary_pct

        weighted_avg_contract_pct = weighted_pct_sum / weight_sum if weight_sum > 0 else None

        # kernel regression over the *entire* pool (every known-salary
        # player-season, not just the top 10) -- see contractprediction.md
        eligible = known_mask & (player_id_arr != r["PLAYER_ID"])
        kernel_weights = kernel_fn(distance[i, eligible] / bandwidth)
        kw_sum = kernel_weights.sum()
        predicted_salary_pct = (
            float(np.sum(kernel_weights * salary_pct_all[eligible]) / kw_sum)
            if kw_sum > 0
            else None
        )

        entry = {
            "PLAYER_NAME": r["PLAYER_NAME"],
            "TEAM": r["TEAM"],
            "SEASON": r["SEASON"],
            "SALARY_PCT_OF_CAP": get_salary_pct(r["SEASON"], r["PLAYER_NAME"]),
            "NEAREST_NEIGHBORS": neighbors,
            "WEIGHTED_AVG_CONTRACT_PCT_OF_CAP": weighted_avg_contract_pct,
            "PREDICTED_SALARY_PCT_OF_CAP": predicted_salary_pct,
        }
        results_by_season.setdefault(r["SEASON"], {})[r["PLAYER_ID"]] = entry

    return results_by_season


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=float, default=4.0, help="parallel:perpendicular axis ratio")
    parser.add_argument(
        "--bandwidth",
        default="auto",
        help="kernel regression bandwidth (float), or 'auto' for the median pairwise distance",
    )
    parser.add_argument("--kernel", choices=list(KERNELS), default="gaussian")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    bandwidth = None if args.bandwidth == "auto" else float(args.bandwidth)
    results_by_season = build_model(args.ratio, bandwidth=bandwidth, kernel=args.kernel)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for season, players in results_by_season.items():
        out_path = out_dir / f"similarity_{season}.json"
        out_path.write_text(json.dumps(players, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(players)} player comps to {out_path}")


if __name__ == "__main__":
    main()
