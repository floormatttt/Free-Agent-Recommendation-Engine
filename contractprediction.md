# Player Similarity & Contract Prediction Pipeline

---

## Inputs

- Player vectors (the 12-category vector from player.md / defense.md), pooled across every season on file (~4,400 player-seasons)
- `SALARY_PCT_OF_CAP` for every player-season that has a known contract (data/salaries/), so contracts are comparable across a decade of cap growth

## Outputs

- List of the 10 closest players (by style/quality, drawn from any season in the pool) for interpretability -- "who does this player play like"
- A predicted contract (`SALARY_PCT_OF_CAP`) for any player, via kernel regression over the whole pool -- not just the top 10

---

## Process

### Step 1 — Standardize the pooled vectors

Z-score each of the 12 categories (mean 0, std 1) across the full cross-season pool, not per season. The categories don't share a natural scale, and standardizing once across the pool (rather than per season) keeps a z-score from 2018-19 and one from 2025-26 on the same footing, which cross-season comparison requires.

### Step 2 — Find the parallel axis

Compute PC1 (first principal component) of the standardized pool. Empirically this is the dominant shared direction of variation, which in practice reads as general "quality" -- players who are good tend to be good across many categories at once, correlating together along this axis. Everything orthogonal to PC1 (the "perpendicular axes") captures archetype/style differences independent of that shared quality direction.

### Step 3 — Anisotropic distance

Given a parallel:perpendicular axis-length ratio `R` (4:1, per similaritymodel.md):

```
delta = z_i - z_j                          (in standardized space)
delta_parallel = delta . pc1                (signed scalar)
delta_perp_sq  = ||delta||^2 - delta_parallel^2
distance = sqrt((delta_parallel / R)^2 + delta_perp_sq)
```

A larger `R` makes the metric more forgiving of differences along the quality axis and relatively more sensitive to differences in shape/style -- the point is to find players who play a similar *style*, not just players of similar overall quality.

Distances are computed across the entire cross-season pool. A neighbor can never be the same underlying player in a different season -- that would just be comparing someone to themselves.

### Step 4 — Report the 10 closest players

For interpretability: the 10 nearest player-seasons by the Step 3 distance, each with its own season's `SALARY_PCT_OF_CAP` where known. This output is retained from the original Similar Player Model and is useful on its own for "who does this player resemble," independent of the contract prediction in Step 5.

### Step 5 — Predict the contract via kernel regression

The original model reported a distance-weighted average of just those 10 neighbors' contracts. Two problems with using that average as an actual *prediction*, rather than just a display of comps:

- **No usable prior for free agents.** A free agent either has no current contract, or their expiring contract is exactly the thing being replaced by a prediction (a rookie-scale deal was never market-negotiated, so anchoring to it re-injects the bias the prediction is supposed to correct). So blending toward the player's own salary isn't an option here.
- **Top-10 averaging breaks down for outliers.** A hard cutoff at 10 neighbors means the estimate can never exceed the *highest* salary among those specific 10 players. For a true outlier (Jokić, Curry), even the closest available comps sit at a meaningfully large distance, so averaging them systematically underestimates value -- the model has no way to say "better than everyone in the comp pool."

Instead, predict via Nadaraya-Watson kernel regression over the **entire** pool, where every player contributes, weighted by a smoothly decaying function of distance rather than a hard cutoff:

```
predicted_salary_pct(player) = Σᵢ K(distance_i / h) × salary_pct_i
                                ─────────────────────────────────────
                                Σᵢ K(distance_i / h)
```

- The sum runs over every player-season with a known contract in the pool, not a fixed top-N.
- `distance_i` is the Step 3 distance.
- `K` is a kernel function, largest at distance 0 and decaying toward 0 as distance grows. Candidates:
  - Gaussian: `K(u) = exp(-u^2 / 2)`
  - The `1/distance` weighting used for the Step 4 display (a valid but more aggressive kernel)
- `h` (bandwidth) controls how fast weight decays with distance. Small `h` behaves close to k-NN (only near neighbors matter); large `h` smooths over more of the population. A reasonable default is the median pairwise distance across the pool, tuned after inspecting behavior on known outliers (Jokić, Curry) and common archetypes.

No hard neighbor cutoff means a true outlier still receives contribution from the next tier of moderately-similar players (distance 3, 4, 5, ...) instead of being capped by whoever happened to rank 10th.

Kernel regression needs nothing from the target player besides their vector -- a free agent with no current contract, or one whose current deal is irrelevant to their market value, is handled identically to any signed player. The prediction comes entirely from where their vector sits relative to the pooled population, not from any contract of their own.

---

## Known Limitations

- **Bounded by the observed range.** Kernel regression is a weighted average of observed contracts, so the prediction can never exceed the highest `SALARY_PCT_OF_CAP` anywhere in the pooled dataset. It smooths, it does not extrapolate. (Tree-based regression, e.g. XGBoost, shares this exact limitation -- it also predicts via averaging leaf-node training examples. Only a genuinely parametric fit, such as a smooth curve fit directly to the quality axis, can extrapolate past the observed range.)
- **This matters less here than it would generally**, because max contracts are hard-capped by the CBA at a fixed percentage of the cap by years of service (confirmed empirically: Giannis/Butler/Davis all compute to exactly 35.000000% of the 2025-26 cap). The ceiling a model needs to predict up to is a real, legally fixed ceiling that already exists in the historical data, not an open-ended target -- so the inability to extrapolate past the observed max is a much smaller practical problem than it would be for an unconstrained continuous outcome.
- **Kernel and bandwidth are not yet data-driven.** `K` and `h` are manually chosen starting points, the same category of limitation already flagged for `K_SHRINKAGE` in defense.md. A cross-validated bandwidth (e.g. minimizing LOSO prediction error) would be more rigorous, and ties into the LOSO XGBoost validation question raised separately.
- **The parallel:perpendicular ratio (`R`) and the choice of PC1 as the parallel axis are also manually chosen**, not fit from data -- same category of limitation as the kernel/bandwidth choice above.
- **Cross-season pooling requires salary-cap normalization**, which is already in place (`SALARY_PCT_OF_CAP`, not raw dollars) -- without it, `salary_pct_i` values from different eras of cap growth wouldn't be comparable.
