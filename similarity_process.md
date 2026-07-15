# Similarity & Contract Prediction Process

Covers `build_similarity_model.py` -- the pipeline from player.md/defense.md's 12-category vector to a similar-player list and a predicted contract, per similaritymodel.md. Consolidates and supersedes similaritymodel.md and contractprediction.md as the record of what's actually built, tested, and confirmed.

Output: `data/similarity/similarity_<season>.json`, one file per season, keyed by `PLAYER_ID`.

---

## Pipeline (current state)

### Step 1 -- Pool every season

Loads all 8 seasons of `data/player_vectors_json/` (~4,466 player-seasons total) into one matrix. One player who appears in multiple seasons contributes one row per season.

### Step 2 -- Standardize once, across the pool

Z-score each of the 12 categories (mean 0, std 1) across the *entire pooled* population, not per season -- a z-score from 2018-19 and one from 2025-26 need to be on the same scale for cross-season comparison to mean anything.

### Step 3 -- Parallel axis (PC1)

PC1 (first principal component) of the standardized pool was intended as the "quality axis" -- the idea being that good players tend to be good across many categories at once, so the dominant shared direction of variation would reflect overall quality. **This assumption was tested directly against salary and found to be false** (see Changelog): PC1 correlates ~0 with `SALARY_PCT_OF_CAP` despite explaining ~40% of the pool's variance. It more likely reflects offensive usage/involvement (players who touch the ball more score higher across *multiple* playtype categories simultaneously) than value. PC1 is still used as the parallel axis for the distance metric below, since that's similaritymodel.md's specified design, but it should not be treated as a validated "quality" measurement anywhere else.

### Step 4 -- Anisotropic distance

Given ratio `R` (default 4, per similaritymodel.md's 4:1/5:1 spec):

```
delta = z_i - z_j
delta_parallel = delta . pc1
delta_perp_sq  = ||delta||^2 - delta_parallel^2
distance = sqrt((delta_parallel / R)^2 + delta_perp_sq)
```

Computed across the full pool. A neighbor can never be the same underlying player in a different season.

### Step 5 -- Top 10 comps (display)

For each player-season, the 10 nearest by Step 4 distance, with each neighbor's own season's `SALARY_PCT_OF_CAP` (not raw dollars -- see Changelog) where known. Also reports `WEIGHTED_AVG_CONTRACT_PCT_OF_CAP`, the `1/distance`-weighted average of just those 10 neighbors' contracts. This output remains useful for "who does this player resemble" independent of the contract prediction below.

### Step 6 -- Kernel regression contract prediction

`PREDICTED_SALARY_PCT_OF_CAP`: Nadaraya-Watson kernel regression over *every* player-season in the pool with a known salary (not just the top 10):

```
predicted_salary_pct(player) = sum_i( K(distance_i / h) * salary_pct_i ) / sum_i( K(distance_i / h) )
```

Gaussian kernel by default (`K(u) = exp(-0.5*u^2)`); bandwidth `h` configurable, no longer defaults to "auto" (median pairwise distance) -- see Changelog for why that default was wrong.

**Status: not accurate enough to use as-is.** See Changelog for the full diagnosis. Current recommendation on record: pivot to a direct supervised regression (e.g. XGBoost, validated via leave-one-season-out) rather than continuing to patch this approach, since the core problems (bandwidth-driven over-smoothing, archetype-dominated-by-cheap-players bias, PC1 not being quality) are structural to the distance/neighbor framing, not tuning issues. Not yet implemented.

---

## Salary-cap normalization

All contract figures use `SALARY_PCT_OF_CAP`, not raw dollars, because neighbors can be drawn from any season across a decade of cap growth -- a $30M salary meant something very different in 2018-19 than in 2025-26, but the same salary's share of that season's cap is comparable.

---

## Rookie-scale contract contamination

Rookie-scale deals are fixed by draft slot per the CBA, not market-negotiated. A great 2nd-year player on a cheap rookie deal looks like "great player, tiny contract" to any model, corrupting the learned quality-to-pay relationship. Filtering to `EXPERIENCE_YRS > 4` (past the rookie-scale window) measurably improves correlation for simple predictors (see Changelog) and should be applied to the training population of whatever contract-prediction approach is used going forward -- **not yet implemented as a standard filter in the pipeline itself.**

---

## Origin-distance experiments (exploratory, not part of the pipeline)

Tested as simpler alternatives to the full neighbor/kernel approach, using a single scalar per player instead of a distance matrix:

| Predictor | Corr. w/ salary_pct (all players) | Corr. w/ salary_pct (veteran-only, exp>4) |
|---|---|---|
| PC1 coordinate alone | -0.006 | -- |
| Standardized-vector Euclidean norm | 0.27 | -- |
| Raw-vector Euclidean norm | 0.50 | 0.51 |
| Raw-vector geometric mean ("generalist" skew) | 0.50 | **0.51** |
| Kernel regression (full pipeline, tuned bandwidth) | **0.72-0.78** | -- |

The geometric mean version (product of all 12 raw category scores, 12th root) was built specifically to reward all-around production over specialization -- a single very low category drags the geometric mean down hard, unlike a plain norm. It modestly outperforms plain norm once rookie-scale noise is filtered, but both remain well below the full neighbor-based approach. Conclusion: a one-dimensional summary leaves too much signal on the table; a real predictive model needs the full 12-dimensional relationship.

For reference, "furthest from origin" (2025-26, min. 1000 minutes) looks different depending on which version: plain Euclidean norm surfaces single-skill specialists who post one huge number (Zubac, Kalkbrenner, Queta, Kornet, Gafford, Jarrett Allen all crack that top 25 via rim protection/rebounding alone), while the geometric-mean version excludes all of them and surfaces only modern do-everything hybrids (Wembanyama, Holmgren, Jokić, Towns, Mobley, Barnes, Reid, Siakam, Randle, Bam).

---

## Known Limitations

- **PC1 is not a validated quality axis** -- it's still used as the parallel axis per similaritymodel.md's design, but should not be relied on as a quality measurement elsewhere.
- **Kernel regression is bounded by the observed salary range** -- it's a weighted average of real contracts, so it can never predict higher than the highest `SALARY_PCT_OF_CAP` anywhere in the pool. This matters less than it sounds, since max contracts are hard-capped by the CBA at a fixed % of cap by years of service -- a real ceiling that already exists in the historical data.
- **Kernel regression's archetype bias**: even at a well-tuned bandwidth, a player whose style archetype is dominated league-wide by cheap/replacement-level players (e.g. limited-offense rim protectors) gets a badly underestimated prediction, because the metric can't separate "elite at this archetype" from "replacement level at this archetype" using only style-weighted distance.
- **Ratio `R` and bandwidth `h` are both manually chosen**, not data-driven. A cross-validated choice (e.g. minimizing LOSO prediction error) would be more rigorous.
- **Salary-name matching** relies on accent-stripped, case-insensitive normalization; a residual ~21 salary-list names don't match any vector player due to suffix/nickname formatting differences (e.g. "Jimmy Butler" vs "Jimmy Butler III").

---

## Changelog

- Built `build_similarity_model.py` per similaritymodel.md: single-season standardized vectors, PC1 as parallel axis, anisotropic distance (ratio 4), top-10 neighbors excluding self, salary joined by accent-normalized player name.
- Diagnosed salary-join gaps: ~172 of ~582 vector players per season have no salary (the ranked salary list is a top-~450 cutoff, not exhaustive); accent normalization recovered matches for names like "Jokić"; ~21 residual mismatches remain (suffix/nickname formatting).
- Extended to cross-season pooling (~4,466 player-seasons across 8 seasons): standardization and PC1 computed once on the pooled data; neighbor search excludes the same player across seasons; each neighbor's salary comes from its own season. Verified: Jokić's cross-season comps span 5 different seasons and include 4 different vintages of Karl-Anthony Towns.
- Switched all contract figures from raw `SALARY` to `SALARY_PCT_OF_CAP` once cap-history data existed, since raw dollars aren't comparable across a decade of cap growth. Verified against real CBA max-salary rules.
- Wrote contractprediction.md diagnosing two problems with using the top-10 average as an actual *prediction* (rather than just a display of comps): no usable salary prior for genuine free agents, and a hard ceiling at the max of 10 specific neighbors' salaries that systematically underestimates true outliers.
- Implemented Nadaraya-Watson kernel regression (`PREDICTED_SALARY_PCT_OF_CAP`) over the full pool, Gaussian/inverse-distance kernel options, configurable bandwidth.
- **Diagnosed severe kernel regression problems:**
  - Default "auto" bandwidth (median pairwise distance, ~3.67) caused massive over-smoothing: predictions collapsed toward the population mean (~7.6%) regardless of player quality -- predicted std 0.70 vs actual std 8.71, predicted range compressed to 5.76-9.42% vs actual 0.01-38.5%.
  - Bandwidth sweep found peak correlation (~0.77-0.78) around bw=0.6-0.9, but even there, MAE=3.98pp, RMSE=5.68pp, 90th-percentile error=9.2pp, and worst-case misses up to 23.8 percentage points of cap (Anthony Davis: actual 35%, predicted 11%). 8 of the 10 worst misses were stars systematically underestimated.
  - Root cause traced past bandwidth: Rudy Gobert's nearest neighbors even at tight bandwidth are cheap bench-caliber rim-runners (Capela, Looney, Ed Davis, Favors, Poeltl, Bitadze, Queta, Mitchell Robinson, Zubac), because the R=4 ratio (designed to discount quality in favor of style-matching) can't separate "elite at this archetype" from "replacement level at this archetype" when the whole archetype skews cheap.
  - Tested whether PC1 actually correlates with pay (the core assumption behind the R=4 ratio's rationale): found ~0 correlation (-0.006). This invalidated the "PC1 = quality" assumption baked into the original design.
- Given the above, current recommendation on record (not yet implemented): pivot to a direct supervised regression (XGBoost, LOSO-validated) rather than continuing to patch the distance/kernel approach.
- Tested "distance from origin" as a simpler single-number predictor at the user's request: raw-vector Euclidean norm (0.50 correlation) beats standardized-vector norm (0.27), but both are well below kernel regression.
- Tested a "generalist-skewed" version (geometric mean of the 12 raw categories) per the user's request to reward all-around production over specialization: modestly outperforms plain norm (0.51 vs 0.51 -- effectively tied on veteran-only, geo mean ekes ahead) once rookie-scale contracts are filtered out, but still far below kernel regression.
- Tested rookie-scale contract filtering (`EXPERIENCE_YRS > 4`) as a general data-quality fix: modestly improved both origin-distance metrics; flagged as something that should be applied to any future model's training population, not yet implemented as a standing pipeline filter.
- Computed "furthest from origin" leaderboards both ways (plain norm vs. geometric mean) for 2025-26, confirming the geometric-mean version excludes single-skill specialists that the plain-norm version rewards.
