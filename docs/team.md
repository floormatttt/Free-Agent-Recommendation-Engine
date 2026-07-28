# Team Vector Pipeline

Builds a team **need** vector: the same 12 categories as player.md/defense.md's
player vector, but directly from team-level stats
(`data/team_stats/team_stats_<season>.csv`) -- **not** aggregated from player
vectors. An earlier design built team vectors as a minutes-weighted average of
player vectors; that approach required a correction step because averaging
~15 player z-scores per team collapses the spread by the central limit
theorem. Computing directly from team-level stats sidesteps that problem
entirely: each number is already a team-level quantity, so z-scoring across
the 30 teams naturally lands at a full, comparable spread with no
re-anchoring needed.

Unlike the player vector (where a higher score is always better), each
category here is a **need score from 0 to 1**: 0 means the team is already
elite in that category (no need), 1 means the team is effectively
nonexistent in that category (maximum need). This is the intended output for
matching free agents against team needs -- a team's need score in a category
can be compared directly against a free agent's strength score in the same
category from the player vector.

---

## Input

`data/team_stats/team_stats_<season>.csv`: one row per team, with the 9
playtypes' `FREQ_PCT`/`PPP` (already team-level, from Synergy) and the raw
defense/rebounding inputs (`DEF_RATING`, `STL`, `BLK`, `DEFLECTIONS`,
`CONTESTED_SHOTS_3PT`, `DEF_BOXOUTS`, `OREB_PCT`, `DREB_PCT`, the rim
`RIM_DFG_PCT`/`RIM_DIFF_PCT`/`RIM_DFGA`, and `GP` for rate-adjusting the
season-total counting stats).

## Output

One need vector per team, same 12 categories as the player vector, keyed by
team abbreviation: `data/team_vectors_json/team_vectors_<season>.json`, one
file per season in `STANDARD_SEASONS`. Each category value is a need score in
`[0, 1]`, normalized against a rolling multi-season window (that season plus
its 2 trailing seasons by default) rather than just that file's own season
-- see "Need-score conversion" below.

---

## Playtype scores (9 categories)

Single season only (no multi-season decay blend -- see Design Notes), but
**structurally different from build_playtype_vectors.py's player-side
formula**, not just frequency-inverted:

```
freq_pct_rank = percentile_rank(FREQ_PCT)      # 1.0 = most frequent in the league
ppp_pct_rank  = percentile_rank(PPP)           # 1.0 = most effective in the league
ppp_badness   = 1 - ppp_pct_rank
need_raw      = freq_pct_rank * ppp_badness ** EFFECTIVENESS_GATE   # EFFECTIVENESS_GATE = 1
score         = 1 - need_raw
```

For a player, frequency and effectiveness point the same direction on
purpose: a notable *skill* is a playtype a player both leans on and is good
at, so player.md's `score` rewards high-frequency + high-effectiveness
together. Team *need* is a different question -- it's about exposure, not
skill -- a team that runs a playtype often *and* is bad at it is the real
weakness (highest need), while a team that rarely runs a playtype but
happens to be effective when it does isn't a live problem (lowest need).

**This is a gated interaction, not a weighted geometric mean of two
z-scores** -- an earlier version of this formula computed
`freq_z_inv^a * ppp_z^b` (frequency max-shifted and inverted, effectiveness
left as-is, weighted exponents), and it had two real problems, both found by
testing actual output against real teams:

1. The max-shift (`max(freq_z) - freq_z`) hits exactly `0` for whichever
   team has the single highest frequency in a category. `0` raised to any
   exponent is still `0`, so that team's score collapsed to `0` regardless
   of effectiveness -- confirmed on 2025-26 Spot Up, where Miami (34.3%
   frequency, next-highest was 28.7%) scored a `0` raw strength purely for
   being the frequency outlier, with above-average effectiveness (1.055 PPP)
   completely ignored.
2. Even after patching that specific collapse, a fixed-exponent weighted
   geometric mean structurally can't let effectiveness *fully* override an
   extreme frequency outlier -- any nonzero exponent on a very small
   frequency term still drags the product down meaningfully. Tested
   directly: a hypothetical team that was simultaneously the league's #1 in
   frequency *and* #1 in effectiveness still ranked dead last even at a
   0.25/0.75 (effectiveness-favoring) split.

The gated-interaction formula fixes both: `need_raw` is a product of *how
frequent* and *how bad*, so a team with `ppp_badness = 0` (literally the
best in the league at the playtype) gets `need_raw = 0` -- a full pass on
frequency -- no matter how extreme its frequency is. `EFFECTIVENESS_GATE`
controls how generously less-than-elite effectiveness forgives a
high-frequency team: `1` (the production default, chosen after comparing
`k=1` through `k=4` in the scatter/heatmap artifacts) is a straight
proportional trade-off -- being "half as bad" at a playtype forgives "half
as much" of the frequency penalty; higher values would forgive more readily
for meaningfully-above-average-but-not-literally-best effectiveness. At
`k=1`, 2025-26 Spot Up: Miami (good but not best effectiveness, extreme
frequency) lands at `0.72`, while Brooklyn (genuinely the league's worst
Spot Up effectiveness) correctly reads as the bigger weakness at `0.95`.

Percentile ranks (bounded in `(0, 1]`) are used instead of the zero-anchored
z-scores everywhere else in this pipeline specifically because
`EFFECTIVENESS_GATE` needs to function as an actual dial -- a z-score can
run arbitrarily large for an extreme outlier (confirmed: the real Miami's
frequency z-score was ~10, and a plain z-score's *scale* isn't bounded the
way a percentile rank's is), which is why raising the exponent on a
z-score-based version didn't meaningfully rescue an outlier team the way it
does here.

## Defense & rebounding (3 categories)

Same 3-block structure as defense.md, adapted where team_stats doesn't have
a direct player-level analog (see Known Limitations):

### Perimeter defense block

```
contested_3pt_pg = CONTESTED_SHOTS_3PT / GP
deflections_pg   = DEFLECTIONS / GP
stl_pg           = STL / GP                 # stands in for player-level STL%
activity_score   = contested_3pt_z^0.35 * deflections_z^0.35 * stl_z^0.30

adjusted_drtg    = DEF_RATING - (0.20 * activity_score * std(DEF_RATING))
adjusted_drtg_z  = (mean(adjusted_drtg) - adjusted_drtg) / std(adjusted_drtg)
perimeter_d_block = adjusted_drtg_z - min(adjusted_drtg_z)
```

No empirical-Bayes shrinkage step (unlike defense.md's on-court DEF_RATING)
-- a team's season `DEF_RATING` is already the full-sample outcome, not a
noisy low-minute on/off split, so there's nothing to shrink toward.

Rescaled afterward to match `RIM_BLOCK`/`REB_BLOCK`'s **std**, not mean.
defense.md's player-level pipeline rescales to match mean (its std's were
already comparable across blocks at 500-player sample size). That doesn't
hold at 30-team sample size: `PERIMETER_D_BLOCK`'s raw std already lines up
with the other two blocks, but its raw mean is tiny by comparison
(zero-anchored z-score means are far less stable with only 30 samples --
`REB_BLOCK`'s mean alone can run ~5x `RIM_BLOCK`'s). A uniform multiplier
that forces the mean to match drags the std along with it -- confirmed
empirically: `PERIMETER_DEFENSE`'s spread blew out to ~4x every other
category's (std ~4.7 vs ~1.0-1.2 elsewhere) under a mean-matching rescale.
Matching std instead keeps every category's spread comparable -- what
actually matters for any downstream z-score-based distance/regression use
-- while still preserving the zero-anchor (0 stays 0 under a positive
multiply).

### Rim protection block

```
blk_pg    = BLK / GP                        # stands in for player-level BLK%
dfga_pg   = RIM_DFGA / GP                   # RIM_DFGA on file is a season total
rim_block = BLK_PG_z^0.25 * DFG%_z_inv^0.30 * Diff%_z_inv^0.35 * DFGA_PG_z^0.10
```

`RIM_DFG_PCT`/`RIM_DIFF_PCT` are already team-level percentages, used as-is
(same inversion-via-max-shift as defense.md).

### Rebounding block

```
box_outs_pg = DEF_BOXOUTS / GP
reb_block   = (DREB_PCT_z * OREB_PCT_z * box_outs_pg_z) ^ (1/3)
```

`OREB_PCT`/`DREB_PCT` are already team-level rate stats, used as-is.

---

## Need-score conversion

Every category above (all 9 playtype scores plus the 3 defense/rebounding
blocks) is computed first as a **strength** score per team-season, same
shape and direction as the player vector (higher = better) -- this part is
still single-season, computed and z-scored against just that season's 30
teams (see "Playtype scores" / "Defense & rebounding" above).

The need conversion is where seasons come back in, but not the entire
`STANDARD_SEASONS` history at once -- a rolling window. For a given season,
it pools that season plus its `ROLLING_WINDOW - 1` (default 3, so 2
predecessors) trailing seasons -- e.g. 2025-26's need scores are normalized
against 2023-24/2024-25/2025-26 only, not all 8 standard seasons -- and each
category is standardized (z-scored) and squashed through a logistic sigmoid
across that rolling pool:

```
z          = (strength - mean(strength)) / std(strength)   # across the rolling pool
need_score = 1 / (1 + exp(z))
```

where `mean`/`std` are computed once across that season's ~90-team-season
rolling pool (fewer for the earliest seasons on file, which don't have 2 full
predecessors -- see `rolling_window()`), not the full 8-season history and
not just that season's own 30 teams. An average team-season lands at `0.5`;
need approaches `0` for a dominant team and `1` for a nonexistent one, but
-- unlike a min-max approach -- never actually reaches either endpoint.
That matters because a min-max version pins `0`/`1` to whichever single
team-season happens to be the pool's literal best/worst, an outlier that
then stretches or compresses everyone else's score around it; the sigmoid
instead treats "worst/best ever" as an asymptote, so one extreme
team-season doesn't distort the whole scale.

The rolling window (rather than pooling the full 8-season history) is
deliberate: the league goes through genuine playstyle shifts -- 3-point
volume, pace, defensive scheme trends -- so what counts as "elite" or
"nonexistent" in a category should be judged against how the league has
actually been playing the last few seasons, not averaged flat across a
decade that includes eras that no longer resemble the current game. A
team's need score is still judged against more than just its own season's
30-team field (so one season's specific cohort doesn't pin the scale), but
that reference frame itself now slides forward with the league rather than
staying fixed. This mirrors similarity_model.md's "standardize once,
across the pool" step for player vectors, adapted to a rolling window
instead of the full pool; the underlying per-team-season strength scores
themselves are still not blended across seasons (see Design Notes) --
pooling only widens (and now also slides) the reference frame those
single-season numbers are judged against.

This is done per category independently, so a team's 12-value need vector is
not itself standardized as a whole -- each entry only reflects that
team-season's standing within its own category's rolling-pooled
distribution.

A category with zero spread across the entire pool (`std == 0`) is scored as
a neutral `0.5` rather than dividing by zero -- not expected in practice
given real team-stat variance, but guards against a degenerate pool.

---

## Design Notes

- **No multi-season decay of the underlying stats -- but the need-score
  normalization does use a rolling multi-season window, and those are not
  the same thing.** Player vectors blend up to 3 seasons because a player's
  skill is a continuous, individually-tracked thing. A team's identity
  resets far more abruptly -- a coaching change, a core trade, or even just
  roster turnover can make last season's team a poor proxy for this one.
  Blending across seasons risks mixing genuinely different rosters rather
  than smoothing out noise, so each team-season's *strength* score is
  computed from that single season's stats only, never blended with any
  other season's. The rolling window in "Need-score conversion" above only
  changes what those single-season strength scores are *compared against*
  when converting to a need score -- it doesn't touch the strength scores
  themselves.
- **No CLT re-anchoring step.** The prior (player-aggregation) design of
  this pipeline needed a rescale-by-team-std step because averaging ~15
  players compressed the range. Since these numbers are already team-level,
  that problem doesn't exist here -- z-scoring directly across the 30 teams
  already produces a full spread.

## Validation

Two follow-up experiments test this pipeline's two core design bets from opposite ends: can the direct-from-team-stats `NEED_VECTOR` be *reconstructed* from a roster instead (revisiting the player-aggregation approach this doc pivoted away from, but as a learned mapping), and does the `NEED_VECTOR` this pipeline outputs actually *predict winning* (the thing it exists to proxy for)?

### Can a roster's player vectors reconstruct the team's NEED_VECTOR? (`predict_team_need_from_roster.py`)

Collapses each team-season's roster (variable player count) into a fixed 25-feature vector -- minutes-weighted mean and minutes-weighted std of each of the 12 player-vector categories, plus total tracked minutes -- then fits models to predict that team-season's actual `NEED_VECTOR`, validated via leave-one-season-out CV across `STANDARD_SEASONS` (~210 training team-seasons per fold). Two models compared against a naive (train-mean) baseline: 12 independent `XGBRegressor`s (one per category), and a single multi-output `PLSRegression` (chosen at the user's request as a better fit for correlated features/correlated targets with n not much larger than p; `n_components` picked per fold via inner 5-fold CV on that fold's training rows only).

**Results** (pooled across all 8 LOSO folds, RMSE on `NEED_VECTOR`'s `[0,1]` scale):

| Model | Pooled RMSE | vs. naive |
|---|---|---|
| Naive baseline (train-mean) | 0.2095 | -- |
| XGBoost (12 independent models) | 0.1976 | +5.7% |
| PLS (joint multi-output) | 0.1877 | +10.4% |

Per-category (sorted by PLS RMSE, best to worst):

| Category | Naive | XGBoost | PLS |
|---|---|---|---|
| PERIMETER_DEFENSE | 0.2122 | **0.1465** | 0.1518 |
| SPOT_UP | 0.2104 | 0.1915 | **0.1724** |
| PNR_HANDLER | 0.2108 | 0.1933 | **0.1749** |
| RIM_DEFENSE | 0.2076 | 0.2045 | **0.1839** |
| HANDOFF | 0.2111 | 0.1926 | **0.1892** |
| ISOLATION | 0.2037 | 0.1827 | **0.1896** |
| PUT_BACKS | 0.2090 | 0.1982 | **0.1904** |
| REBOUNDING | 0.2145 | 0.2136 | **0.1944** |
| CUT | 0.2115 | 0.2148 | **0.1967** |
| PNR_ROLLMAN | 0.2026 | 0.2023 | **0.1974** |
| OFF_SCREEN | 0.2081 | 0.2138 | **0.1993** |
| POST_UP | 0.2122 | 0.2074 | **0.2057** |

PLS beats XGBoost in every category except `PERIMETER_DEFENSE`, where XGBoost's win over both is the largest single-category edge either model has (~31% vs. naive) -- consistent with `predict_win_pct_from_need.py` below finding `PERIMETER_DEFENSE` the single strongest predictor of winning, i.e. it's also the category where a roster's own personnel most cleanly determines team need. Two `CUT`/`OFF_SCREEN` rows show XGBoost doing *worse* than naive, with a per-fold train/test RMSE gap (e.g. `PERIMETER_DEFENSE` 0.014 train vs. 0.081 test on one fold) indicating real overfitting at ~210 rows for the 12-independent-model setup.

**Conclusion: a roster's aggregate player vectors only partially reconstruct the team-stats-derived `NEED_VECTOR`, and do so unevenly across categories** -- confirming (rather than just assuming, as the original pivot away from player-aggregation did) that team-level need genuinely carries information beyond what's recoverable from the roster's individual player vectors alone. This doesn't change the production pipeline (team.md's direct-from-team-stats approach remains what's shipped) -- it's a validation check on whether that pivot actually mattered, and it does.

### Does NEED_VECTOR predict winning? (`predict_win_pct_from_need.py`)

Tests whether the 12-category `NEED_VECTOR` predicts a team's actual win percentage (`data/team_wins/team_wins_<season>.csv`, pulled via `LeagueDashTeamStats`) -- i.e. whether the weaknesses this pipeline identifies actually cost teams wins. ~240 team-seasons (30 teams x 8 `STANDARD_SEASONS`), 12 features predicting `W_PCT`, evaluated via LOSO CV. Two regularized linear models (Ridge, Elastic Net -- chosen over a black-box model for interpretability: at this sample size the more useful question is "which categories predict losing," not just "what's the RMSE"), regularization strength picked via inner 5-fold CV per fold, plus a 1-feature baseline (`COMPOSITE_NEED`, the mean of all 12 categories) to check whether the category breakdown adds anything over "how needy is this team overall."

**Results** (LOSO RMSE on `W_PCT`):

| Model | RMSE |
|---|---|
| Naive baseline (train-mean) | 0.1480 |
| Composite need (1 feature) | 0.0872 |
| Ridge (12 features) | **0.0630** |
| Elastic Net (12 features) | 0.0630 |

The 12-category breakdown clearly beats the single composite-need number (0.063 vs. 0.087 RMSE), confirming the category-level detail carries real signal about *why* a team is losing, not just *that* it is. Ridge and Elastic Net land almost identically (Elastic Net's L1 term doesn't zero out any category -- see coefficients below), so there's no evidence any category is pure noise for this target.

Ridge coefficients (averaged across LOSO folds; negative = more need in that category -> fewer wins, which is the expected sign since `NEED_VECTOR` runs 0=best/no need to 1=worst/nonexistent):

| Category | Coefficient |
|---|---|
| PERIMETER_DEFENSE | **-0.355** |
| SPOT_UP | -0.179 |
| PNR_HANDLER | -0.167 |
| HANDOFF | -0.081 |
| ISOLATION | -0.084 |
| OFF_SCREEN | -0.058 |
| PNR_ROLLMAN | -0.058 |
| REBOUNDING | -0.058 |
| POST_UP | -0.054 |
| CUT | -0.046 |
| RIM_DEFENSE | -0.044 |
| PUT_BACKS | -0.008 |

`PERIMETER_DEFENSE` is by far the single strongest predictor of losing -- roughly double the next-largest coefficient -- with `SPOT_UP` and `PNR_HANDLER` need (i.e. weak shooting/creation in the league's two highest-value shot types) next. `PUT_BACKS` carries almost no weight, consistent with it being the lowest-value/lowest-frequency playtype league-wide.

A trimmed export of this run (`rmse`, ridge `coef_mean`, and the ridge calibration table only -- naive/composite/enet coefficients and per-fold breakdowns dropped) was written to `data/win_pct_artifact_data.json` for charting; it is not read by any pipeline script and isn't regenerated automatically if `predict_win_pct_from_need.py` is re-run.

**Conclusion: yes -- the `NEED_VECTOR` this pipeline produces predicts real winning, well above both a naive baseline and a single composite-need number, and the category with the strongest signal (`PERIMETER_DEFENSE`) is also the category the roster-reconstruction experiment above found hardest to fake from personnel alone.** Neither script's output currently feeds back into the production team-vector or contract-prediction pipelines -- both are standalone validation checks, not inputs to `build_team_vectors.py` or `build_contract_model_v3.py`.

---

## Known Limitations

- **STL_PG and BLK_PG are per-game substitutes, not true rate stats.**
  Player-level `PCT_STEALS`/`PCT_BLOCKS` are normalized against opponent
  possessions/shot attempts; `team_stats` has no equivalent team-level rate
  on file, so per-game counts (already the convention used for the other
  counting stats in this pipeline) are the closest available substitute.
  Since all 30 teams play a similar number of possessions per game, this is
  a reasonable approximation, not a pace-adjusted one -- a team-level
  `STL_PCT`/`BLK_PCT` (or a per-100-possessions rate, using the `POSS`
  column already available from the Advanced team-stats endpoint) would be
  more rigorous if added later.
- **RIM_DFGA is a season total on file (`per_mode_simple="Totals"`), not
  already a rate stat**, so it's divided by `GP` here -- important given
  that `GP` can vary team to team, especially mid-season.
- **No team-level equivalent of the player pipeline's `EXPERIENCE_YRS`
  decay weighting** -- see Design Notes above for why that's a deliberate
  omission, not a gap.
