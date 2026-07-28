# Contract Model (`build_contract_model_v3.py`)

Covers the supervised-regression contract-prediction pipeline: an XGBoost model predicting `SALARY_PCT_OF_CAP` from a player's 12-category vector plus a handful of contextual features. This is the second of two competing contract-prediction approaches in the repo -- see `similarity_model.md` for the other (a distance/kernel-regression approach over the same 12-category vector), and `free_agent_evaluation.md` for how they compare against real free-agent signings.

**Versioning**: each materially different iteration of this pipeline gets a version number and its own build script (`build_contract_model.py` was v2, `build_contract_model_v3.py` is v3), rather than accumulating flags/branches in one script. `data/salary_prediction/` is kept to only the current version's output at any time -- superseded versions' scripts and CSVs are deleted once a new version is validated as better, not archived, so there's exactly one file per artifact and no ambiguity about which is current.

**v3 is the current recommended pipeline** -- output `data/salary_prediction/contract_model_v3_predictions.csv`, kept deliberately slim (`PLAYER_ID`, `PLAYER_NAME`, `PREV_SEASON`, `TARGET_SEASON`, `ACTUAL_SALARY_PCT_OF_CAP`, `PREDICTED_SALARY_PCT_OF_CAP`, `RESIDUAL_PP`). See "v3" below for what changed from v2 and why.

---

## Salary-cap normalization

All contract figures use `SALARY_PCT_OF_CAP`, not raw dollars, because both this model and the similarity model draw on data across a decade of cap growth -- a $30M salary meant something very different in 2018-19 than in 2025-26, but the same salary's share of that season's cap is comparable. This convention was adopted first in the similarity model (see `similarity_model.md`'s Changelog) and carried forward unchanged here.

## Rookie-scale contract contamination

Rookie-scale deals are fixed by draft slot per the CBA, not market-negotiated. A great 2nd-year player on a cheap rookie deal looks like "great player, tiny contract" to any model, corrupting the learned quality-to-pay relationship. This pipeline applies the fix that was diagnosed (but left unimplemented) against the similarity model: **training rows are restricted to `EXPERIENCE_YRS > 4`** (`ROOKIE_SCALE_CUTOFF` in `build_contract_model.py`), past the rookie-scale window. The fitted model is still *applied* to every player at prediction time, including those still on rookie-scale deals -- estimating open-market value regardless of current pay is exactly the free-agent-valuation use case (many restricted free agents, e.g. 4th-year players signing second contracts, are on the edge of this cutoff).

---

## v1: the original supervised-regression attempt (historical, superseded)

Following the similarity model's diagnosis, a first direct supervised regression predicting `SALARY_PCT_OF_CAP` was built and run across the full 8-season pool, output to `data/salary_prediction/predicted_vs_actual_all_seasons.csv` (4,466 rows). **That file no longer exists in the repo** -- once its input features were extracted into `player_season_features.csv` (see below) and v2 was built on top of that, the v1 output (which mixed those inputs with v1's own now-superseded predictions) was deleted as part of a salary_prediction/ cleanup, keeping the folder to only the current pipeline's data.

**The script that generated this file was never present in this documentation pass either** -- the following is reverse-engineered from the output CSV itself, not from reading source, so treat exact methodology details as inferred rather than confirmed:

- `ROOKIE_CONTRACT` is exactly `EXPERIENCE_YRS <= 4` (verified directly against the data) -- the same rookie-scale threshold now formalized as `ROOKIE_SCALE_CUTOFF` below.
- Features visible in the output were `EXPERIENCE_YRS`, `ROOKIE_CONTRACT`, `USG_PCT` -- notably *not* the 12-category vector itself.
- **Predictions went negative** (min observed: -10.7%) for a real percentage-of-cap figure, which cannot be negative -- a strong signal it was a plain linear/ridge regression rather than a tree ensemble (trees can't extrapolate below their training targets' minimum the way a linear fit can).
- Overall performance (rows with known salary, n=3,389): correlation 0.70, MAE 5.04pp, RMSE 6.34pp. Restricted to non-rookie-contract players (n=1,913): correlation improved to 0.77, MAE 4.95pp -- roughly in line with the similarity model's best-tuned kernel regression (0.72-0.78 correlation), not clearly better or worse.
- Award history (All-Star/All-NBA selections, via the `PlayerAwards` endpoint) had apparently been fetched and merged into this same CSV at some point (`ALL_STAR_SELECTION`, `ALL_NBA_TEAM_NUMBER`, `ALL_NBA_SELECTION` columns exist, populated for all 4,466 rows) -- though the fetching script (`_fetch_awards.py`) and its raw output no longer exist in the repo, and the columns weren't used as features in v1's own predictions.

Its input columns (`PLAYER_ID`, `PLAYER_NAME`, `TEAM`, `SEASON`, `EXPERIENCE_YRS`, `ROOKIE_CONTRACT`, `USG_PCT`, `SALARY_PCT_OF_CAP`, `ALL_STAR_SELECTION`, `ALL_NBA_TEAM_NUMBER`, `ALL_NBA_SELECTION`) were extracted into `data/salary_prediction/player_season_features.csv` before deletion, dropping only v1's own `PREDICTED_SALARY_PCT_OF_CAP`/`RESIDUAL_PP` columns -- verified to join 1:1 by `(PLAYER_ID, SEASON)` against every row in the player-vector pool, so this is exact reuse of on-file data, not an approximation. This decouples v2 from v1's output entirely: v2 now reads its auxiliary features from `player_season_features.csv`, not from any v1 prediction file.

---

## v2: full rebuild (`build_contract_model.py` -- deleted, superseded by v3 below)

**This script and its output (`contract_model_v2_predictions.csv`) no longer exist in the repo** -- v3 beat it on every metric checked (see "v3" below), and per the versioning policy in this doc's header, superseded versions are deleted rather than kept alongside. This section is retained as history: v3 reuses v2's feature set, hyperparameters, and out-of-fold discipline unchanged, only swapping the training population and adding a post-processing gate.

A from-scratch, reproducible replacement for v1, acting on the standing recommendation to move past both the unreproducible v1 script and the similarity model's kernel regression.

### Features

- The full 12-category player vector (same input as the similarity model): `ISOLATION`, `PNR_HANDLER`, `PNR_ROLLMAN`, `POST_UP`, `SPOT_UP`, `HANDOFF`, `CUT`, `OFF_SCREEN`, `PUT_BACKS`, `RIM_DEFENSE`, `PERIMETER_DEFENSE`, `REBOUNDING`.
- `EXPERIENCE_YRS`, `USG_PCT` (reused from `data/salary_prediction/player_season_features.csv`, not re-fetched from `nba_api`).
- **New relative to v1**: `ALL_STAR_SELECTION` and `ALL_NBA_SELECTION` (both 0/1), also reused from the same CSV.

`ROOKIE_CONTRACT` is not included as a feature -- it's used only to build the training filter below, and would be constant (always `False`) within the filtered training population anyway.

### Model & training

- One `XGBRegressor` (`max_depth=3`, `reg_alpha=0.1`, `reg_lambda=1.0`, `n_estimators=200`, `random_state=0` -- the same hyperparameter convention as `predict_team_need_from_roster.py`'s XGBoost model, for consistency across the repo's tree-based models).
- Training rows restricted to `EXPERIENCE_YRS > 4` and known `SALARY_PCT_OF_CAP` (see Rookie-scale contamination above).
- `fit_full_model()` trains on every eligible season at once. **This model's predictions are not exported** -- see "Out-of-fold exports" below for why.

### Out-of-fold exports (leakage fix)

An earlier version of this pipeline exported `fit_full_model()`'s own predictions -- i.e. a model trained on every season, predicting rows it had already seen during training. For a player like Kyrie Irving, whose actual 2024-25 salary was itself one of the training labels, that "prediction" had already seen the right answer. Caught when the user asked "are you sure these are correct?" and prompted a direct check: in-sample MAE was 2.94pp / corr 0.876, meaningfully better than the honest LOSO numbers below (4.2-4.4pp / 0.76-0.82) -- confirming the gap was leakage, not real accuracy.

Fixed via `out_of_fold_predictions()`: every row is predicted by the model from the LOSO fold that *excluded* its own season, so no row's prediction ever comes from a model trained on that row's true salary. This is what's now written to `contract_model_v2_predictions.csv` and what `evaluate_free_agents.py` reads. `fit_full_model()` is kept only for the feature-importance diagnostic below -- its predictions are never exported or scored anywhere.

### Validation: leave-one-season-out (LOSO)

Per `similarity_model.md`'s standing recommendation and the same convention already used in `predict_team_need_from_roster.py`. Each fold trains on 7 seasons' veteran rows and predicts the 8th, scored two ways:

- **All held-out rows** (veteran + rookie-scale) -- the realistic deployment scenario, since a fielded model has to value everyone.
- **Veteran-only held-out rows** -- a fairer yardstick for the model itself, since fixed-slot rookie pay isn't something a quality-based model should be expected to reproduce.

**Results** (pooled across all 8 folds), against a naive train-mean baseline (MAE 7.48pp, RMSE 8.82pp):

| Subset | N | MAE | RMSE | Corr | Bias |
|---|---|---|---|---|---|
| All held-out rows | 3,389 | 4.36pp | 5.83pp | 0.763 | -1.67pp |
| Veteran-only held-out rows | 1,913 | 4.20pp | 5.57pp | **0.823** | -0.23pp |

Both clearly beat v1 (0.70/0.77 correlation) and the similarity model's kernel regression (0.72-0.78, achieved only after manual bandwidth tuning).

### Feature importances (full model)

| Feature | Importance |
|---|---|
| `USG_PCT` | 0.309 |
| `ALL_STAR_SELECTION` | 0.207 |
| `ISOLATION` | 0.094 |
| `EXPERIENCE_YRS` | 0.060 |
| `OFF_SCREEN` | 0.058 |
| `POST_UP` | 0.050 |
| `SPOT_UP` | 0.038 |
| `REBOUNDING` | 0.036 |
| `PNR_HANDLER` | 0.028 |
| `HANDOFF` | 0.022 |
| `RIM_DEFENSE` | 0.022 |
| `CUT` | 0.020 |
| `PERIMETER_DEFENSE` | 0.018 |
| `PNR_ROLLMAN` | 0.016 |
| `PUT_BACKS` | 0.013 |
| `ALL_NBA_SELECTION` | 0.007 |

`USG_PCT` and `ALL_STAR_SELECTION` together carry over half the model's weight -- pay tracks recognized stardom and offensive load far more than any single playtype skill. `ALL_NBA_SELECTION` barely registers, likely because it's rare (120/4,466 rows) and mostly redundant with All-Star + high usage.

---

## v3: trained on free-agent signings, plus a minutes-reliability gate (`build_contract_model_v3.py`)

Two changes from v2, both requested directly by the user and validated against the free-agent eval harness rather than assumed to help.

### Change 1: train on real signings, not the full historical pool

v2's target -- a player's contemporaneous `SALARY_PCT_OF_CAP` in the full 8-season pool -- is contaminated by **contract-lag noise**: most rows are players mid-way through a multi-year deal, so their salary that season reflects *past* performance, not present (see the Westbrook/Beal/Simmons/Wade/Nowitzki examples surfaced while investigating the location-vs-magnitude regression -- residuals of 15-25pp, dwarfing every other effect in the model). Every row in `data/free_agents/*.txt`, by contrast, is a fresh market valuation event: the target season's actual salary was set by a deal signed directly off the prior season's production.

`build_contract_model_v3.py` builds its training set entirely from these files instead: for each free-agent signing, the prior season's 12-category vector + `EXPERIENCE_YRS`/`USG_PCT`/award flags (same feature set as v2) predict the *following* season's actual salary. Same XGBoost hyperparameters, same leave-one-season-out validation, just ~580-670 training rows per fold instead of v2's ~1,900 veteran rows -- and despite the much smaller sample, this measurably helped: free-agent-eval MAE dropped from v2's 3.88pp to 3.42pp before the minutes gate.

The `EXPERIENCE_YRS > 4` rookie-scale filter from v2 isn't applied to v3's training data -- free-agent signings are, by construction, market-negotiated deals rather than fixed rookie-scale ones (the rare exception being a 4th-year restricted free agent on the edge of that boundary, too small a group to need a separate filter here).

### Change 2: minutes-reliability gate

The user's critique after inspecting the row-level output directly: several models overpredict players who barely played -- a deep-bench or two-way player's 12-category vector can look statistically "average" in a tiny sample (or lean entirely on player.md's baseline substitution for play types they never got enough run to attempt), producing a plausible-looking prediction for someone who's actually a replacement-level roster afterthought.

Fix (`apply_minutes_gate.py`, imported directly into the v3 build): compute each player's trailing 3-year total minutes (that season plus up to 2 prior, same trailing-window convention as `build_playtype_vectors.py`'s experience decay -- no imputation for seasons outside `data/player_stats/`'s range). Below a 2,400-minute threshold (roughly a full healthy season, since this is a 3-year *total*), the prediction is scaled toward 0 by `sqrt(total_min / 2400)` -- a smooth curve, not a hard floor, at the user's explicit request: a player at the threshold keeps their full prediction, a player at a quarter of the threshold keeps half of it (`sqrt(0.25) = 0.5`), a player with ~0 minutes shrinks to ~0.

This dropped free-agent-eval MAE further, from 3.42pp to 3.31pp, without changing the worst single miss at all (still ~21pp) -- confirming this fixes the "lots of small errors on scrubs" problem specifically, not the separate contract-lag problem Change 1 targets.

### v3 results

Pooled across the 8 usable free-agent-eval seasons (2019-20 through 2026-27, n=754 -- see `free_agent_evaluation.md` for the full per-year breakdown). The v2/before-gate rows below are frozen from the original build exercise (not re-run since); the final row is the current production number, re-run most recently after the perimeter-defense rework (self-computed Defensive Win Shares -- see defense.md):

| Version | MAE | RMSE | Corr |
|---|---|---|---|
| v2 (full historical pool) | 3.88pp | 5.29pp | 0.697 |
| v3, before minutes gate | 3.42pp | 4.79pp | 0.699 |
| v3, final (with minutes gate), pre-DWS-rework | 3.31pp | 4.73pp | 0.710 |
| **v3, final (with minutes gate), current** | **3.13pp** | **4.69pp** | **0.719** |

### A detour that's worth recording: the non-black-box formula attempt

Before landing on v3, the user asked for a non-supervised-regression alternative: a hand-specified linear formula (`value = intercept + w1*magnitude_z + w2*usage_z + w3*location_z`, where magnitude = raw-vector Euclidean norm and location = PC1, the big/screener-vs-perimeter-creator axis from the vector-space map). This was built, tested with several feature subsets (dropping magnitude, dropping usage, usage-only -- all tested directly rather than assumed, and all performed *worse* than keeping all three), and eventually retrained on the free-agent list the same way v3 was (MAE 4.04pp, an improvement over its own full-pool version's 5.11pp, but still behind XGBoost's richer feature set). Once XGBoost was retrained on the free-agent list too and won outright, the formula's scripts and outputs were deleted per the versioning policy -- but the exercise directly validated two things v3 relies on: (1) location is a real, independent, additive effect on pay (see the magnitude/location regression -- location's coefficient stayed significant even after controlling for usage and awards), and (2) training on free-agent signings instead of the full pool helps regardless of which model sits on top of that data.

---

## Known Limitations

- **Still overpredicts some below-market/"discount" deals**: the model only sees on-court production, not contract-negotiation dynamics -- v3's worst misses (see `free_agent_evaluation.md`) are still players who took a hometown discount or a prove-it deal. Neither the free-agent-training change nor the minutes gate targets this directly (confirmed: the worst single miss barely moved, 20.1pp before the minutes gate, 21.5pp after -- the gate fixes small errors on low-minutes players, not this).
- **Much smaller training population per fold** (~580-670 free-agent rows vs. v2's ~1,900 veteran rows) -- helped in every check run so far, but is inherently more exposed to a single unusual season swinging the fit than v2's larger pool was.
- **`ALL_NBA_SELECTION` is data-starved**: only 120 of 4,466 rows in the full pool are `True` (a similarly small share of the free-agent-only population), and its v2 importance (0.007) suggested the model can't extract much signal from it independent of `ALL_STAR_SELECTION`/`USG_PCT`. Not re-measured for v3 specifically.
- **No cap-max clipping**: XGBoost's leaf-averaging keeps predictions bounded by training targets in practice, but this hasn't been explicitly verified against the CBA's max-salary-by-experience rules.
- **2,400-minute threshold and the sqrt shrinkage curve are both hand-chosen**, not cross-validated -- a sweep over threshold/curve-shape combinations (mirroring the similarity model's bandwidth sweep) hasn't been done.
- **v1's script is unrecoverable**: its exact preprocessing (e.g. how `USG_PCT` was computed, whether awards were fetched via `_fetch_awards.py` or some other path) remains inferred from the output CSV, not confirmed from source.

---

## Changelog

- v1: built a first supervised regression (script since lost) predicting `SALARY_PCT_OF_CAP` from `EXPERIENCE_YRS`/`ROOKIE_CONTRACT`/`USG_PCT`, output to `predicted_vs_actual_all_seasons.csv`. Correlation 0.70 overall / 0.77 non-rookie-only. Diagnosed negative predictions (unclipped linear/ridge fit) as a data-quality flag.
- Started (not completed) `_fetch_awards.py` to pull All-Star/All-NBA history via `PlayerAwards`; no raw output file was ever found in the repo, but award columns (`ALL_STAR_SELECTION`, `ALL_NBA_TEAM_NUMBER`, `ALL_NBA_SELECTION`) are nonetheless present and fully populated in `predicted_vs_actual_all_seasons.csv`, suggesting the fetch was completed and merged by some other path before that script was removed.
- **Rebuilt as v2** (`build_contract_model.py`), at the user's request to act on the standing "pivot to supervised regression, LOSO-validated" recommendation and to add award flags as features: full 12-category vector + `EXPERIENCE_YRS` + `USG_PCT` + `ALL_STAR_SELECTION` + `ALL_NBA_SELECTION`, XGBoost, training restricted to `EXPERIENCE_YRS > 4`, validated via LOSO CV across `STANDARD_SEASONS`. Verified all auxiliary fields join 1:1 by `(PLAYER_ID, SEASON)` against the vector pool before reusing them. Result: LOSO correlation 0.763 (all rows) / 0.823 (veteran-only), a clear improvement over both v1 and the similarity model's kernel regression. Feature importances show `USG_PCT` and `ALL_STAR_SELECTION` dominating (>0.5 combined).
- Evaluated against the real 2025 free-agent class alongside the similarity model (see `free_agent_evaluation.md`): v2 MAE 2.99pp / corr 0.75 vs. the similarity model's 4.49pp / 0.55 -- v2 adopted as the primary recommended pipeline. Its remaining errors are concentrated on players who took below-market deals, a different failure mode than the similarity model's star-underprediction problem.
- **Cleaned up `data/salary_prediction/`** at the user's request, so the folder holds only the current pipeline's data: extracted v1's input columns into `player_season_features.csv`, deleted the v1 output (`predicted_vs_actual_all_seasons.csv`) now that v2 no longer depends on it, and trimmed `contract_model_v2_predictions.csv` itself down to identity columns plus actual vs. predicted (dropping the 12-category vector and other training features from the export, since those are inputs, not results). Re-ran both `build_contract_model.py` and the free-agent eval after the change and confirmed identical LOSO and free-agent-eval results.
- **Fixed an in-sample data-leakage bug** in the exported predictions (caught when the user asked "are you sure these are correct?"): `contract_model_v2_predictions.csv` had been populated from `fit_full_model()`, trained on every season at once, so most players' own true salary was a training label for their own "prediction." Added `out_of_fold_predictions()`, which predicts each season using only the LOSO fold that excluded it -- this is now what's exported. Corrected free-agent-eval numbers: MAE 3.40pp / corr 0.679 (previously an inflated 2.99pp / 0.753).
- **Multi-year free-agent validation**: once `data/free_agents/` gained 7 more yearly files (`18-19.txt` through `24-25.txt`, alongside the original `25-26.txt`), the eval harness (`evaluate_free_agents.py`, see `free_agent_evaluation.md`) was rebuilt to pool multiple years instead of just one.
- **Fixed an off-by-one introduced by that generalization**: each free-agent file is named for the season these players *just played*, not the season their new contract covers (e.g. `19-20.txt`'s players signed contracts for 2020-21, not 2019-20) -- verified by matching AAV against the following season's real salary (82-100% close-match rate) vs. the filename's own season (21-38%). The original single-season script (this doc's "MAE 2.99pp/3.40pp" entries above) had this right by construction, since its file paths were hardcoded (2024-25 vectors, 2025-26 salary) rather than derived from a filename -- the bug was specific to the new multi-year harness's `season_pair()`, which mistakenly treated each filename as the target season. Once corrected, all 8 files became usable (`18-19.txt` needs 2018-19 vectors, which exist -- not 2017-18, which don't) across target seasons 2019-20 through 2026-27, 754 matched rows: v2 now beats the similarity model's MAE and correlation in **every single one of the 8 seasons** (pooled: 3.88pp MAE / 0.697 corr vs. 5.05pp / 0.583), the strongest evidence yet for the "v2 is better" conclusion.
- **Built and tested the non-black-box formula detour** (magnitude + usage + location, see "A detour that's worth recording" above), at the user's request for a non-supervised-regression alternative. Tested several feature subsets directly rather than assuming which mattered -- dropping magnitude or usage both made every metric worse, confirming the redundancy the user suspected between them was partial, not total.
- **Discovered training on the free-agent list directly beats training on the full historical pool**, at the user's suggestion ("try this too... those are the more accurate contract perceptions of players"): rebuilt the formula with leave-one-season-out CV on just the free-agent signings instead of the full veteran pool -- MAE dropped from 5.11pp to 4.04pp despite the much smaller training population, confirming the contract-lag-noise hypothesis. Applied the same change to the XGBoost model (**this became v3**): MAE dropped from v2's 3.88pp to 3.42pp.
- **Added the minutes-reliability gate** (`apply_minutes_gate.py`), at the user's request, after they flagged that predictions were "all over the place" and specifically that low-minutes players were getting overpredicted: computes trailing-3-year total minutes and shrinks predictions below a threshold toward 0. First built as a hard floor at 1,000 minutes; the user then asked for a 2,400-minute threshold with a smooth `sqrt(total_min/threshold)` curve instead of a hard zero, which is what shipped. Folded directly into v3's build script as a final step. Dropped MAE further to 3.31pp.
- **Deleted v2 and the formula detour's scripts and outputs** once v3 (XGBoost + free-agent training + minutes gate) beat both on every metric, per this doc's versioning policy: `build_contract_model.py`, `build_value_formula.py`, `build_freeagent_regression.py`, `build_freeagent_contract_model.py` (folded into `build_contract_model_v3.py`), and their corresponding CSVs. `data/salary_prediction/` now holds only `contract_model_v3_predictions.csv`, `free_agent_eval_multi_year.csv`, and `player_season_features.csv` (the shared input data). `evaluate_free_agents.py` updated to point at v3 and re-run to confirm identical results post-cleanup.
- **Re-ran v3 after the perimeter-defense rework** (self-computed Defensive Win Shares via Dean Oliver's formula, replacing an on-court-DRTG-based approach -- see `defense.md`), at the user's request ("recalculate similarity + salary predictions"): no changes to `build_contract_model_v3.py` itself, since it consumes the 12-category vector (including `PERIMETER_DEFENSE`) as-is from `data/player_vectors_json/`, which had already been regenerated. Pooled free-agent-eval MAE improved from 3.31pp to 3.13pp (RMSE 4.73pp -> 4.69pp, corr 0.710 -> 0.719), consistent with the defense rework producing a cleaner, less noisy perimeter-defense signal.
