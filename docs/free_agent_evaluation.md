# Free Agent Evaluation (`evaluate_free_agents.py`)

Covers the script that checks both contract-prediction pipelines -- `similarity_model.md`'s kernel regression and `contract_model.md`'s v3 XGBoost model -- against real money: every real NBA free-agent class on file, not just one offseason. This is the out-of-sample check that motivated every iteration of `contract_model.md`'s rebuild (v2, then v3) and is what settled which pipeline is currently recommended.

Output: `data/salary_prediction/free_agent_eval_multi_year.csv`.

---

## Inputs

- **`data/free_agents/<YY-YY>.txt`**: one copy/pasted free-agent tracker per offseason, named by **the season these free agents just finished playing**, NOT the season their new contract covers -- e.g. `25-26.txt` lists players who just played 2025-26 and signed contracts covering **2026-27**. This was not obvious from the filename alone and was verified empirically (see `season_pair()`'s docstring and the Changelog entry below): for every file, matching AAV against the *following* season's actual salary gives an 82-100% close-match rate, vs. only 21-38% when matching against the filename's own season. Currently covers `18-19` through `25-26` (8 files, all usable -- see Changelog for why `18-19.txt` was wrongly thought unusable before this fix). Each row has position, contract years, total value, and AAV (average annual value = total / years).
- Predictions are drawn from the **named season's** vectors/predictions (e.g. `25-26.txt` uses 2025-26) -- the data that would actually have been available at signing time -- compared against actual salary the *following* season.
- **`data/salaries/salaries_<target_season>.csv`**: each player's *actual* salary for the target season (their real first-year cap hit) -- used as ground truth in preference to AAV, since they can differ substantially for back/front-loaded deals (e.g. Hassan Whiteside's 2020-21 Kings deal was exactly his `19-20.txt` AAV of $2,320,044, distinct from his unrelated $27M 2019-20 Trail Blazers salary from an earlier contract -- the mismatch that surfaced this whole bug).
- Player names are matched across all sources via accent-stripped, suffix-stripped, case-insensitive normalization (`normalize_name()`).

---

## Results (most recent run, pooled across all 8 seasons)

| | Similarity/kernel model | Contract model v3 |
|---|---|---|
| n | 754 | 754 |
| MAE | 4.93pp | **3.13pp** |
| RMSE | 6.33pp | **4.69pp** |
| Bias | -1.21pp | +0.04pp |
| Corr | 0.616 | **0.719** |

The contract model wins on every metric, and -- unlike the single-year check this replaced -- **the margin holds consistently across all 8 independent offseasons** (targets 2019-20 through 2026-27), not just one lucky/unlucky year:

| Target season | Similarity MAE / corr | Contract v3 MAE / corr |
|---|---|---|
| 2019-20 | 5.77pp / 0.699 | 3.35pp / 0.826 |
| 2020-21 | 4.54pp / 0.538 | 3.09pp / 0.749 |
| 2021-22 | 4.89pp / 0.606 | 3.58pp / 0.718 |
| 2022-23 | 4.83pp / 0.645 | 2.82pp / 0.785 |
| 2023-24 | 5.13pp / 0.637 | 3.37pp / 0.627 |
| 2024-25 | 5.28pp / 0.686 | 2.92pp / 0.776 |
| 2025-26 | 4.33pp / 0.617 | 2.65pp / 0.747 |
| 2026-27 | 4.09pp / 0.440 | 3.00pp / 0.499 |

Contract v3 beats the similarity model's MAE and correlation in all 8 seasons -- a much stronger basis for "which model is better" than the original single-season check.

---

## Recommendation

Use `contract_model.md`'s v3 pipeline as the primary contract predictor going forward; the similarity model's Step 5 top-10 comps ("who does this player resemble") remain useful as a display feature independent of the contract-prediction question.

---

## Known Limitations

- **Actual-salary coverage gap**: roughly 70-90% of each year's free-agent list has a confirmed actual salary on file (`salaries_<season>.csv` doesn't cover 2-way/minimum-deal fringe players), so each season's n is smaller than its full free-agent list.
- **AAV vs. actual salary**: AAV is reported in the free-agent files for reference, but scoring uses actual salary only -- a model tuned against AAV instead could show different results, especially for backloaded deals.
- **2026-27 is the newest, thinnest target season** (n=67) and has the lowest correlation for both models (0.440 similarity / 0.499 v3) -- worth revisiting once more of that season's actual salaries are confirmed on file.
- **v3 trains directly on these same free-agent files** (leave-one-season-out, so no season's own signings ever inform its own fold's prediction) -- this evaluation and v3's own validation now share a data source, which is a tighter loop than v2's had (v2 trained on the separate historical player-season pool, so this eval was a fully independent check on it). The per-year LOSO numbers reported in `contract_model.md` are the same numbers as this table for v3, not an independent confirmation of them.

---

## Changelog

- Built the eval against an early free-agent list (85 players, mostly top-of-market names, single season): similarity model MAE 4.46pp, corr 0.34 -- confirmed the archetype-bias failure mode on Trae Young, Austin Reaves, Walker Kessler.
- Re-run against an expanded/replaced free-agent list (127 players, single season, broader veteran/minimum-deal coverage) with actual salary (not just AAV) as ground truth, after discovering `salaries_2025-26.csv` already had real first-year cap hits for these signings.
- Added the v1 supervised-regression model to the single-season comparison: essentially tied with the similarity model (MAE 4.46pp vs. 4.49pp, corr 0.539 vs. 0.549) -- different failure mode (overpredicts discount vets) but no clear win.
- Swapped in the `contract_model.md` v2 rebuild once built (still single-season): MAE dropped to 2.99pp, corr rose to 0.753 -- adopted as the current recommendation.
- **Caught and fixed a data-leakage bug**, prompted by the user asking "are you sure these are correct?" about the exported predictions: the contract model's predictions had been generated in-sample (trained on every season including the one being predicted). Switched to genuine out-of-fold predictions (see `contract_model.md`'s "Out-of-fold exports" section) -- corrected single-season numbers were MAE 3.40pp / corr 0.679.
- **Rebuilt as a multi-year harness** (`evaluate_free_agents.py`, replacing `evaluate_2025_free_agents.py`), at the user's request, once 7 additional yearly free-agent files (`18-19.txt` through `24-25.txt`) were added to `data/free_agents/`: generalized the season/file-matching logic to derive both the prediction season and the target season from each file's name, pooling 7 "usable" years (2019-20 through 2025-26, 677 total matched rows -- `18-19.txt` was believed unusable at this point) into one RMSE/MAE/corr comparison plus a per-year breakdown. Result at the time: contract v2 (MAE 4.18pp, corr 0.680) beat the similarity model (MAE 5.17pp, corr 0.555).
- **Caught and fixed a systematic off-by-one in the multi-year harness's season mapping**, prompted by the user asking to double check the actual-salary-percentage-of-cap calculation: the new `season_pair()` (written when generalizing to multiple years) had been treating each filename as the *target* season (e.g. `19-20.txt` -> predict from 2018-19, compare to actual 2019-20 salary), but every file's AAV actually matches the *following* season's real salary far better (82-100% close-match rate) than the filename's own season (21-38%) -- confirmed concretely on Hassan Whiteside, whose `19-20.txt` row ($2,320,044 AAV) is his real 2020-21 Kings deal, not his unrelated $27M 2019-20 Trail Blazers salary from an earlier contract. The single-season numbers earlier in this Changelog (2.99pp/3.40pp) were unaffected -- that script's paths were hardcoded correctly (2024-25 vectors, 2025-26 salary), not derived from a filename; the bug was introduced only by the new multi-year generalization above. Fixed `season_pair()` to treat the filename as the *previous* (predicting-from) season and the target as filename+1; this also means `18-19.txt` is no longer unusable (it needs 2018-19 vectors, which exist, not 2017-18) -- all 8 files are now used, covering target seasons 2019-20 through 2026-27 (754 matched rows). Corrected pooled result: contract v2 MAE 3.88pp / corr 0.697 vs. similarity MAE 5.05pp / corr 0.583 -- contract v2 now wins in literally every one of the 8 seasons, not just most of them.
- Added a third model (a hand-specified magnitude+usage+location formula) to this harness temporarily while exploring a non-black-box alternative; removed once it was outperformed and superseded by v3 (see `contract_model.md`'s Changelog for the full detour).
- **Swapped in contract model v3** once it was built (trained directly on free-agent signings, plus the minutes-reliability gate) -- pooled result improved to MAE 3.31pp / corr 0.710, still beating the similarity model in every one of the 8 seasons. `evaluate_free_agents.py` updated to read `contract_model_v3_predictions.csv` (joined on `PREV_SEASON` rather than `SEASON`, matching v3's file shape) and re-run to confirm.
- **Re-run after the perimeter-defense rework** (self-computed Defensive Win Shares, replacing an on-court-DRTG-based approach -- see `defense.md`), at the user's request: regenerated `data/player_vectors_json/`, then `build_similarity_model.py`, `build_contract_model_v3.py`, and this harness in sequence. Both models improved -- similarity MAE 5.05pp -> 4.93pp / corr 0.583 -> 0.616; contract v3 MAE 3.31pp -> 3.13pp / corr 0.710 -> 0.719 -- and contract v3 now beats the similarity model's correlation in all 8 seasons rather than 7 of 8 (2026-27, the prior exception, flipped: 0.440 vs. 0.499).
