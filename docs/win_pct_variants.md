# Win-% Model Variants (`data/win_pct_model/`)

Extends `team.md`'s "Does `NEED_VECTOR` predict winning?" section with three follow-up questions, all working from the same 240 team-seasons (30 teams x `STANDARD_SEASONS`) and the same LOSO-across-`STANDARD_SEASONS` discipline used everywhere else in the repo:

1. Can win% be predicted directly from a roster's aggregate player vectors, skipping the `NEED_VECTOR` step entirely? (`evaluate_direct_roster_win_pct.py`)
2. Where does the `NEED_VECTOR` model's remaining error come from? (`analyze_win_pct_bias.py`)
3. Does inserting a predicted OFF/DEF rating step in between close that gap? (`analyze_player_vectors_vs_ratings.py`, `compare_models_vectors_ratings_wins.py`, `predict_rating_detour_win_pct.py`)

All four win%-prediction variants compared, LOSO RMSE on `W_PCT`:

| Model | RMSE | Uses real (in-season) `NEED_VECTOR`? |
|---|---|---|
| Naive baseline (train-mean) | 0.1480 | -- |
| Direct roster features -> win% | 0.0946 | No |
| **Rating-detour (roster -> rating -> win%)** | **0.0777** | No |
| Roster -> reconstructed `NEED_VECTOR` -> win% (`team.md`'s blend) | 0.0915 | No |
| True team-stats `NEED_VECTOR` -> win% (`team.md`'s Ridge) | 0.0630 | Yes |

The true-`NEED_VECTOR` Ridge remains the ceiling, but it isn't available before a season is played -- it's a backward-looking validation of `team.md`'s pipeline, not something a pre-season roster projection can use. Of the three genuinely pre-season-usable options (direct, `NEED_VECTOR`-reconstructed blend, rating-detour), rating-detour wins, which is why `predict_2026_27_team_profile.py` and the free-agency simulation scripts (`free_agency_2026.md`) are split between the `NEED_VECTOR`-blend and rating-detour pipelines rather than one single production model -- see each doc's script for which it uses.

---

## Direct roster -> win% (`evaluate_direct_roster_win_pct.py`)

Skips `NEED_VECTOR` as an intermediate target: `RidgeCV` directly on `predict_team_need_from_roster.py`'s 25 roster-aggregate features (`TOTAL_MIN` + minutes-weighted mean/std of each of the 12 player-vector categories) predicting `W_PCT`.

LOSO RMSE: naive 0.1480 -> **0.0946**. Refit on all 8 seasons and applied to 2026-27 rosters (`W_PCT_CLIP = (0.05, 0.95)`).

Output: `data/win_pct_model/direct_roster_win_pct_results.json` -- `feature_columns`, `rmse`, `historical` (240 rows: season/team/actual_wins/pred_wins), `projection_2026_27` (30 rows). 2026-27 top predicted win totals: OKC 69.3, GSW 55.6, HOU 54.5, SAS 53.4, CLE 52.7.

---

## Bias diagnostic (`analyze_win_pct_bias.py`)

Runs against `team.md`'s existing production Ridge win-pct model (the true-`NEED_VECTOR` one, not a new fit), joining its predicted-vs-actual output against `NEED_VECTOR` plus raw team stats (`DEF_RATING`, a derived `OFF_RATING`, `NET_RATING`, `PACE`, `OREB_PCT`, `DREB_PCT`, `RIM_DIFF_PCT`) and computing `ERROR = PRED_WINS - ACTUAL_WINS`.

Checks run: top/bottom-15 most over/under-predicted team-seasons; team-level mean error; Pearson correlation of every joined feature against `ERROR`; a top/bottom-quartile group comparison; error broken out by win-pct tercile (regression-to-mean check) and by pace tercile; error by season.

**Key finding**: `ERROR` correlates with `OFF_RATING`/`NET_RATING` (r ~= -0.32 / -0.33) even though neither is an input to the win-pct model -- the model's residual is picking up real offensive-rating information that `NEED_VECTOR`'s 12 categories don't fully capture. This is the finding that motivated the rating-detour line of work below.

Output: `data/win_pct_model/win_pct_bias_analysis.csv` (240 rows, curated columns) and `win_pct_bias_analysis_full.csv` (240 rows, all intermediate columns including `QUALITY_TERCILE`/`PACE_TERCILE`).

---

## Can a roster reconstruct OFF/DEF rating? (`analyze_player_vectors_vs_ratings.py`)

Follow-up to the bias finding above: since `OFF_RATING`/`DEF_RATING` predict win-pct error, can a roster's aggregate vectors (the same 25 features used throughout) predict those ratings directly?

**Stage 1** -- `RidgeCV`, roster features -> `OFF_RATING` and separately -> `DEF_RATING`, LOSO, reporting RMSE and Pearson correlation(actual, predicted) plus the top-10 |coefficient| features for each target.

**Stage 2** -- augments the 25-feature direct win% model with each fold's out-of-fold predicted `OFF_RATING`/`DEF_RATING`/`NET_RATING` (28 features total) and re-measures win% RMSE, checking whether this narrows the gap to the true-`NEED_VECTOR` Ridge (0.0630).

Output: `data/win_pct_model/player_vectors_vs_ratings.csv` (240 rows: `SEASON, TEAM, OFF_RATING, PRED_OFF_RATING, ERR_OFF, DEF_RATING, PRED_DEF_RATING, ERR_DEF, W_PCT`).

## Model-family search (`compare_models_vectors_ratings_wins.py`)

Prints-only, no persisted output. Compares 5 regressor families -- Ridge, ElasticNet, PLS, XGBoost, RandomForest (`RF_PARAMS`: `n_estimators=300, max_depth=5, min_samples_leaf=3`) -- at both stages of the rating-detour pipeline (roster -> rating, and 25-features-plus-ratings -> win%), under the same LOSO discipline, to pick the winning combination that became `predict_rating_detour_win_pct.py` below.

## Production rating-detour pipeline (`predict_rating_detour_win_pct.py`)

The winning combination from the comparison above: `RidgeCV` roster-vectors -> OFF/DEF rating (stage 1), then `RidgeCV`(25 roster features + predicted OFF/DEF/NET rating) -> win% (stage 2).

LOSO results: `stage1_off` RMSE 2.637 rating-points, `stage1_def` RMSE 2.431 rating-points, **`stage2_win` RMSE 0.0777** -- beats the direct 25-feature model (0.0946) and the `NEED_VECTOR`-reconstruction blend (0.0915), though it's still short of the true team-stats `NEED_VECTOR` Ridge (0.0630, which isn't available before a season is played).

Refits on all 8 seasons and applies to 2026-27 rosters. Output: `data/win_pct_model/rating_detour_win_pct_results.json` -- `feature_columns` (28), `rmse`, `historical` (240 rows), `projection_2026_27` (30 rows, including `pred_off_rating`/`pred_def_rating`/`pred_net_rating`). 2026-27 top: OKC 71.2 wins (`OFF 119.17` / `DEF 107.33` / `NET 11.84`), GSW 57.0, HOU 55.8, SAS 54.2, ORL 52.9.

This is the model `exhaustive_free_agent_combos.py` and `simulate_free_agency_impact.py` (see `free_agency_2026.md`) reuse for scoring free-agent combinations, since it's the best-performing pre-season-usable variant found.

---

## Known Limitations

- **`compare_models_vectors_ratings_wins.py` has no persisted output** -- the exact per-model RMSE table that justified picking Ridge over XGBoost/RandomForest/PLS at each stage exists only as console output from whatever run produced `predict_rating_detour_win_pct.py`'s design, not as a committed file.
- **Rating-detour still trails the true-`NEED_VECTOR` ceiling by ~23%** (0.0777 vs. 0.0630 RMSE) -- there remains real information in a season's actual team stats that neither roster vectors nor a predicted rating step recovers.
- **`analyze_win_pct_bias.py`'s correlations are diagnostic only, not causal** -- `OFF_RATING`/`NET_RATING` correlating with prediction error doesn't establish that adding them fixes the error for the right reason rather than just adding a fitted proxy for the same 12 categories.
