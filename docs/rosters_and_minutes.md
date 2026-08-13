# Rosters & Minutes Prediction

Two things needed before any 2026-27 forward-looking pipeline (`team_2026_27_projection.md`, `free_agency_2026.md`) can run: an actual roster to score, and a minutes estimate for each player on it (the player-vector pipeline is built from played minutes, which 2026-27 doesn't have yet).

---

## Roster pull (`pull_rosters.py`)

Pulls full team rosters via `nba_api`'s `CommonTeamRoster` endpoint for all 30 teams, for a given `--season` (default `2026-27`), with a 0.6s delay between calls. Unlike the free-agent `.txt` dumps in `data/free_agents/` (guaranteed-money signings only, ~432 players), this captures every roster body -- two-way, camp, non-guaranteed -- including a `HOW_ACQUIRED` column (e.g. `"Signed on 07/22/26"`, `"Traded from GSW on 02/05/26"`, `"#4 Pick in 2026 Draft"`, `"Draft Rights Traded from ..."`) that downstream scripts parse via regex to classify transaction type and date. This `HOW_ACQUIRED` field is what `free_agency_2026.md`'s scripts use to identify the actual 2026 free-agent signing class (`FA_SIGNED_PATTERN = "^Signed on 07/\d{2}/26$"`) and what `analyze_rookie_draft_capital_vs_wins.py` parses for draft-pick numbers.

Output: `data/rosters/rosters_<season>.csv` -- columns `TEAM, PLAYER_NAME, PLAYER_ID, POSITION, HEIGHT, WEIGHT, AGE, EXP, SCHOOL, HOW_ACQUIRED`. On file: `data/rosters/rosters_2026-27.csv`, 587 rows across all 30 teams (`EXP == "R"` marks rookies).

---

## Minutes model, validated (`predict_minutes_from_roster_context.py`)

Predicts a player's minutes-per-game from their own 12-category skill vector plus a fixed-size, minutes-weighted summary of their teammates' vectors -- the same roster-to-fixed-vector convention `predict_team_need_from_roster.py` (`team.md`) uses at team level, applied here per player.

### Inputs

- `data/player_vectors_json/player_vectors_<season>.json`: the 12-category vector (`VECTOR_CATEGORIES`, same as `NEED_CATEGORIES` elsewhere).
- `data/player_stats/player_stats_<season>.csv`: `GP`, `MIN` (rows for players traded mid-season are summed via groupby first).

### Features (27 per row)

Own 12 categories (`OWN_<cat>`) + teammates' minutes-weighted mean and std of each category (`TEAMMATE_<cat>_WMEAN` / `_WSTD`, self excluded) + `TEAMMATE_COUNT` + `TEAMMATE_TOTAL_MIN`.

### Filtering & target

`MIN_TOTAL_MINUTES = 1000` -- a season-total minutes floor (not a `GP` floor, to avoid garbage-time-heavy seasons polluting the signal), applied both to prediction targets and to which players count as teammate context. Target: `TARGET_MPG = MIN / GP`.

### Model & validation

Single `XGBRegressor` (`max_depth=3, reg_alpha=0.1, reg_lambda=1.0, n_estimators=200, random_state=0` -- the same `XGB_PARAMS` convention reused throughout the newer scripts), LOSO across `STANDARD_SEASONS` against a naive train-mean baseline.

Output: prints pooled and per-season LOSO RMSE/MAE; writes `data/minutes_prediction/predicted_vs_actual_mpg.png`, a predicted-vs-actual scatter (out-of-fold, pooled) with the 8 worst-error points annotated by name/season.

`build_dataset()`, `feature_columns()`, `VECTOR_CATEGORIES`, `XGB_PARAMS`, and `MIN_TOTAL_MINUTES` are imported directly by `predict_2026_27_minutes.py` below, so the forward-inference model is the same fitted procedure as the validated one, not a re-derivation.

---

## Minutes model, forward inference (`predict_2026_27_minutes.py`)

Scores the actual `data/rosters/rosters_2026-27.csv` (587 players) -- a genuine prediction, not a backtest, since 2026-27 has no ground-truth vectors or minutes yet.

### Sourcing inputs for an unplayed season

- Each player gets their most-recent `STANDARD_SEASONS` vector (`build_latest_vectors()`) and most-recent season-total minutes as the teammate-weighting proxy (`build_latest_minutes()`).
- **Rookies with no historical vector** get a *replacement-level vector* (`build_replacement_vector()`) -- the mean skill vector across real deep-bench player-seasons in the `REPLACEMENT_MIN_RANGE = (200, 800)` minutes band -- rather than being dropped, flagged via the output's `REASON` column. Only a team with zero rotation-caliber historical players at all is left genuinely unscorable.

### Training difference from the validated model

Trains with `target_min_minutes=0` (every player-season, not just >=1000-minute rows), since scoring a full roster means predicting bench and two-way minutes too. `context_min_minutes` stays at `MIN_TOTAL_MINUTES` (rotation-only teammate context) both in training and at scoring time.

### Calibration: quantile mapping

`quantile_map_to_target()` -- rank-based quantile mapping of raw XGBoost predictions onto the shape of the real historical `TARGET_MPG` distribution (the k-th smallest raw prediction maps to the k-th percentile of the real distribution). This replaced an earlier fixed value-interpolation curve that under-corrected: 2026-27's raw predictions run ~1.4 MPG higher on average than the historical rows the earlier curve was fit against (verified: mean 20.8 vs. 19.4).

### Calibration: team-total rescale

`rescale_to_team_total()` -- a power-law reallocation (`weight = value.clip(0.1) ** REALLOCATION_EXPONENT`, exponent **1.6**, chosen empirically) forces each team's calibrated MPGs to sum to exactly `TEAM_TOTAL_MPG = 240` (5 x 48). A flat proportional rescale would discount stars and scrubs equally -- it would have cut Jokic's ~35.8 calibrated MPG down to ~23.1. 1.6 was chosen as the highest exponent that keeps every roster's single highest rescaled player under ~40 MPG, while lifting the league-wide top-5 average from 22.7 to 26.8 MPG. A sigmoid-based alternative was tried and reverted: it clustered Denver's top 4 players within ~2.6 MPG of each other despite a real ~10 MPG gap between them.

### Output

`data/2026_27_projections/multi_year_minutes_predictions.csv` -- 587 rows: `TEAM, PLAYER_NAME, PLAYER_ID, PRED_MPG, PRED_MPG_UNSCALED, VECTOR_SEASON, MIN_SEASON, REASON`. Sample: CJ McCollum (ATL), `PRED_MPG=24.8`, `PRED_MPG_UNSCALED=28.7`, `VECTOR_SEASON=2025-26`.

Also prints a team-total sanity check (min and max across teams should both land at 240) and lists every scored player whose `VECTOR_SEASON` is stale (not 2025-26) -- i.e. anyone who didn't play last season and is being projected off older data.

This CSV's `PRED_MPG` is one of the two inputs (with roster composition) `predict_2026_27_team_profile.py` uses to build each 2026-27 team's minutes-weighted feature vector -- see `team_2026_27_projection.md`.

---

## Known Limitations

- **Quantile mapping and the 1.6 reallocation exponent are both hand-tuned against the 2026-27 roster's specific raw-prediction distribution**, not cross-validated the way the underlying XGBoost model is -- a different season's raw-prediction shift (relative to the historical training distribution) could need a different exponent, and there's no automated re-tuning step.
- **Replacement-level vectors for rookies are a single fixed archetype** (the mean deep-bench player in a 200-800 minute band), not position- or role-aware -- a rookie point guard and a rookie center get the same starting-point vector shape before the minutes model adjusts for teammate context.
- **`REASON` flags stale-vector players but nothing downstream currently filters or discounts on it** -- a team's minutes-weighted roster feature vector treats a `VECTOR_SEASON=2023-24` player identically to a `2025-26` one.
