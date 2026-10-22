# Free-Agency Simulation & Combo Search

Everything downstream of `team_2026_27_projection.md`'s roster/win% pipeline that's specifically about free agency: identifying who actually is a free agent, historically backtesting the whole "swap the roster, re-project" idea against real past offseasons, then running it forward on the real 2026 free-agent class -- both as an exhaustive combinatorial search and as a per-player team-fit ranking.

---

## Identifying a free-agent class

### From historical signing trackers (`load_free_agent_class.py`)

Library module (also runnable standalone for a per-season count print). Parses `data/free_agents/<season>.txt` -- the same real historical free-agent-class dumps `free_agent_evaluation.md` and `contract_model.md` use for salary prediction -- filtering to rows whose status line reads `"Official"` or `"Agreed"`. Exposes `load_free_agent_names(season)` and `normalize_name()` (accent/punctuation-stripping, shared across the historical free-agency scripts for cross-source name matching).

Framed in its docstring as superseding an earlier roster-diff "movers" heuristic: matching against the real free-agent tracker captures same-team re-signings (a player who stays with their team is still a free agent) and excludes mid-season trades by construction, since a real free agent by definition signed a new contract in that window.

### From 2026-27 rosters directly (`find_midseason_trades.py` -- superseded)

Pulls `CommonTeamRoster` for all 30 teams across the 7 `season_next` values and classifies each `HOW_ACQUIRED` entry's month as offseason (Jul-Sep) vs. mid-season (Oct-Jun) via regex date extraction, deliberately not hardcoding exact game-calendar boundaries (the league has had bubble/delayed seasons). The script's own docstring calls it superseded -- "no longer needed here" -- once `load_free_agent_class.py`'s tracker-based approach was built. Its intended output, `data/free_agency/midseason_trades_by_season.json`, is not present on disk.

### The 2026 free-agent class specifically

For the current (2026-27) class, no historical tracker exists yet, so every 2026 script instead uses `pull_rosters.py`'s `HOW_ACQUIRED` field directly: a free agent is any 2026-27 roster player matching `FA_SIGNED_PATTERN = "^Signed on 07/\d{2}/26$"` -- the July 2026 signing period, chosen because the roster data shows a clean gap with zero June 2026 signings. `simulate_free_agency_impact.py` writes this pool out to `data/free_agency/free_agency_pool_2026.csv` (62 rows: `TEAM, PLAYER_NAME, PLAYER_ID, HOW_ACQUIRED`).

### Multi-year contract parsing (`build_multi_year_contracts.py` -- inputs missing)

A standalone utility that would parse `26-27.txt` through `30-31.txt` (5 seasons, repo root, same rank/name/team-pos/salary scrape format as the `parse-nba-salary-list` skill) into one wide CSV (`multi_year_contracts.csv`, also repo root) -- one row per player, one salary column per season, blank where a player has no guaranteed deal recorded, `TEAM` taken from their earliest-appearing season. **None of the 5 source `.txt` files nor the output CSV exist in the repo at present**, so this script can't currently be run or its output verified.

---

## Historical backtesting

Two scripts validate the "swap a roster's free agents, re-project" methodology against the 7 real historical offseason cycles (Summer 2019 through Summer 2025) before trusting it on the 2026 class.

### Building the historical (team, cycle) roster deltas (`historical_free_agency_fit.py`)

For each `season -> season_next` transition: `PRE_FA(team)` = that team's real season-`S` roster minus every player name-matched to that summer's real free-agent class (`load_free_agent_class.py`); the real destination is wherever that name appears on the season-`S_next` roster. This reconstructs, purely from historical rosters plus the free-agent tracker, exactly what each team's roster looked like right before free agency opened that summer -- the historical analog of `team_2026_27_projection.md`'s 2026-27 rosters.

Reuses `exhaustive_free_agent_combos_need.py`'s linearized PLS `fit_need_pls()` and its win-contribution `weights` (see below) to score, per (mover, candidate-team) pair, the same weighted %-need-reduction metric: `((need_before - need_after) / max(1 - need_before, DENOM_FLOOR)) @ weights`. Only single-player (k=1) marginal scoring -- historical free-agent pools (113-175+ players per cycle) are too large for the exhaustive combinatorial search used on the much smaller 2026 class.

Output: `data/free_agency/historical_fit_data.json` -- compact encoding (deduped `player_names`/`team_codes` index arrays plus flat `[player_idx, team_idx, pct, team_z, is_real]` rows, to keep the embedded JSON small), keyed by `season_next` (7 keys, 2019-20 through 2025-26). Each entry: `season, season_next, n_movers, n_teams, team_summary` (each team's best/worst single free-agent fit by `TEAM_Z`), `player_names, team_codes, team_names, destinations, rows`. Sample (2019-20 cycle): `n_movers=113`; Atlanta's best single fit was Harrison Barnes (+39.9% weighted need reduction), worst was Elfrid Payton (-2.6%); Washington's best was Brook Lopez (+31.8%). Prints per-cycle match-rate diagnostics (players matched to both a pre-season roster spot and a real destination, vs. unmatched).

### Backtesting the full pipeline (`evaluate_historical_need_projections.py`)

For each of the 210 (30 teams x 7 cycles) team-seasons, reconstructs `PRE_FA + real arrivals` (same free-agent-class matching as above), runs it through the linearized PLS model for a projected `NEED_VECTOR`, and compares against the real team-stats `NEED_VECTOR`; feeds the projection through a fresh production `RidgeCV` (fit on all 8 `STANDARD_SEASONS` of real `NEED_VECTOR -> W_PCT`) for a projected win%, compared against the real win%.

This is a genuinely *before-the-season* backtest, unlike `team.md`'s `predict_win_pct_from_need.py` LOSO check (which scores each team's real, hindsight `NEED_VECTOR`) -- every input here (pre-FA roster, real signings) is knowable before `season_next` tips off.

Output: `data/free_agency/historical_need_projection_eval.csv` -- 210 rows: `SEASON, SEASON_NEXT, TEAM, N_PRE_FA, N_REAL_ARRIVALS, PROJECTED_WPCT, REAL_WPCT, NEED_RMSE` plus `PROJ_<cat>`/`REAL_<cat>` for all 12 categories. Prints win% RMSE (in both `W_PCT` and win units), win% correlation, mean per-category `NEED_RMSE`, and per-cycle win% RMSE -- see the CSV for exact values.

---

## Free-agent combo search (2026 class)

Both combo scripts share the same trick: since the downstream Ridge/PLS models are linear/affine, the whole pipeline collapses into closed-form linear algebra, so a combination of free agents can be scored via vectorized numpy batches (`BATCH_SIZE=200,000`) instead of a full pandas/sklearn pass per combination. The actual bottleneck at that point is `itertools.combinations` enumeration itself at large `k`.

### Scored by projected win total (`exhaustive_free_agent_combos.py` -- not run/no output on disk)

For each team, enumerates every size-`k` subset of the full 62-player 2026 free-agent pool (`k` = that team's real July-2026 signing count) and scores each combo via `predict_rating_detour_win_pct.py`'s production Ridge stack (`win_pct_variants.md`), which is linear and so collapses to dot products. Reports the best/worst-scoring combo per team plus each free agent's average projected wins across every combo containing them -- a same-coalition-size marginal-contribution proxy, not a true Shapley value. `--max-k N` caps/skips searches too large to run. Combinatorial scale is real: e.g. a 7-of-62 search is on the order of tens of millions of combinations. Its intended outputs (`data/free_agency/exhaustive_fa_combos_summary.csv`, `..._player_impact.csv`) are **not present in the repo** -- this variant does not appear to have been run to completion, or its output was not committed.

### Scored by weighted need reduction (`exhaustive_free_agent_combos_need.py` -- output on disk)

Architecture difference: scores combos by weighted %-reduction in `NEED_VECTOR` rather than predicted win total, using only the PLS reconstruction (not the PLS+XGBoost ensemble), because PLS alone is exactly affine and can be "linearized" via unit-basis-vector evaluation into an explicit `(25, 12)` weight matrix `W` plus a `(12,)` intercept (`fit_need_pls()`, with a sanity assertion that the linearized predictions match the real PLS output to <1e-6).

**Metric**: `pct_increase = (need_before - need_after) / max(1 - need_before, DENOM_FLOOR) * 100` per category, then a weighted composite -- weights are the absolute value of each category's Ridge coefficient from `win_pct_model_results.json` (`team.md`), normalized to sum to 1. `PERIMETER_DEFENSE` (coefficient -0.355) dominates the weighting, more than double the next-largest category, so a free agent who only helps a low-weight category (e.g. `PUT_BACKS`, -0.008) scores worse here than an unweighted "how many needs did they fix" count would suggest. `DENOM_FLOOR = 0.15` caps the blowup from a near-1 baseline need -- documented example: Washington's `PERIMETER_DEFENSE` need_before of 0.986 would otherwise produce a +358% single-category swing before flooring. `NEED_CLIP = (0.01, 0.99)`.

Output: `data/free_agency/exhaustive_fa_combos_need_summary.csv` (28 rows -- GSW and WAS excluded, zero July-2026 signings) and `data/free_agency/exhaustive_fa_combos_need_player_impact.csv` (1,316 rows). Notable results: LAL (`k=7`) `N_COMBOS=62,891,499`, best combo score +23.50, worst -21.34; PHI (same `k`/combo count) best +20.28, worst -13.81; SAC (`k=2`) best +31.35, the highest of any small-`k` team; UTA (`k=5`) best +51.83; **CHI (`k=3`) best +67.92**, the single highest best-combo score in the table, on the 2nd-highest pre-FA weighted composite need in the table (0.646).

### Per-player team-fit ranking (`rank_teams_per_free_agent.py`)

Flips the team-centric combo search into a player-centric one: for each of the (up to 47 of 62) usable free agents, ranks all 28 evaluable teams (GSW/WAS excluded, no real July-2026 signings) 1-30 by the same weighted %-need-reduction score, scored as a single-player (`k=1`) addition on top of that team's real pre-FA roster.

Two z-score normalizations, both explicitly documented as non-rank-changing rescales that only fix cross-comparability, not new information:
- **`TEAM_Z_SCORE`** -- standardized within each team's 47-player score distribution. Motivated because the raw % score correlates 0.80 with the team's own pre-FA weighted composite need (i.e. it mostly measures "is this team bad" rather than "does this specific player help"); z-scoring within team drops that correlation to an insignificant 0.10.
- **`PLAYER_Z_SCORE`** -- the mirror image, standardized within each player's 30-team score distribution. This one is *not* correcting a confound -- "which teams gain the most from this player" being dominated by weak teams is real signal -- it's purely for comparing scores across different players.

Output: `data/free_agency/free_agent_team_fit_rankings.csv` -- long format, 1,316 rows (47 players x 28 teams): `PLAYER_NAME, TEAM, RANK, WEIGHTED_PCT_NEED_INCREASE, TEAM_Z_SCORE, PLAYER_Z_SCORE`.

---

## Realized impact of the actual 2026 free agency (`simulate_free_agency_impact.py`)

Unlike the combo scripts above, this one is not linearized -- it calls `predict_rating_detour_win_pct.py`'s full `predict_wins()` per roster, not a closed-form dot product.

1. **PRE_FA vs. full-roster win impact**: predicts wins for each team's real full 2026-27 roster and for that same roster with July-2026 signings removed; `FA_WIN_IMPACT = WINS_FULL - WINS_PRE_FA`. Intended output: `data/free_agency/free_agency_impact.csv` -- **not present on disk.**
2. **Swap scenarios**: pairs teams that signed the same *number* of free agents this offseason and swaps which team actually landed them (roster size held constant), predicting each team's wins under the swap vs. under their real signings. Intended output: `data/free_agency/free_agency_swap_scenarios.csv` -- **not present on disk.**
3. Writes the free-agent pool itself: `data/free_agency/free_agency_pool_2026.csv` (62 rows, described above under "Identifying a free-agent class") -- **this one is present.**

Two of this script's three outputs are missing from the current working copy; either it hasn't been run to completion, or those two files were generated and later removed. Treat `FA_WIN_IMPACT`/swap-scenario numbers as not currently reproducible from a committed file.

---

## Fit vs. contract value (`analyze_fit_vs_contract.py`)

Cross-analysis run once across all 8 offseason cycles (2026-27 plus the 7 historical ones), joining the fit-ranking machinery above against `contract_model.md`'s v3 salary predictions.

1. **Player fit vs. contract range**: for each cycle, computes each free agent's global average weighted-fit-score (mean across every evaluable team -- "how broadly useful, independent of destination") and compares it to the contract model's predicted salary. Since the salary model outputs only a point prediction with no native uncertainty, this constructs an empirical range from the model's real historical residuals (`ACTUAL - PREDICTED`, pooled across all 754 backtested rows in `contract_model_v3_predictions.csv`): 25th/75th percentile = -2.07pp / +1.65pp of cap, applied as `LOW/HIGH = point_prediction +/- that residual band`. Dollar conversion uses that specific season's own salary cap (`data/salary_cap/cap_history.csv`), not 2026-27's.
2. **Team average fit score** (2026-27 only): mean fit-score across all 47 free agents per team -- a measure of how broadly needy/easy to upgrade a team is, independent of who actually signs there.

Output: `data/free_agency/player_fit_vs_contract.csv` (687 rows, `SEASON` column stacking all 8 cycles; includes `AVG_FIT_SCORE, PREDICTED_SALARY_PCT_OF_CAP, PRED_LOW_PCT, PRED_HIGH_PCT, PRED_DOLLARS, PRED_LOW_DOLLARS, PRED_HIGH_DOLLARS, ACTUAL_SALARY_PCT_OF_CAP, ACTUAL_DOLLARS, RESIDUAL_PP`) and `data/free_agency/team_avg_fit_score.csv` (28 rows -- teams with no signings excluded; top: CHI 6.055, UTA 3.727, SAC 3.224, matching the highest-need teams surfaced by the combo search above). Prints Pearson r(avg fit score, predicted contract %-of-cap) and r(avg fit score, actual contract %-of-cap) for the 2026-27 cycle, plus the top 10 by average fit score.

---

## Team-specific price within the band (`apply_team_fit_to_contract_band.py`)

Takes the single LOW/PRED/HIGH band per player from the section above and gives one price per (player, team) instead -- "what would this player cost team X specifically," not just "what's this player worth."

**The obvious approach was tried first and rejected.** Regressing the model's historical `RESIDUAL_PP` (`ACTUAL - PREDICTED`) against `TEAM_Z_SCORE` at each free agent's *real* signing team (826 historical (player, real-team) pairs from `historical_fit_data.json`, 660 matched to a contract prediction) looks real when pooled: r=0.114, p=0.003, slope +0.521pp of cap per SD of `TEAM_Z_SCORE` -- fit reads as a price *premium*, not a hometown discount. A leave-one-cycle-out check breaks it, though: the slope's sign flips in 3 of the 7 cycles, and one single cycle (2023-24, r=0.392) accounts for nearly all of the pooled correlation; a slope fit on the other 6 cycles predicts the held-out cycle's residuals with an out-of-fold correlation ranging from -0.10 to +0.39 with no consistent direction. That's noise, not a validated coefficient to calibrate a shift from.

**What's shipped instead is a labeled heuristic, not a fitted model**: `TEAM_Z_SCORE` (2026-27: `free_agent_team_fit_rankings.csv`; historical: every `(player, team)` row in `historical_fit_data.json`, not just each player's real destination) is squashed into a saturating position within the *existing, residual-quantile-derived* band:

```
position = clip(TEAM_Z_SCORE / Z_SCALE, -1, 1)          # Z_SCALE = 2.0
FIT_ADJUSTED_PCT = PRED + position * (HIGH - PRED)        if position >= 0
                  = PRED + position * (PRED - LOW)         if position <  0
```

`TEAM_Z_SCORE = 0` (average fit among that player's evaluated teams) recovers the original `PRED` exactly; a team at `|TEAM_Z_SCORE| >= 2` (a genuinely extreme fit) saturates at the band edge rather than extrapolating past it. This is a directional design choice -- "a team-specific standout fit should plausibly move price within the range the model already considers plausible" -- not a statistically demonstrated effect, and is documented as such everywhere it's surfaced (the script's own docstring, and any UI built on top of it).

Output: `data/free_agency/team_fit_adjusted_contract_bands.csv` -- 20,556 rows across all 8 seasons (`SEASON, PLAYER_NAME, TEAM, TEAM_Z_SCORE, PRED_LOW_PCT, PREDICTED_SALARY_PCT_OF_CAP, PRED_HIGH_PCT, FIT_ADJUSTED_PCT`, dollar versions of all four using that season's own cap, plus `ACTUAL_SALARY_PCT_OF_CAP`/`ACTUAL_DOLLARS`). 2026-27 slice: 27 players x 28 teams (756 rows) -- average band width $5.74M, average spread of fit-adjusted price across a player's 28 teams $2.19M (i.e. the heuristic uses roughly 38% of the available band width on average, not routinely saturating). Only 8 of those 27 players' real July-2026 signing team landed in that player's own top-10 teams by fit score -- a reminder that real free-agency destinations are driven by plenty of things (money, role, location, existing relationships) this pipeline doesn't model at all.

---

## Hand-trimmed frontend exports (not script-generated)

`data/free_agency/artifact_data.json`, `artifact_contract_data.json`, and `artifact_eval_data.json` exist in the data directory but no script in the repo writes those exact filenames -- almost certainly manual/frontend-charting exports, the same pattern `team.md`'s Design Notes already documents for `win_pct_artifact_data.json`. Not regenerated automatically by re-running any pipeline script.

---

## Known Limitations

- **`exhaustive_free_agent_combos.py` (win-total-scored variant) and two of `simulate_free_agency_impact.py`'s three outputs are not present on disk** -- see the sections above for exactly which files are missing.
- **The combo search's "average projected wins across all combos containing a player" is a same-coalition-size marginal-contribution proxy, not a true Shapley value** -- it doesn't average over varying coalition sizes the way a proper Shapley computation would.
- **Historical backtesting only scores single-player (`k=1`) marginal additions**, not full combinatorial searches, given historical free-agent pool sizes (113-175+ per cycle).
- **Team-specific contract prices (`FIT_ADJUSTED_PCT`) are a hand-specified heuristic, not a fitted model.** A fitted version was tried first: regressing the contract model's historical residuals (`ACTUAL - PREDICTED`) on `TEAM_Z_SCORE` at each free agent's real signing team, to learn how much a strong team-specific fit moves a player's price. Pooled across 660 matched signings it looked significant (r=0.114, p=0.003, +0.521pp of cap per SD), but under leave-one-cycle-out the slope's sign flipped in 3 of 7 cycles and 2023-24 alone drove most of the correlation, so there was no stable coefficient to ship. The `Z_SCALE = 2.0` squash into the existing band therefore encodes a directional assumption, not a measured effect.
- **`TEAM_Z_SCORE`'s 0.80-correlation-with-need finding is specific to the 2026 class/pool** -- not re-verified against the historical cycles' fit scores.
- **`build_multi_year_contracts.py`'s required input files are absent from the repo**, so it can't currently be run or its logic verified end-to-end.
