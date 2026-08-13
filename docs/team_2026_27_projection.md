# 2026-27 Team Projection & Market Checks

Chains everything upstream (`rosters_and_minutes.md`'s roster + minutes prediction, `team.md`'s `predict_team_need_from_roster.py` roster-to-`NEED_VECTOR` reconstruction, `win_pct_variants.md`'s win% models) into a genuine forward-looking projection for the season that hasn't been played yet, then sanity-checks it against two external references: Vegas win totals and rookie draft capital.

---

## Team profile projection (`predict_2026_27_team_profile.py`)

Chains `predict_team_need_from_roster.py`'s PLS/XGBoost ensemble (roster -> `NEED_VECTOR`, `team.md`) into `evaluate_roster_need_win_pct.py`'s best-performing "blend" model (the average of a `RidgeCV` fit directly on the 25 roster features and a PLS fit on the 37 augmented features), both refit on all 8 `STANDARD_SEASONS` with no holdout -- there's no 2026-27 ground truth to hold out against.

### Minutes-pool rescale

Introduces `TEAM_TOTAL_MIN_TARGET = 240 * 82 = 19,680`: each 2026-27 roster's aggregate minutes are rescaled to this fixed pool rather than summed from each player's raw most-recent-season minutes independently. Without this, some rosters (Houston, Atlanta) landed above the historical training range and others (Chicago, injury-affected rosters) near the historical minimum -- out-of-distribution inputs for the Ridge/PLS models, whose historical `TOTAL_MIN` feature averages ~19,156. This does **not** fix a related, smaller issue: within a team, relative playing-time *shares* still come from each player's raw last-season minutes, so a recently-traded or injury-shortened star is still underweighted relative to a healthy bench player -- the rescale fixes the team-level total, not the within-team distribution.

`W_PCT_CLIP = (0.05, 0.95)` bounds the output the same way `team.md`'s validated models are clipped.

### Output

`data/2026_27_projections/multi_year_team_profile_predictions.csv` -- 30 rows: `TEAM, ROSTER_N, PRED_W_PCT, PROJECTED_WINS, COMPOSITE_NEED` plus the 12 `NEED_CATEGORIES`. Top projected win totals: OKC 72.2 (`COMPOSITE_NEED 0.368`), HOU 56.1, GSW 55.6.

`TEAM_TOTAL_MIN_TARGET`, `build_2026_27_team_rows()`, `GAMES_PER_SEASON`, and `W_PCT_CLIP` are reused directly by nearly every later 2026-27/free-agency script: `predict_rating_detour_win_pct.py`, `evaluate_direct_roster_win_pct.py`, both `exhaustive_free_agent_combos*.py` scripts, `simulate_free_agency_impact.py`, `rank_teams_per_free_agent.py`, and `historical_free_agency_fit.py` (see `win_pct_variants.md`, `free_agency_2026.md`).

### Sanity-check artifact (`allnba_2026_projection.json`)

`data/2026_27_projections/allnba_2026_projection.json` has no corresponding committed script -- it reads as an ad hoc/interactive run rather than a saved pipeline step. Builds a hypothetical 15-man "All-NBA-caliber" roster (Cunningham, Holmgren, Mitchell, Brunson, Duren, Johnson, Murray, Brown, Leonard, Durant, Doncic, Jokic, Gilgeous-Alexander, Maxey, Wembanyama) using each player's most-recent minutes/vector, rescales to `realistic_total_min_used=19,156.12`, and reports what the three competing win% models (direct Ridge, augmented PLS, and their blend) say about this fictional superteam: `direct_ridge_pred=0.7107`, `augmented_pls_pred=0.7672`, `blend_pred=0.7390` (~58 wins), plus the reconstructed `NEED_VECTOR`. Functions as a check that the blend model doesn't produce an absurd or unbounded win% for an extreme, unrealistic roster -- not a real projection of any actual team.

---

## Vegas comparison (`compare_win_totals_vs_vegas.py`)

A one-time market snapshot, not a live odds feed -- 30 hardcoded `(team, total, over_odds, under_odds)` tuples, per the script's own docstring, rather than an API pull.

Converts American odds to implied probability, then to a vig-free "fair" over probability: `OVER_FAIR_PROB = OVER_IMPLIED / (OVER_IMPLIED + UNDER_IMPLIED)`. Merges against `multi_year_team_profile_predictions.csv` and computes `DIFF = PROJECTED_WINS - VEGAS_TOTAL`.

Output: `data/2026_27_projections/multi_year_win_totals_vs_vegas.csv` -- 30 rows, sorted by `DIFF` descending. Also prints mean absolute diff, correlation(projection, Vegas total), sum-of-totals comparison, and average vig.

Largest positive `DIFF` (model more bullish than Vegas): LAC +18.4 (46.9 projected vs. 28.5 line), GSW +16.1 (55.6 vs. 39.5), OKC +11.7 (72.2 vs. 60.5).

---

## Rookie draft capital vs. projection gap (`analyze_rookie_draft_capital_vs_wins.py`)

Extracts drafted rookies' pick numbers from `rosters_2026-27.csv`'s `HOW_ACQUIRED` field via regex (`"#(\d+) Pick in 2026 Draft"`), classifying each rookie as `known_pick`, `traded_pick_unknown`, or `undrafted_signed`. Per-pick value uses a geometric decay curve, `value = 100 * 0.94^(pick - 1)`, aggregated to a team-level `DRAFT_CAPITAL` score, then correlated against the Vegas-vs-projection `DIFF` from the script above.

Output: `data/2026_27_projections/rookie_draft_capital_vs_wins.csv` -- 30 rows: `TEAM, TEAM_NAME, DRAFT_CAPITAL, ROOKIE_COUNT, KNOWN_PICK_COUNT, BEST_KNOWN_PICK, PROJECTED_WINS, VEGAS_TOTAL, DIFF`. Prints Pearson/Spearman correlation of draft capital vs. `DIFF`, plus top-5/bottom-5 teams by draft capital.

Draft capital here is an explicit "partial-information lower bound" -- rookies acquired via traded pick rights have no recoverable pick number from `HOW_ACQUIRED`, so those rookies contribute nothing to `DRAFT_CAPITAL` even though the team did spend a real asset to get them.

---

## Known Limitations

- **The Vegas odds table is a hardcoded, one-time snapshot** -- it will silently go stale as the market moves and is not re-pulled automatically alongside the projection.
- **Draft capital undercounts teams that acquired rookies via traded pick rights** rather than their own pick, since `HOW_ACQUIRED`'s regex only recovers a pick number for players drafted directly by that team.
- **`TEAM_TOTAL_MIN_TARGET`'s within-team share problem (raw last-season minutes) is explicitly unfixed** -- see "Minutes-pool rescale" above.
- **`allnba_2026_projection.json` has no backing script**, so its exact construction (which season's vectors/minutes for each of the 15 players, the precise rescale factor) can't be regenerated or re-verified from a saved file, only reproduced by re-deriving it from the documented method above.
