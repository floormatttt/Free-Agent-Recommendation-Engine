# Data Extraction Process

Covers `pull_nba_data.py`, the `parse-nba-salary-list` skill, and `parse_cap.py` -- everything that pulls or parses raw data into `data/`.

---

## Player & team stats (`pull_nba_data.py`)

Pulls the dataset in Data.txt via `nba_api`, per season, into `data/player_stats/player_stats_<season>.csv` and `data/team_stats/team_stats_<season>.csv`.

### Sources per category

| Category | Endpoint |
|---|---|
| Bios (age, height, draft year/round/pick, team, GP) | `LeagueDashPlayerBioStats` |
| Minutes | `LeagueDashPlayerStats` (Base) |
| Rebounding % (OREB/DREB) | `LeagueDashPlayerStats` (Advanced) |
| % Steals, % Blocks | `LeagueDashPlayerStats` (Usage) |
| Deflections, Contested 3PT, Def Box Outs | `LeagueHustleStatsPlayer` / `LeagueHustleStatsTeam` |
| Rim protection (DFGM/DFGA/DFG%/opp FG%/Diff%, <6ft) | `LeagueDashPtDefend` / `LeagueDashPtTeamDefend` |
| Playtype Freq%/PPP (9 types) | `SynergyPlayTypes` |
| On-court DEF_RATING, on-court MIN | `TeamPlayerOnOffSummary` (looped over all 30 teams) |
| Team DEF_RATING, team BLK | `LeagueDashTeamStats` (Base/Advanced) |

Team IDs for the on-court DRTG loop come from `nba_api.stats.static.teams` (a local list, no extra API calls).

### Known dedup issues (fixed)

- **Synergy playtypes** returns one row per team stint for a player traded mid-season, and merging on `PLAYER_ID` alone produced a cross-product blowup (up to 16 duplicate rows for one player). Fixed in `fetch_playtypes` / `_combine_stints`: stints are combined into a single possession-weighted row before merging across play types.
- **Rim protection** (`LeagueDashPtDefend`) has the identical per-team-stint splitting issue. Fixed via `dedupe_rim_defense`: stints combined into one attempts (FGA)-weighted row.
- **On-court DRTG** (`TeamPlayerOnOffSummary`) also splits traded players by team. Fixed in `fetch_on_court_def_rating`: stints combined via a minutes-weighted average, and total on-court minutes (`ON_COURT_MIN`) is summed across stints (kept as its own column for later shrinkage use, not just used internally).

All three fixes were verified season-by-season (`duplicated(subset=['PLAYER_ID'])` == 0 across every season on file).

### Seasons on file

2014-15 through 2026-27 (2026-27 is salary-only, no stats yet), stored per-category in `data/player_stats/` and `data/team_stats/`.

**Data quality caveat:** hustle-stat coverage (`CONTESTED_SHOTS_3PT`, `DEFLECTIONS`, `DEF_BOXOUTS`) is incomplete for the earliest seasons -- 2014-15 has zero coverage (0/492 players), 2015-16 has partial coverage (147/476, ~31%), consistent with the NBA's hustle-stat tracking not yet being rolled out league-wide. Playtype and rim-defense data are unaffected and well-populated in both seasons. These two seasons exist only to extend the decay window for 2016-17/2017-18's playtype vectors -- they were never used to generate their own player-vector JSONs (2016-17/2017-18 vector JSONs were later deleted; see vector_process.md).

---

## Salaries (`parse-nba-salary-list` skill)

Parses a raw ranked salary-list text dump (repeating 4-line groups: rank-or-blank / player name / "TEAM, POSITION" / "$salary") into `data/salaries/salaries_<season>.csv` with columns `RANK, PLAYER_NAME, TEAM, POSITION, SALARY`.

- A blank rank line means tied with the previous rank; the parser carries the last seen rank forward.
- `$` and commas stripped from salary, converted to a plain integer.
- Multi-position players (e.g. `SF/PF`) preserved as-is.

Applied for seasons 2016-17 through 2026-27 (11 seasons -- one season ahead of the stats data, since 2026-27 salary info exists before that season's stats can be pulled).

The salary-list source is a ranked cutoff (~450 players), not exhaustive -- players outside the list (deep bench/two-way) have no salary on file. This shows up later as ~172 of ~582 vector players in a given season having no matching salary.

---

## Salary cap history (`parse_cap.py`)

Parses `cap.md` (a tab-separated table) into `data/salary_cap/cap_history.csv`: `SEASON, SALARY_CAP, LUXURY_TAX, FIRST_APRON, SECOND_APRON, BAE, NON_TAXPAYER_MLE, TAXPAYER_MLE, TEAM_ROOM_MLE`, covering 2016-17 through 2034-35 (19 seasons, including future projections). `$` and commas stripped to plain integers; `N/A` (pre-2023-24 seasons without a 2nd apron) becomes a null.

Also adds `SALARY_PCT_OF_CAP` to every `data/salaries/salaries_<season>.csv` file: `SALARY / that season's SALARY_CAP * 100`. Verified against real CBA rules -- 10+ year veteran max-salary players (Giannis, Butler, Davis in 2025-26) compute to exactly 35.000000% of cap, confirming correctness.

---

## Changelog

- Built `pull_nba_data.py` covering Data.txt's full player/team stat spec; ran for 2016-17 through 2025-26.
- Fixed Synergy playtype duplicate-row bug (possession-weighted stint combination).
- Fixed rim-defense duplicate-row bug (attempts-weighted stint combination).
- Added `CONTESTED_SHOTS_3PT` next to `DEFLECTIONS`; re-pulled all 10 seasons.
- Added `ON_COURT_DEF_RATING` via new `TeamPlayerOnOffSummary`-based fetch (30 calls/season); re-pulled all 10 seasons.
- Added `ON_COURT_MIN` to the same fetch (needed later for shrinkage weighting); re-pulled all 10 seasons again.
- Reorganized `data/` into buckets: `player_stats/`, `team_stats/`, `salaries/`, `player_vectors_full/`, `player_vectors_json/` (later: `salary_cap/`, `similarity/`).
- Pulled 2014-15 and 2015-16 to extend the decay window for the earliest seasons; discovered and documented the hustle-stat coverage gap in both.
- Built the `parse-nba-salary-list` skill; applied it for 2016-17 through 2026-27.
- Built `parse_cap.py`; parsed 19 seasons of cap history and added `SALARY_PCT_OF_CAP` to all 11 salary files.
