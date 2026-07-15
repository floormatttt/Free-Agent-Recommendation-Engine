# Vector Process

Covers `build_playtype_vectors.py`, `build_defense_vectors.py`, and `build_player_vectors.py` -- the pipeline that turns raw player stats into the finalized 12-category player vector (player.md, defense.md).

Output: `data/player_vectors_json/player_vectors_<season>.json`, one file per season, keyed by `PLAYER_ID`, each entry `{PLAYER_NAME, TEAM, SEASON, VECTOR}` where `VECTOR` is a 12-key category -> score map. Currently generated for 2018-19 through 2025-26 (8 seasons -- see Changelog for why 2016-17/2017-18 were dropped).

---

## Shared machinery: experience-adjusted time decay

Both `build_playtype_vectors.py` and `build_defense_vectors.py` blend up to 3 trailing seasons (current, current-1, current-2) using the same decay scheme:

- `experience` = years since draft (`DRAFT_YEAR`), or number of seasons on file as a fallback for undrafted players.
- `decay_strength(experience)`: U-shaped in experience -- high at the start and end of a career (recent form dominates), low ("bottoms out") at the prime (`PRIME_CENTER=7`, `PRIME_SPREAD=5`, `DECAY_MIN=0.15`, `DECAY_MAX=1.2`).
- Weights for available seasons: `exp(-decay_strength * i)` for `i` in `{0, 1, 2}` (0 = current), normalized over whichever seasons actually exist in `data/player_stats/`.
- A missing trailing season (rookie, or simply outside our data window) is left out of the weighting and the remaining weights renormalize -- no imputation, no distinction between "missed due to injury" and "missing because our dataset doesn't go back further." Verified directly: in 2016-17 (earliest season on file), every player's decayed value equals their raw single-season value exactly (only `i=0` available); in 2017-18, values are genuine 2-season blends.

---

## Playtype scores (`build_playtype_vectors.py`)

For each of the 9 play types (Isolation, PNR Handler, PNR Roll Man, Post Up, Spot Up, Handoff, Cut, Off Screen, Put Backs):

1. Decay-blend `FREQ%` and `PPP` across up to 3 seasons (see above).
2. **Baseline substitution:** if a season has no recorded data for a play type (player never ran it), substitute `BASELINE_FREQ=0.005`, `BASELINE_PPP=0.9` for that season before blending, per player.md.
3. **Score conversion:** player.md specifies `z(value) - z(0)` rather than a plain z-score. Since `z(value) - z(0) = (value-mean)/std - (0-mean)/std = value/std`, this simplifies to just `value / population_std` -- a player who never runs a play type scores exactly 0, not a large negative z-score. Applied to both FREQ and PPP.
4. **Playtype score:** geometric mean of the FREQ-score and PPP-score (`sqrt(freq_score * ppp_score)`).

Output columns: `{TYPE}_FREQ_DECAYED`, `{TYPE}_PPP_DECAYED`, `{TYPE}_FREQ_SCORE`, `{TYPE}_PPP_SCORE`, `{TYPE}_SCORE` (the final playtype score used in the 12-category vector).

---

## Defense & rebounding scores (`build_defense_vectors.py`)

Implements defense.md's 3 blocks. Same decay-blend machinery as playtypes; missing raw stats (no baseline rule defined for defense) fall back to that season's population mean as a neutral stand-in.

### Perimeter defense block

1. Rate-adjust: `contested_3pt_pg = CONTESTED_SHOTS_3PT / GP`, `deflections_pg = DEFLECTIONS / GP`.
2. **Shrink on-court DRTG toward team DRTG** (added after the fact -- see Changelog): a player's raw on-court DEF_RATING is noisy at low minutes, so it's blended with their own team's season DEF_RATING, weighted by on-court minutes against `K_SHRINKAGE=500` pseudo-minutes: `on_drtg = (on_court_min * on_court_drtg + K * team_drtg) / (on_court_min + K)`.
3. Zero-anchored z-scores (`value/std`, no mean subtraction -- these raw stats are naturally non-negative, so this stays non-negative) for `contested_3pt_pg`, `deflections_pg`, `STL_PCT`.
4. `activity_score = contested_3pt_z^0.35 * deflections_z^0.35 * stl_pct_z^0.30`.
5. `adjusted_drtg = on_drtg - (0.20 * activity_score * std(on_drtg))`.
6. `adjusted_drtg_z = (mean(adjusted_drtg) - adjusted_drtg) / std(adjusted_drtg)` (flipped so higher = better).
7. `perimeter_d_block = adjusted_drtg_z - min(adjusted_drtg_z)` (shifted so the worst player = 0).
8. **Rescaled to match RIM_BLOCK/REB_BLOCK's magnitude** (see Changelog) -- the min-shift in step 7 systematically inflated the mean relative to the other two blocks.

### Rim protection block

1. Zero-anchored z-scores for `BLK%` (rate, always non-negative), `DFG%` (always non-negative), `Diff%` (can be negative), `DFGA` (always non-negative).
2. Invert `DFG%_z` and `Diff%_z` via max-shift (`max(z) - z`) so higher = better and both stay non-negative regardless of sign -- this is the fix for a sign-flip bug found in an earlier version (see Changelog).
3. `rim_block = BLK%_z^0.25 * DFG%_z_inv^0.30 * Diff%_z_inv^0.35 * DFGA_z^0.10`. All 4 factors are non-negative by construction, so ordinary exponentiation is safe (no signed-power trick needed).

### Rebounding block

1. Rate-adjust: `def_box_outs_pg = DEF_BOXOUTS / GP`.
2. Zero-anchored z-scores for `DREB%`, `OREB%`, `def_box_outs_pg` (all non-negative).
3. `reb_block = (DREB%_z * OREB%_z * def_box_outs_pg_z) ^ (1/3)`.

---

## Final assembly (`build_player_vectors.py`)

Merges the 9 `{TYPE}_SCORE` columns from the playtype CSV with `RIM_BLOCK -> RIM_DEFENSE`, `PERIMETER_D_BLOCK -> PERIMETER_DEFENSE`, `REB_BLOCK -> REBOUNDING` from the defense CSV, into the final 12-key `VECTOR` per player, saved as JSON.

---

## Changelog

- Built `build_playtype_vectors.py` per player.md: decay blend, baseline substitution, `z(v)-z(0)` scoring, geometric mean playtype score. Ran for all seasons on file.
- Built an initial `build_defense_vectors.py` covering only player.md's brief 2-formula spec (`rim_block`, `reb_block`), using a sign-preserving power (`sign(z)*|z|^p`) to handle negative z-scores in the product.
- defense.md formalized the full 3-block spec (adding the perimeter defense block, and replacing the sign-preserving-power approach with max-shift inversion for rim protection). Rewrote `build_defense_vectors.py` to match exactly.
- `build_player_vectors.py` initially left `RIM_DEFENSE`/`PERIMETER_DEFENSE`/`REBOUNDING` as `null` placeholders (9 of 12 categories implemented); backfilled once defense.md's blocks were built.
- Generated vectors for all 10 seasons on file (2016-17 through 2025-26) -- confirmed 2016-17 and 2017-18 use 1- and 2-season decay windows respectively (data-boundary effect, not a bug). Later, `player_vectors_2016-17.json` and `player_vectors_2017-18.json` were deleted at the user's request, leaving 8 seasons (2018-19 through 2025-26) as the standard set.
- Added empirical-Bayes shrinkage (`K_SHRINKAGE=500`) for on-court DEF_RATING in the perimeter defense block, to fix wild small-sample-driven outliers (e.g. a 6.8-minute player swinging the whole distribution). Re-pulled `ON_COURT_MIN` from `pull_nba_data.py` to support this. Verified: `PERIMETER_D_BLOCK` range compressed from 0-14.7 to 0-6.1, and garbage-time outliers were eliminated from both tails. Known residual limitation: shrinking toward the player's own *team's* DRTG doesn't separate individual defense from scheme, so team-context still leaks through (observed: 2025-26 leaderboard still shows an OKC cluster).
- Tried applying the `z(v)-z(0)` (zero-anchored, no mean-centering) convention to rim protection's `BLK%`/`DFG%`/`DFGA` inputs, at the user's request, mirroring the playtype-score convention. This broke rim defense: `DFG%` and `BLK%`/`DFGA` are always-positive stats with no meaningful zero crossing at the population mean, so removing mean-centering rewards raw volume over defensive quality, and combining an always-positive `-DFG%_z` term with a variable-sign `Diff%_z` term produced an outright sign-flip bug (elite rim protectors like Holmgren/Wembanyama scored *negative*). Reverted to standard mean-centered z-scores for these inputs. Confirmed: the `z(v)-z(0)` convention is correct where "zero" is a meaningful natural floor (playtype Freq%/PPP, where "never runs this play" is common) but does not generalize to always-positive rate/count stats.
- Diagnosed and fixed a `PERIMETER_DEFENSE` mean-inflation bug: the min-shift step (`- min(adjusted_drtg_z)`) systematically ran its mean 0.5-1.4 points above the average of the other 11 categories across every season checked (2018-19 through 2025-26), because `RIM_BLOCK`/`REB_BLOCK` have no equivalent shift. Fixed by rescaling `PERIMETER_D_BLOCK` to match the average of `RIM_BLOCK`'s and `REB_BLOCK`'s means (preserves the "worst player = 0, non-negative" design, since scaling a value already anchored at 0 by a positive constant leaves it at 0). Verified: the gap dropped to ~0.04-0.2 points across seasons checked. Regenerated all 8 seasons.
