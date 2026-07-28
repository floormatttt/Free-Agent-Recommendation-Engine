# Defensive & Rebounding Vector Pipeline

---

## Data Sources

| Block | Source |
|-------|--------|
| Perimeter defense (activity) | `LeagueHustleStatsPlayer` |
| Perimeter defense (outcome) | Self-computed Defensive Win Shares (Dean Oliver's formula) via `scripts/compute_dws.py`, from `LeagueDashPlayerStats`/`LeagueDashTeamStats` (measure_type="Advanced"/"Base") -- not NBA.com's own `DEF_WS` field, which is missing for 5 of 12 seasons on file |
| Rim protection | `PlayerDashPtShotDefend` (defense_category="Less Than 6 Ft") |
| Rebounding | `LeagueDashPlayerStats` (measure_type="Advanced") + `LeagueHustleStatsPlayer` |

---

## Perimeter Defense Block

### Raw stats

| Stat | Type | Source column |
|------|------|---------------|
| Contested 3PT shots | Total → rate-adjust | `CONTESTED_SHOTS_3PT` / `GP` |
| Deflections | Total → rate-adjust | `DEFLECTIONS` / `GP` |
| STL% | Already rate-adjusted | `STL_PCT` |
| Defensive Win Shares | Outcome, computed then rate-adjusted (Step 0) | `DWS_PER_GAME`, self-computed by `scripts/compute_dws.py` |

### Step 0 — Compute Defensive Win Shares (Dean Oliver's formula) and convert to a per-game rate

Rather than depend on NBA.com's own `DEF_WS` field (absent for 5 of 12 seasons on file, confirmed via repeated retries -- not a transient outage), `scripts/compute_dws.py` implements Basketball-Reference's exact DWS methodology directly from raw inputs already on file:

```
marginal_defense      = (MP_player / MP_team) × Poss_def_team × (1.08 × PPP_league − DRtg_player / 100)
marginal_pts_per_win   = 0.32 × PPG_league × (Pace_team / Pace_league)
DWS_RAW                = marginal_defense / marginal_pts_per_win
```

Where:
- `DRtg_player` = NBA.com's individual `DEF_RATING` (from `LeagueDashPlayerStats`, measure_type="Advanced") — an Oliver-style per-player estimate incorporating team defense plus the player's own steals/blocks/rebounds/fouls. This is *not* the on-court/off-court DEF_RATING split used by an earlier version of this pipeline (see Known Limitations for why that was replaced).
- `MP_team` = the sum of that team's own players' `MIN` for the season (not the team_stats `MIN` column, which is the team's single-unit game minutes, not the roster's combined 5-on-court total).
- `Poss_def_team` = the team's own season `POSS` (team_stats, measure_type="Advanced") — a team's defensive possessions are approximated by its own tempo-free possession count, since possessions for/against balance out over a full season.
- `Pace_team` = team_stats `PACE`.
- `PPP_league = sum(team PTS) / sum(team POSS)`, `PPG_league = sum(team PTS) / sum(team GP)`, `Pace_league` = GP-weighted mean of team `PACE` — all computed once per season across the 30 teams.

`DWS_RAW` is a season-total stat (accumulates with playing time, like real win shares), so it's converted to a per-game rate before use elsewhere in this block:

```
dws_per_game = DWS_RAW / GP
```

All later steps use `dws_per_game`.

This replaces an earlier approach that shrunk on-court DEF_RATING toward the player's own team's season DEF_RATING (empirical-Bayes shrinkage, K=500 pseudo-minutes). That approach fixed small-sample noise but leaked team scheme into individual ratings — a low-minute player on an elite defensive team shrank toward a flattering number for reasons unrelated to their own defense (observed directly: the 2025-26 leaderboard was dominated by Thunder players). DWS isn't fully isolated from team context either (it's still a box-score-derived stat, not a play-by-play isolation of individual defense), but it doesn't have that specific team-prior leak. See Known Limitations.

Verified against real-world numbers: 2023-24's computed leaderboard is topped by Rudy Gobert (`DWS_RAW` ≈ 5.24), matching Basketball-Reference's actual 2023-24 DWS leader and value closely; the next several spots are dominated by Minnesota players, consistent with the Timberwolves' #1 defensive rating that season.

### Step 1 — Rate adjust counting stats

```
contested_3pt_pg = CONTESTED_SHOTS_3PT / GP
deflections_pg   = DEFLECTIONS / GP
```

### Step 2 — Zero-anchored z-scores

```
contested_3pt_z = contested_3pt_pg / std(contested_3pt_pg)
deflections_z   = deflections_pg   / std(deflections_pg)
stl_pct_z       = STL_PCT         / std(STL_PCT)
```

### Step 3 — Activity composite

```
activity_score = contested_3pt_z^0.35 × deflections_z^0.35 × stl_pct_z^0.30
```

Contested 3PT and deflections share weight since they are correlated — both measure active perimeter engagement. STL% captures a distinct skill (anticipation, hand speed) so carries roughly equal weight independently.

### Step 4 — Adjust DWS/game using activity composite

DWS/game captures overall defensive impact but is still partly team-influenced (see Known Limitations). The activity composite corrects for this by pulling players with high individual activity toward a better rating and vice versa:

```
adjusted_dws = dws_per_game + (0.20 × activity_score × std(dws_per_game))
```

`dws_per_game` here is the per-game value from Step 0. Higher adjusted_dws = better defender (unlike DRTG, DWS is already oriented so higher is better, so this step adds rather than subtracts the activity term).

### Step 5 — Z-score and shift positive

```
adjusted_dws_z    = (adjusted_dws - league_avg_dws) / std(adjusted_dws)
perimeter_d_block = adjusted_dws_z - min(adjusted_dws_z)
```

Shifted so worst player = 0, all values non-negative.

---

## Rim Protection Block

### Raw stats

| Stat | Direction | Source column |
|------|-----------|---------------|
| BLK% at rim | Higher = better | `BLK_PCT` filtered to < 6ft |
| DFG% at rim | Lower = better | `DFG_PCT` (< 6ft) |
| Diff% at rim | Lower = better | `PCT_PLUSMINUS` (< 6ft) |
| DFGA at rim | Higher = better (volume) | `D_FGA` (< 6ft) |

Note: DFGM is dropped — it is fully redundant with DFG% and DFGA combined. League average FG% at rim is used only to compute Diff% and is not a separate vector dimension.

### Step 1 — Zero-anchored z-scores

```
BLK%_z  = BLK%  / std(BLK%)
DFG%_z  = DFG%  / std(DFG%)
Diff%_z = Diff% / std(Diff%)
DFGA_z  = DFGA  / std(DFGA)
```

### Step 2 — Invert DFG% and Diff% so higher = better

```
DFG%_z_inv  = max(DFG%_z)  - DFG%_z
Diff%_z_inv = max(Diff%_z) - Diff%_z
```

Subtracting from the maximum value in the dataset flips the direction while keeping all values non-negative. Best rim defender = highest value, worst = 0.

### Step 3 — Weighted geometric mean

```
rim_block = BLK%_z^0.25 × DFG%_z_inv^0.30 × Diff%_z_inv^0.35 × DFGA_z^0.10
```

| Stat | Weight | Rationale |
|------|--------|-----------|
| Diff% at rim | 0.35 | Primary outcome signal — accounts for baseline difficulty |
| DFG% at rim | 0.30 | Raw efficiency allowed — direct impact measure |
| BLK% at rim | 0.25 | Shot deterrence — distinct skill from suppressing FG% |
| DFGA at rim | 0.10 | Volume only — not a quality signal, minimal weight |

---

## Rebounding Block

### Raw stats

| Stat | Type | Source column |
|------|------|---------------|
| DREB% | Already rate-adjusted | `DREB_PCT` |
| OREB% | Already rate-adjusted | `OREB_PCT` |
| Defensive box outs | Total → rate-adjust | `DEF_BOXOUTS` / `GP` |

### Step 1 — Rate adjust box outs

```
def_box_outs_pg = DEF_BOXOUTS / GP
```

### Step 2 — Zero-anchored z-scores

```
DREB%_z           = DREB_PCT          / std(DREB_PCT)
OREB%_z           = OREB_PCT          / std(OREB_PCT)
def_box_outs_pg_z = def_box_outs_pg   / std(def_box_outs_pg)
```

### Step 3 — Cubic root geometric mean

```
reb_block = (DREB%_z × OREB%_z × def_box_outs_pg_z) ^ (1/3)
```

Equal weighting across all three — they measure genuinely distinct skills:
- DREB% = defensive glass production
- OREB% = offensive glass production (different physical and positional skill)
- Def box outs = effort and positioning intent, independent of whether rebound was secured

---

## Summary

| Block | Formula | All positive? |
|-------|---------|---------------|
| Perimeter defense | adjusted_dws_z shifted by min | Yes — shifted |
| Rim protection | weighted geometric mean of 4 inverted/positive stats | Yes — inverted then geometric mean |
| Rebounding | cubic root geometric mean of 3 stats | Yes — zero-anchored inputs |

---

## Known Limitations

- **Perimeter defense** remains the weakest block — no public stat captures staying in front of elite ball handlers, help defense rotations, or deterrence effects. `PCT_PLUSMINUS on 3PT defense` from `LeagueDashPtDefend` would still be a valuable addition as a direct shot-quality anchor, independent of the DWS approach below.
- **DWS is still noisy at low minutes, though less than the earlier per-48-DEF_WS approach.** A player with very few games can post an extreme `dws_per_game` from a small sample of good possessions -- observed directly on the 2025-26 vectors, where Zach Edey (11 GP) tops the perimeter-defense leaderboard and a few sub-100-minute players (Colby Jones at 1 GP, Kadary Richmond at 3 GP) score unrealistically high, though the great majority of the leaderboard is now legitimate rotation defenders (SGA, Wembanyama, Holmgren, Derrick White), a marked improvement over the prior approach's near-total domination by garbage-time players. There's no shrinkage step guarding against this. `apply_minutes_gate.py` partially compensates downstream at the final salary-prediction stage, but the raw perimeter-defense vector itself doesn't correct for it. A minutes/games-weighted shrinkage toward a league-average prior (the same shape as the old `K_SHRINKAGE` step) would be the natural fix.
- **Individual DRtg (`DEF_RATING`) is still team-influenced.** It's an Oliver-style estimate incorporating team defense alongside a player's own steals/blocks/rebounds/fouls, not an isolation of individual defense from scheme -- a player on a good defensive team still benefits somewhat, though nowhere near as directly as the old on-court-DRTG-shrunk-toward-team-DRTG approach did.
- **`Poss_def_team` is approximated by the team's own offensive-tempo possession count** (`POSS` from `LeagueDashTeamStats`), the standard simplification since defensive and offensive possessions balance out over a full season -- not an exact per-team defensive-possession count.
- **A traded player's team-level inputs (`MP_team`, `Poss_def_team`, `Pace_team`) come from only their most-recent team** for the season (the same simplification used elsewhere in this pipeline, e.g. the old team-DRTG shrinkage prior lookup), not a stint-weighted blend across every team they played for.
- **Rebounding** compares all positions on the same scale without positional normalization — a guard at 20% DREB% appears below average because they are compared to centers. Consider positional archetype grouping for this block specifically if rebounding drives key recommendations.
- **DFG% and Diff% inversion** ties the scale to whoever the worst rim defender in the dataset is — if a new player worse than any current player is added, all inverted scores shift slightly. Acceptable for a seasonally retrained model.
- **Activity composite weight on DWS** (0.20) is manually calibrated — a regression fitting this weight against historical data would be more rigorous.