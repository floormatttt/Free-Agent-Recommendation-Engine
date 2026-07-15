# Defensive & Rebounding Vector Pipeline

---

## Data Sources

| Block | Source |
|-------|--------|
| Perimeter defense (activity) | `LeagueHustleStatsPlayer` |
| Perimeter defense (outcome) | `TeamPlayerOnOffSummary` (on/off DRTG, on-court MIN) |
| Perimeter defense (shrinkage prior) | `LeagueDashTeamStats` (season team DEF_RATING) |
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
| On-court DRTG | Outcome (shrunk, see Step 0) | `DEF_RATING` (on court) + `MIN` (on court) from `TeamPlayerOnOffSummary` |
| Team DRTG | Shrinkage prior only | Season `DEF_RATING` from `LeagueDashTeamStats`, matched to the player's team |

### Step 0 — Shrink on-court DRTG toward team DRTG

A player's raw on-court DEF_RATING is noisy at low minutes — a handful of possessions can swing it wildly, which previously let low-minute players dominate both tails of the perimeter defense block. Before anything else, it's stabilized with empirical-Bayes shrinkage toward the player's own team's season DEF_RATING, weighted by on-court minutes against a fixed pseudo-minutes constant:

```
on_drtg = (on_court_min × on_court_drtg + K × team_drtg) / (on_court_min + K)
K = 500  (pseudo-minutes of prior weight)
```

A player with on-court minutes well under `K` ends up close to their team's average; a full-season starter (minutes >> `K`) is barely pulled off their own raw number. All later steps use this shrunk `on_drtg`, not the raw on-court DEF_RATING.

Note: this only fixes small-sample *noise*. It does not separate individual defense from team scheme — the prior itself is the team's own rating, so a low-minute player on an elite defensive team still shrinks toward a flattering number for reasons unrelated to their own defense. See Known Limitations.

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

### Step 4 — Adjust on/off DRTG using activity composite

On/off DRTG captures overall defensive impact but is team-influenced. The activity composite corrects for team context by pulling players with high individual activity toward a better rating and vice versa:

```
adjusted_drtg = on_drtg - (0.20 × activity_score × std(on_drtg))
```

`on_drtg` here is the shrunk value from Step 0, not the raw on-court DEF_RATING. Lower adjusted_drtg = better defender.

### Step 5 — Z-score and shift positive

```
adjusted_drtg_z   = (league_avg_drtg - adjusted_drtg) / std(adjusted_drtg)
perimeter_d_block = adjusted_drtg_z - min(adjusted_drtg_z)
```

Flipped so higher = better. Shifted so worst player = 0, all values non-negative.

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
| Perimeter defense | adjusted_drtg_z shifted by min | Yes — shifted |
| Rim protection | weighted geometric mean of 4 inverted/positive stats | Yes — inverted then geometric mean |
| Rebounding | cubic root geometric mean of 3 stats | Yes — zero-anchored inputs |

---

## Known Limitations

- **Perimeter defense** remains the weakest block — no public stat captures staying in front of elite ball handlers, help defense rotations, or deterrence effects. `PCT_PLUSMINUS on 3PT defense` from `LeagueDashPtDefend` should be added as an anchor when pipeline is complete — it directly measures shot quality impact and would replace the activity-adjusted DRTG as primary signal.
- **Team context still leaks through on-court DRTG.** Step 0's shrinkage fixes small-sample *noise* (a low-minute player no longer swings wildly off a handful of possessions) but shrinks toward the player's own *team's* DRTG, so it does not separate individual defense from scheme — a bench player on an elite defensive team still shrinks toward a flattering number for reasons unrelated to their own defense. Observed directly: the 2025-26 perimeter defense leaderboard is dominated by Thunder players. A cleaner fix would shrink toward the team's *off-court* DRTG (i.e. the team's rating without that specific player on the floor, also available from `TeamPlayerOnOffSummary`) or explicitly team-mean-center on_drtg before z-scoring.
- **`K_SHRINKAGE = 500`** (pseudo-minutes of prior weight) is manually chosen, not fit from data. A cross-validated choice (minimizing out-of-sample variance of on/off splits by minutes bucket) would be more rigorous.
- **Rebounding** compares all positions on the same scale without positional normalization — a guard at 20% DREB% appears below average because they are compared to centers. Consider positional archetype grouping for this block specifically if rebounding drives key recommendations.
- **DFG% and Diff% inversion** ties the scale to whoever the worst rim defender in the dataset is — if a new player worse than any current player is added, all inverted scores shift slightly. Acceptable for a seasonally retrained model.
- **Activity composite weight on DRTG** (0.20) is manually calibrated — a regression fitting this weight against historical on/off data would be more rigorous.