# Team Vector Pipeline

---

## Input

The per-player vectors produced by `build_player_vectors.py` (12 categories: 9 playtypes + `RIM_DEFENSE`, `PERIMETER_DEFENSE`, `REBOUNDING`), plus each player's season `MIN` from `player_stats_{season}.csv` for weighting.

## Output

One vector per team, same 12 categories, keyed by team abbreviation.

---

## Process

### Step 1 — Minutes-weighted aggregation

For each team and each category, take the minutes-weighted average across every player on that team's roster:

```
team_score[cat] = Σ(player_score[cat] × player_MIN) / Σ(player_MIN)
```

Players with 0 or missing minutes are excluded. A player missing a specific category (e.g. no defense data) is excluded from that category's average only, not from the team entirely.

### Step 2 — Re-anchor the range (the compression problem)

Player categories are already zero-anchored z-scores against the ~500-player league (`value / std(value)`, std ≈ 1 by construction — see `player.md` / `defense.md`). Averaging ~15 of those per team collapses the spread by the central limit theorem: a naive team vector clusters tightly near the same value for every team, with almost no separation between a great offense and a mediocre one.

To counter this, each aggregated category is rescaled by its own standard deviation across the 30 teams:

```
team_score_final[cat] = team_score[cat] / std_across_teams(team_score[cat])
```

This is the same "zero-anchored" z-score used everywhere else in the model — divide only, no mean subtraction. That matters because it preserves what 0 means at the player level (never runs that playtype / worst-in-league defender): since the transform is a pure positive rescale, a category that was 0 stays 0. What changes is the *spread* — dividing by the team-level std undoes the CLT compression and restores a full, comparable range (std ≈ 1 across teams) instead of a flat cluster.

---

## Notes / Known Limitations

- **Weighting by raw season minutes** treats every minute equally regardless of when it happened or lineup context (e.g. garbage time, a since-traded player's minutes with a different team). A minutes-decay or recency weighting could be layered in later, matching the experience-adjusted decay already used at the player level.
- **No positional or roster-construction adjustment.** A team heavy on one archetype (e.g. all high-usage isolation scorers) isn't flagged for redundancy here — that's left to the fit/recommendation layer that consumes these vectors, not this pipeline.
- **Injuries/trades mid-season** aren't specially handled — a traded player's minutes are attributed to whichever `TEAM` their row shows in `player_stats_{season}.csv` (their current team as of the data pull), not split across teams.
