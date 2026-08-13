"""
Tests whether a team's 2026 rookie draft capital correlates with how far our
2026-27 win projection (multi_year_team_profile_predictions.csv, via
multi_year_win_totals_vs_vegas.csv's DIFF column) sits above/below the Vegas
market line -- the closest available "predicted vs. actual" signal, since
the 2026-27 season hasn't been played yet and Vegas totals are the best
external ground-truth proxy we have.

Draft pick numbers only come through cleanly on data/rosters/rosters_2026-27.csv's
HOW_ACQUIRED text ("#N Pick in 2026 Draft") for rookies still on their
original draft-day team. Rookies whose rights were traded (roster shows
"Draft Rights Traded from <team>", not a pick number) or who signed as
undrafted free agents have no pick number recoverable from this dataset --
they're reported separately rather than guessed at, since fabricating a
pick number would corrupt the correlation rather than just shrink its
sample. This means team-level draft capital here is a partial-information
lower bound, not a complete account of every rookie's capital.

Pick value uses a standard geometric decay curve (value = 100 * 0.94^(pick-1)),
an approximation of real draft-trade-value charts' shape (steep early
dropoff, flattening by the 2nd round) -- not a precise market valuation.

Output: data/2026_27_projections/rookie_draft_capital_vs_wins.csv, one row per team with
draft capital, rookie counts, and the win-projection DIFF vs Vegas.

Usage:
    python scripts/analyze_rookie_draft_capital_vs_wins.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROSTER_PATH = Path("data/rosters/rosters_2026-27.csv")
VEGAS_PATH = Path("data/2026_27_projections/multi_year_win_totals_vs_vegas.csv")
OUTPUT_PATH = Path("data/2026_27_projections/rookie_draft_capital_vs_wins.csv")

PICK_VALUE_DECAY = 0.94  # value(pick) = 100 * PICK_VALUE_DECAY ** (pick - 1)


def pick_value(pick: pd.Series) -> pd.Series:
    return 100.0 * PICK_VALUE_DECAY ** (pick - 1)


def main():
    rosters = pd.read_csv(ROSTER_PATH, encoding="utf-8")
    rookies = rosters[rosters["EXP"] == "R"].copy()

    pick = pd.to_numeric(rookies["HOW_ACQUIRED"].str.extract(r"#(\d+) Pick in 2026 Draft")[0])
    rookies["PICK"] = pick
    rookies["VALUE"] = np.where(pick.notna(), pick_value(pick), 0.0)
    rookies["STATUS"] = np.select(
        [pick.notna(),
         rookies["HOW_ACQUIRED"].str.startswith("Draft Rights Traded", na=False)],
        ["known_pick", "traded_pick_unknown"],
        default="undrafted_signed",
    )

    print(f"{len(rookies)} rookies on 2026-27 rosters: "
          f"{(rookies['STATUS'] == 'known_pick').sum()} with a known pick number, "
          f"{(rookies['STATUS'] == 'traded_pick_unknown').sum()} drafted but pick unresolved "
          f"(rights traded before the roster snapshot), "
          f"{(rookies['STATUS'] == 'undrafted_signed').sum()} undrafted free-agent signings.")

    team_capital = rookies.groupby("TEAM").agg(
        DRAFT_CAPITAL=("VALUE", "sum"),
        ROOKIE_COUNT=("PLAYER_ID", "count"),
        KNOWN_PICK_COUNT=("STATUS", lambda s: (s == "known_pick").sum()),
        BEST_KNOWN_PICK=("PICK", "min"),
    ).reset_index()

    vegas = pd.read_csv(VEGAS_PATH, encoding="utf-8")
    merged = vegas.merge(team_capital, on="TEAM", how="left")
    merged[["DRAFT_CAPITAL", "ROOKIE_COUNT", "KNOWN_PICK_COUNT"]] = merged[
        ["DRAFT_CAPITAL", "ROOKIE_COUNT", "KNOWN_PICK_COUNT"]
    ].fillna(0)

    merged = merged.sort_values("DRAFT_CAPITAL", ascending=False)
    out_cols = ["TEAM", "TEAM_NAME", "DRAFT_CAPITAL", "ROOKIE_COUNT", "KNOWN_PICK_COUNT",
                "BEST_KNOWN_PICK", "PROJECTED_WINS", "VEGAS_TOTAL", "DIFF"]
    merged[out_cols].to_csv(OUTPUT_PATH, index=False)

    pearson = merged["DRAFT_CAPITAL"].corr(merged["DIFF"], method="pearson")
    spearman = merged["DRAFT_CAPITAL"].corr(merged["DIFF"], method="spearman")
    print(f"\nDraft capital vs. (our projection - Vegas total): "
          f"Pearson r = {pearson:.3f}, Spearman rho = {spearman:.3f}")
    print(f"({len(merged)} teams; caveat above applies -- capital is a partial lower bound)")
    print(f"Wrote {OUTPUT_PATH}")

    print("\nTop 5 by draft capital:")
    print(merged[out_cols].head(5).to_string(index=False))
    print("\nBottom 5 by draft capital:")
    print(merged[out_cols].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
