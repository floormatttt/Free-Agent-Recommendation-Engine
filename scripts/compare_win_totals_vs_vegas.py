"""
Compares multi_year_team_profile_predictions.csv's projected 2026-27 win
totals against sportsbook season win-total lines (over/under + American
odds for each side, hardcoded below from what the user supplied -- no API
for this, it's a one-time market snapshot).

Converts each side's American odds to implied probability, then to a
vig-free "fair" over probability (OVER_IMPLIED / (OVER_IMPLIED +
UNDER_IMPLIED)) so the book's built-in edge doesn't get read as signal.

Output: data/2026_27_projections/multi_year_win_totals_vs_vegas.csv, sorted by
(our projection - Vegas total) descending -- positive means our model is
more bullish on that team than the market, negative means more bearish.

Usage:
    python scripts/compare_win_totals_vs_vegas.py
"""

from pathlib import Path

import pandas as pd

PREDICTIONS_PATH = Path("data/2026_27_projections/multi_year_team_profile_predictions.csv")
OUTPUT_PATH = Path("data/2026_27_projections/multi_year_win_totals_vs_vegas.csv")

# TEAM_NAME, total, over odds, under odds -- American odds as strings to
# preserve the +/- sign.
ODDS = [
    ("Atlanta Hawks", 44.5, "-105", "-125"),
    ("Boston Celtics", 49.5, "-120", "-105"),
    ("Brooklyn Nets", 24.5, "-105", "-120"),
    ("Charlotte Hornets", 37.5, "-105", "-120"),
    ("Chicago Bulls", 29.5, "-110", "-115"),
    ("Cleveland Cavaliers", 47.5, "-105", "-120"),
    ("Dallas Mavericks", 35.5, "+100", "-125"),
    ("Denver Nuggets", 48.5, "-105", "-120"),
    ("Detroit Pistons", 49.5, "-105", "-120"),
    ("Golden State Warriors", 39.5, "-105", "-120"),
    ("Houston Rockets", 46.5, "-110", "-120"),
    ("Indiana Pacers", 43.5, "-120", "-105"),
    ("Los Angeles Clippers", 28.5, "-110", "-115"),
    ("Los Angeles Lakers", 45.5, "-105", "-120"),
    ("Memphis Grizzlies", 28.5, "-110", "-115"),
    ("Miami Heat", 45.5, "-105", "-120"),
    ("Milwaukee Bucks", 24.5, "-120", "-105"),
    ("Minnesota Timberwolves", 48.5, "-110", "-115"),
    ("New Orleans Pelicans", 28.5, "-110", "-115"),
    ("New York Knicks", 51.5, "-115", "-110"),
    ("Oklahoma City Thunder", 60.5, "-105", "-120"),
    ("Orlando Magic", 44.5, "-105", "-120"),
    ("Philadelphia 76ers", 50.5, "-105", "-125"),
    ("Phoenix Suns", 39.5, "-105", "-120"),
    ("Portland Trail Blazers", 43.5, "-105", "-120"),
    ("Sacramento Kings", 21.5, "-115", "-110"),
    ("San Antonio Spurs", 60.5, "-105", "-120"),
    ("Toronto Raptors", 45.5, "-115", "-110"),
    ("Utah Jazz", 36.5, "-120", "-105"),
    ("Washington Wizards", 35.5, "-105", "-120"),
]

NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def american_to_prob(odds: str) -> float:
    o = int(odds)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def build_odds_df() -> pd.DataFrame:
    rows = [
        {"TEAM_NAME": name, "TEAM": NAME_TO_ABBR[name], "VEGAS_TOTAL": total,
         "OVER_ODDS": over, "UNDER_ODDS": under}
        for name, total, over, under in ODDS
    ]
    df = pd.DataFrame(rows)
    df["OVER_IMPLIED"] = df["OVER_ODDS"].apply(american_to_prob)
    df["UNDER_IMPLIED"] = df["UNDER_ODDS"].apply(american_to_prob)
    df["VIG"] = (df["OVER_IMPLIED"] + df["UNDER_IMPLIED"] - 1).round(4)
    df["OVER_FAIR_PROB"] = (df["OVER_IMPLIED"] / (df["OVER_IMPLIED"] + df["UNDER_IMPLIED"])).round(4)
    return df


def main():
    preds = pd.read_csv(PREDICTIONS_PATH)
    odds = build_odds_df()

    merged = preds.merge(odds, on="TEAM", how="inner")
    merged["DIFF"] = (merged["PROJECTED_WINS"] - merged["VEGAS_TOTAL"]).round(1)
    merged = merged.sort_values("DIFF", ascending=False)

    out_cols = ["TEAM", "TEAM_NAME", "PROJECTED_WINS", "VEGAS_TOTAL", "DIFF", "OVER_FAIR_PROB", "VIG"]
    merged[out_cols].to_csv(OUTPUT_PATH, index=False)

    mae = merged["DIFF"].abs().mean()
    corr = merged["PROJECTED_WINS"].corr(merged["VEGAS_TOTAL"])
    print(merged[["TEAM_NAME", "PROJECTED_WINS", "VEGAS_TOTAL", "DIFF"]].to_string(index=False))
    print(f"\nMean absolute diff: {mae:.2f} wins")
    print(f"Correlation (our projection vs. Vegas total): {corr:.3f}")
    print(f"Sum of totals -- Vegas: {merged['VEGAS_TOTAL'].sum():.1f}, ours: {merged['PROJECTED_WINS'].sum():.1f}")
    print(f"Average vig: {merged['VIG'].mean():.4f}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
