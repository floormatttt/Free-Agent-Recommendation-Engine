"""
Parses cap.md (a tab-separated salary cap history table) into a CSV, and
adds a SALARY_PCT_OF_CAP column to every data/salaries/salaries_<season>.csv
file, computed against that season's salary cap.

Usage:
    python scripts/parse_cap.py
"""

import csv
from pathlib import Path

import pandas as pd

SRC = Path("cap.md")
CAP_OUT = Path("data/salary_cap/cap_history.csv")
SALARY_DIR = Path("data/salaries")

FIELDS = [
    "SALARY_CAP",
    "LUXURY_TAX",
    "FIRST_APRON",
    "SECOND_APRON",
    "BAE",
    "NON_TAXPAYER_MLE",
    "TAXPAYER_MLE",
    "TEAM_ROOM_MLE",
]

HEADER_LINES = 9  # "Season" + the 8 column names in FIELDS


def to_num(token: str):
    token = token.strip()
    if token in ("N/A", ""):
        return None
    return int(token.replace("$", "").replace(",", ""))


def normalize_season(token: str) -> str:
    """"2034-2035" -> "2034-35", matching the rest of the repo's convention."""
    start, end = token.strip().split("-")
    return f"{start}-{end[-2:]}"


def parse_cap_history() -> pd.DataFrame:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[HEADER_LINES:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        row = {"SEASON": normalize_season(parts[0])}
        row.update(dict(zip(FIELDS, (to_num(p) for p in parts[1:]))))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("SEASON").reset_index(drop=True)
    return df


def add_pct_of_cap(cap_df: pd.DataFrame):
    cap_by_season = cap_df.set_index("SEASON")["SALARY_CAP"]
    for path in sorted(SALARY_DIR.glob("salaries_*.csv")):
        season = path.stem.replace("salaries_", "")
        if season not in cap_by_season.index or pd.isna(cap_by_season[season]):
            print(f"Skipping {path.name} -- no salary cap on file for {season}")
            continue
        salary_df = pd.read_csv(path, encoding="utf-8")
        salary_df["SALARY_PCT_OF_CAP"] = salary_df["SALARY"] / cap_by_season[season] * 100
        salary_df.to_csv(path, index=False, encoding="utf-8")
        print(f"Updated {path.name} with SALARY_PCT_OF_CAP (cap={cap_by_season[season]:,})")


def main():
    cap_df = parse_cap_history()
    CAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    cap_df.to_csv(CAP_OUT, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(cap_df)} seasons to {CAP_OUT}")

    add_pct_of_cap(cap_df)


if __name__ == "__main__":
    main()
