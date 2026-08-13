"""
Parses the multi-year salary-ranking dumps at the repo root (26-27.txt
through 30-31.txt -- same rank/name/"TEAM, POS"/salary format as
.claude/skills/parse-nba-salary-list) into a single wide CSV: one row per
player, one column per season's salary.

A player who signs elsewhere or falls off guaranteed money in a later
season simply doesn't appear in that season's source file; their cell for
that season is left blank rather than 0, since 0 would misleadingly read as
"guaranteed $0" instead of "not on a guaranteed deal that season, per this
source."

TEAM is taken from the earliest season a player appears in (their current
team), not re-checked against later seasons -- a handful of players are
projected to be on a different team by a later season's list (trades,
free agency), which this format has no way to flag inline.

Usage:
    python scripts/build_multi_year_contracts.py
"""

import csv
import re
from pathlib import Path

ROOT = Path(".")
SEASONS = ["26-27", "27-28", "28-29", "29-30", "30-31"]
OUTPUT = Path("multi_year_contracts.csv")


def parse_season_file(path: Path) -> list[dict]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) % 4 != 0:
        raise ValueError(f"{path}: expected a multiple of 4 lines, got {len(raw_lines)}")

    rows = []
    for i in range(0, len(raw_lines), 4):
        name = raw_lines[i + 1].strip()
        team_pos = raw_lines[i + 2].strip()
        salary_token = raw_lines[i + 3].strip()

        team = team_pos.split(",", 1)[0].strip()
        salary = int(re.sub(r"[^\d]", "", salary_token))

        rows.append({"PLAYER_NAME": name, "TEAM": team, "SALARY": salary})
    return rows


def build() -> list[dict]:
    by_player: dict[str, dict] = {}
    for season in SEASONS:
        for row in parse_season_file(ROOT / f"{season}.txt"):
            name = row["PLAYER_NAME"]
            if name not in by_player:
                by_player[name] = {"PLAYER_NAME": name, "TEAM": row["TEAM"]}
            by_player[name][season] = row["SALARY"]

    return list(by_player.values())


def main():
    players = build()
    # Group by team (26-27 salary, descending, within each team; players with
    # no 26-27 entry -- e.g. a rookie whose deal starts later -- sort last
    # within their team, by name).
    players.sort(key=lambda p: (p["TEAM"], -(p.get("26-27") or -1), p["PLAYER_NAME"]))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["PLAYER_NAME", "TEAM"] + SEASONS
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in players:
            writer.writerow(row)

    counts = {s: sum(1 for p in players if s in p) for s in SEASONS}
    print(f"Wrote {len(players)} players to {OUTPUT}")
    print("Players with a contract per season:", counts)


if __name__ == "__main__":
    main()
