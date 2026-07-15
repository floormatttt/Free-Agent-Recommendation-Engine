"""
Parses a raw NBA salary-list text dump into a single CSV with one row per
player.

Expects the source file to repeat fixed groups of 4 lines per player:
    <rank, or blank if tied with the previous rank>
    <player name>
    <TEAM, POSITION>
    <$salary>

Usage:
    python parse_salaries.py <input.txt> <output.csv>
"""

import argparse
import csv
import re
from pathlib import Path


def parse(src: Path) -> list[dict]:
    raw_lines = src.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) % 4 != 0:
        raise ValueError(f"expected a multiple of 4 lines, got {len(raw_lines)}")

    rows = []
    last_rank = None
    for i in range(0, len(raw_lines), 4):
        rank_token = raw_lines[i].strip()
        name = raw_lines[i + 1].strip()
        team_pos = raw_lines[i + 2].strip()
        salary_token = raw_lines[i + 3].strip()

        rank = int(rank_token) if rank_token else last_rank
        last_rank = rank

        team, position = [part.strip() for part in team_pos.split(",", 1)]
        salary = int(re.sub(r"[^\d]", "", salary_token))

        rows.append(
            {
                "RANK": rank,
                "PLAYER_NAME": name,
                "TEAM": team,
                "POSITION": position,
                "SALARY": salary,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="raw salary list text file")
    parser.add_argument("output", type=Path, help="destination CSV path")
    args = parser.parse_args()

    rows = parse(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["RANK", "PLAYER_NAME", "TEAM", "POSITION", "SALARY"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} players to {args.output}")


if __name__ == "__main__":
    main()
