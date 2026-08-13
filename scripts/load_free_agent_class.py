"""
Parses data/free_agents/<season>.txt -- the real free-agent class for the
offseason following <season> (e.g. 18-19.txt = players who hit free
agency in Summer 2019, heading into 2019-20). Source format is two lines
per player:

    PlayerName\tPos\tYrs\t$Value\t$AAV\t
    Official   (or "Agreed", both treated as real signings)

with a few header/total lines to skip. From/To team columns exist in the
header but are never populated in this scrape, so this only recovers
player names -- their season-of-departure team and real destination are
looked up separately, by name, against player_vectors_json.

Why this matters versus inferring "movers" by diffing two seasons' roster
snapshots (historical_free_agency_fit.py's original approach): this list
is a real transactional category -- a player only appears here if they
actually signed a new contract as a free agent. That fixes two things a
roster diff can't:
  1. Same-team re-signings are now visible (they're in this list even
     though their team didn't change) -- the roster-diff approach missed
     these entirely.
  2. Mid-season trades are automatically excluded -- a traded player
     keeps their existing contract, so they never appear in a free-agent
     signing list. No separate month-based date classification needed
     (see find_midseason_trades.py, which this approach supersedes for
     historical cycles).

Usage (as a library):
    from load_free_agent_class import load_free_agent_names
    names = load_free_agent_names("2018-19")  # -> Summer 2019 signees
"""
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAYER_LINE = re.compile(r"^([^\t]+)\t[A-Za-z/]+\t\d+\t\$[\d,]+\t\$[\d,]+\t?$")
VALID_STATUSES = {"Official", "Agreed"}


def normalize_name(name: str) -> str:
    """Strip accents/punctuation for matching across data sources that
    encode the same player differently (e.g. curly vs. straight
    apostrophe, or dropped diacritics)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", "", ascii_name.lower()).strip()


def load_free_agent_names(season: str) -> list[str]:
    """season is the season the free agent CLASS is coming out of, e.g.
    "2018-19" for the class hitting free agency in Summer 2019. Returns
    the raw (non-normalized) player names as they appear in the file."""
    short = f"{season[2:4]}-{season[5:7]}"
    path = ROOT / "data" / "free_agents" / f"{short}.txt"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    names = []
    for i, line in enumerate(lines):
        m = PLAYER_LINE.match(line)
        if not m:
            continue
        if i + 1 >= len(lines) or lines[i + 1] not in VALID_STATUSES:
            continue
        names.append(m.group(1))
    return names


if __name__ == "__main__":
    for season in ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]:
        names = load_free_agent_names(season)
        print(f"{season}: {len(names)} free agents")
