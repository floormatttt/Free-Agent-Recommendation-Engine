"""
Assembles the finalized per-player vector: the 9 playtype scores from
build_playtype_vectors.py plus the rim defense, perimeter defense, and
rebounding scores from build_defense_vectors.py. All 12 categories from
player.md are now implemented.

Saved as JSON, keyed by player, with each player's vector itself a
key-value mapping of category -> score.

Usage:
    python scripts/build_player_vectors.py --season 2025-26
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
FULL_VECTOR_DIR = DATA_DIR / "player_vectors_full"
JSON_VECTOR_DIR = DATA_DIR / "player_vectors_json"

PLAY_TYPES = [
    "ISOLATION",
    "PNR_HANDLER",
    "PNR_ROLLMAN",
    "POST_UP",
    "SPOT_UP",
    "HANDOFF",
    "CUT",
    "OFF_SCREEN",
    "PUT_BACKS",
]

DEFENSE_CATEGORIES = {
    "RIM_DEFENSE": "RIM_BLOCK",
    "PERIMETER_DEFENSE": "PERIMETER_D_BLOCK",
    "REBOUNDING": "REB_BLOCK",
}


def build(season: str) -> dict:
    playtype_src = FULL_VECTOR_DIR / f"player_playtype_vectors_{season}.csv"
    defense_src = FULL_VECTOR_DIR / f"player_defense_vectors_{season}.csv"
    if not playtype_src.exists():
        raise FileNotFoundError(
            f"{playtype_src} not found -- run build_playtype_vectors.py --season {season} first"
        )
    if not defense_src.exists():
        raise FileNotFoundError(
            f"{defense_src} not found -- run build_defense_vectors.py --season {season} first"
        )
    playtype_df = pd.read_csv(playtype_src, encoding="utf-8")
    defense_df = pd.read_csv(defense_src, encoding="utf-8").set_index("PLAYER_ID")

    players = {}
    for _, row in playtype_df.iterrows():
        pid = row["PLAYER_ID"]
        vector = {pt: row[f"{pt}_SCORE"] for pt in PLAY_TYPES}

        if pid in defense_df.index:
            defense_row = defense_df.loc[pid]
            for category, source_col in DEFENSE_CATEGORIES.items():
                vector[category] = defense_row[source_col]
        else:
            for category in DEFENSE_CATEGORIES:
                vector[category] = None

        players[str(int(pid))] = {
            "PLAYER_NAME": row["PLAYER_NAME"],
            "TEAM": row["TEAM"],
            "SEASON": season,
            "VECTOR": vector,
        }

    return players


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="e.g. 2025-26")
    parser.add_argument("--out-dir", default=str(JSON_VECTOR_DIR))
    args = parser.parse_args()

    players = build(args.season)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"player_vectors_{args.season}.json"
    out_path.write_text(json.dumps(players, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(players)} player vectors to {out_path}")


if __name__ == "__main__":
    main()
