"""
Fixes a real confound in historical_free_agency_fit.py's "movers" list:
that script identifies offseason movers purely by comparing each player's
season-END team between two consecutive player_vectors_json snapshots
(season S vs. season S_next). A player who was traded MID-SEASON *during*
season S_next itself (e.g. Jarrett Allen, BKN -> CLE on 01/16/21, mid the
2020-21 season) still shows up as "team changed between S and S_next" in
that comparison, and gets misattributed as a Summer-S_next offseason
mover -- it's a real trade, just not one that happened in that offseason
window.

This pulls real transaction records (CommonTeamRoster's HOW_ACQUIRED,
the same field the 2026-27 page uses) for all 30 teams across the 7
season_next values historical_free_agency_fit.py covers (2019-20 ..
2025-26), and classifies each acquisition date as offseason (Jul-Sep,
before that season's games start) or in-season (Oct-Jun, i.e. a
mid-season trade) using the month alone -- deliberately not hardcoding
exact game-calendar boundaries from memory (several of these seasons had
irregular calendars: the 2019-20 bubble restart, the 2020-21 season
delayed to December), since a wrong memorized date is worse than a
simple, auditable Jul-Sep vs. Oct-Jun rule that only misclassifies
transactions in the few ambiguous late-June/early-July days around each
season's actual start.

Output: data/free_agency/midseason_trades_by_season.json, one entry per
season_next: {player_name: {team, how_acquired, is_midseason}} for every
roster player with a "Traded from" or "Signed on" HOW_ACQUIRED entry (not
just movers -- keeping the full pull makes this reusable and auditable).

Usage:
    python scripts/find_midseason_trades.py
"""
import json
import re
import time
from pathlib import Path

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as static_teams

from build_team_vectors import STANDARD_SEASONS

ROOT = Path(__file__).resolve().parent.parent
DATE_PATTERN = re.compile(r"(\d{2})/(\d{2})/(\d{2})$")
OFFSEASON_MONTHS = {7, 8, 9}  # Jul-Sep; see module docstring for why not exact game-calendar dates


def classify_month(how_acquired: str):
    m = DATE_PATTERN.search(str(how_acquired))
    if not m:
        return None
    month = int(m.group(1))
    return month


def main():
    teams = static_teams.get_teams()
    season_nexts = STANDARD_SEASONS[1:]  # the 7 cycles historical_free_agency_fit.py covers

    out = {}
    for season in season_nexts:
        print(f"Pulling all 30 team rosters for {season}...")
        season_data = {}
        for t in teams:
            for attempt in range(3):
                try:
                    df = commonteamroster.CommonTeamRoster(team_id=t["id"], season=season).get_data_frames()[0]
                    break
                except Exception as e:
                    print(f"  {t['abbreviation']} attempt {attempt+1} failed: {e}")
                    time.sleep(2)
            else:
                print(f"  {t['abbreviation']}: giving up after 3 attempts")
                continue
            for _, row in df.iterrows():
                how = row["HOW_ACQUIRED"]
                if not isinstance(how, str):
                    continue
                is_trade = how.startswith("Traded from")
                is_signed = how.startswith("Signed on")
                if not (is_trade or is_signed):
                    continue
                month = classify_month(how)
                if month is None:
                    continue
                season_data[row["PLAYER"]] = {
                    "team": t["abbreviation"], "how_acquired": how,
                    "is_midseason": month not in OFFSEASON_MONTHS,
                }
        n_mid = sum(1 for v in season_data.values() if v["is_midseason"])
        print(f"  {season}: {len(season_data)} dated acquisitions, {n_mid} classified mid-season")
        out[season] = season_data

    out_path = ROOT / "data" / "free_agency" / "midseason_trades_by_season.json"
    out_path.write_text(json.dumps(out, indent=None), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
