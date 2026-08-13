"""
Two analyses on top of the win%-weighted need-reduction fit scores:

  1. Player global average fit score (mean fit-score across all evaluable
     teams -- "how broadly useful is this player, independent of
     destination") vs. predicted contract RANGE, not a single point value
     -- for every offseason cycle this repo has fit scores for: the
     current 2026-27 class (data/free_agency/free_agent_team_fit_rankings.csv)
     and the 7 historical cycles (data/free_agency/historical_fit_data.json,
     Summer 2019-2025). The underlying salary model (data/salary_prediction/
     contract_model_v3_predictions.csv) only outputs a point
     PREDICTED_SALARY_PCT_OF_CAP per (player, TARGET_SEASON) -- it has no
     notion of its own uncertainty. The range here is empirical: the
     model's real historical residuals (ACTUAL minus PREDICTED, across all
     754 backtested player-seasons, pooled across every season -- not
     re-derived per season, since a single season's residuals alone would
     be too few to give a stable quantile) have a 25th percentile of -2.07
     and a 75th of +1.65 percentage-points-of-cap, so LOW/HIGH = each
     player's point prediction shifted by those two numbers. Dollar
     conversion uses that SPECIFIC season's own salary cap (data/
     salary_cap/cap_history.csv), not 2026-27's, so historical ranges
     reflect what that season's cap actually was. Matched by name after
     accent-stripping normalization; not every free agent has a contract
     prediction (the model's training population skews toward players
     with an established NBA statistical record, and pre-2023 seasons
     also have fewer scoreable free agents to begin with -- see
     historical_free_agency_fit.py), so each cycle only covers the subset
     that matched.

  2. Team average fit score (2026-27 only, mean fit-score across all 47
     free agents) -- how much the average free agent would help this
     team, independent of who specifically signs. High = broadly needy/
     easy to upgrade; low = already strong, harder to move the needle.

Output: data/free_agency/player_fit_vs_contract.csv (SEASON column, all
        8 cycles stacked), data/free_agency/team_avg_fit_score.csv

Usage:
    python scripts/analyze_fit_vs_contract.py
"""
import io
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent


def normalize_name(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower().strip()


def load_2026_player_avg() -> pd.DataFrame:
    rankings = pd.read_csv(ROOT / "data" / "free_agency" / "free_agent_team_fit_rankings.csv")
    player_avg = rankings.groupby("PLAYER_NAME")["WEIGHTED_PCT_NEED_INCREASE"].mean().reset_index()
    player_avg.columns = ["PLAYER_NAME", "AVG_FIT_SCORE"]
    return player_avg, rankings


def load_historical_player_avgs() -> dict:
    """season_next -> player-avg-fit-score DataFrame, from the compact flat
    rows historical_free_agency_fit.py writes (player_idx, team_idx, pct,
    team_z, is_real per row)."""
    hist = json.loads((ROOT / "data" / "free_agency" / "historical_fit_data.json").read_text(encoding="utf-8"))
    out = {}
    for season_next, cyc in hist.items():
        names = cyc["player_names"]
        sums = [0.0] * len(names)
        counts = [0] * len(names)
        for row in cyc["rows"]:
            p_idx, _, pct = row[0], row[1], row[2]
            sums[p_idx] += pct
            counts[p_idx] += 1
        avg = pd.DataFrame({
            "PLAYER_NAME": names,
            "AVG_FIT_SCORE": [s / c if c else 0.0 for s, c in zip(sums, counts)],
        })
        out[season_next] = avg
    return out


def compute_contract_range(player_avg: pd.DataFrame, season: str, contracts: pd.DataFrame,
                            resid_lo: float, resid_hi: float, cap_by_season: pd.Series) -> pd.DataFrame:
    player_avg = player_avg.copy()
    player_avg["NAME_KEY"] = player_avg["PLAYER_NAME"].apply(normalize_name)

    cs = contracts[contracts["TARGET_SEASON"] == season].copy()
    if cs.empty or season not in cap_by_season.index:
        return pd.DataFrame()
    cap = cap_by_season.loc[season]
    cs["NAME_KEY"] = cs["PLAYER_NAME"].apply(normalize_name)
    cs["PRED_LOW_PCT"] = (cs["PREDICTED_SALARY_PCT_OF_CAP"] + resid_lo).clip(lower=0)
    cs["PRED_HIGH_PCT"] = (cs["PREDICTED_SALARY_PCT_OF_CAP"] + resid_hi).clip(lower=0)
    cs["PRED_DOLLARS"] = cs["PREDICTED_SALARY_PCT_OF_CAP"] / 100 * cap
    cs["PRED_LOW_DOLLARS"] = cs["PRED_LOW_PCT"] / 100 * cap
    cs["PRED_HIGH_DOLLARS"] = cs["PRED_HIGH_PCT"] / 100 * cap
    cs["ACTUAL_DOLLARS"] = cs["ACTUAL_SALARY_PCT_OF_CAP"] / 100 * cap

    merged = player_avg.merge(
        cs[["NAME_KEY", "PLAYER_NAME", "PREDICTED_SALARY_PCT_OF_CAP", "PRED_LOW_PCT", "PRED_HIGH_PCT",
            "PRED_DOLLARS", "PRED_LOW_DOLLARS", "PRED_HIGH_DOLLARS", "ACTUAL_SALARY_PCT_OF_CAP",
            "ACTUAL_DOLLARS", "RESIDUAL_PP"]],
        on="NAME_KEY", suffixes=("", "_contract"),
    )
    merged.insert(0, "SEASON", season)
    return merged.drop(columns=["NAME_KEY"])


def main():
    contracts = pd.read_csv(ROOT / "data" / "salary_prediction" / "contract_model_v3_predictions.csv")
    resid_lo, resid_hi = contracts["RESIDUAL_PP"].quantile([0.25, 0.75])
    print(f"Historical residual 25th/75th percentile: {resid_lo:+.2f} / {resid_hi:+.2f} pp-of-cap "
          f"(n={len(contracts)} backtested player-seasons, pooled across all seasons)")
    cap_by_season = pd.read_csv(ROOT / "data" / "salary_cap" / "cap_history.csv").set_index("SEASON")["SALARY_CAP"]

    player_avg_2026, rankings = load_2026_player_avg()
    historical_avgs = load_historical_player_avgs()

    all_rows = []
    for season, player_avg in [("2026-27", player_avg_2026), *historical_avgs.items()]:
        merged = compute_contract_range(player_avg, season, contracts, resid_lo, resid_hi, cap_by_season)
        print(f"  {season}: matched {len(merged)} of {len(player_avg)} free agents to a contract prediction")
        all_rows.append(merged)

    full = pd.concat(all_rows, ignore_index=True)
    full = full.sort_values(["SEASON", "AVG_FIT_SCORE"], ascending=[True, False])
    full.round(3).to_csv(ROOT / "data" / "free_agency" / "player_fit_vs_contract.csv", index=False)

    c26 = full[full["SEASON"] == "2026-27"]
    r_pred, p_pred = stats.pearsonr(c26["AVG_FIT_SCORE"], c26["PREDICTED_SALARY_PCT_OF_CAP"])
    r_act, p_act = stats.pearsonr(c26["AVG_FIT_SCORE"], c26["ACTUAL_SALARY_PCT_OF_CAP"])
    print(f"\n2026-27 only -- Pearson r(avg fit score, PREDICTED contract %-of-cap) = {r_pred:.3f}  (p={p_pred:.4f})")
    print(f"2026-27 only -- Pearson r(avg fit score, ACTUAL contract %-of-cap)    = {r_act:.3f}  (p={p_act:.4f})")

    print("\n=== 2026-27: highest avg fit score ===")
    print(c26.head(10)[["PLAYER_NAME", "AVG_FIT_SCORE", "PRED_LOW_DOLLARS", "PRED_DOLLARS", "PRED_HIGH_DOLLARS", "ACTUAL_DOLLARS"]]
          .round(0).to_string(index=False))

    # ---- team avg fit score (2026-27 only) ----
    team_avg = rankings.groupby("TEAM")["WEIGHTED_PCT_NEED_INCREASE"].mean().reset_index()
    team_avg.columns = ["TEAM", "AVG_FIT_SCORE"]
    team_avg = team_avg.sort_values("AVG_FIT_SCORE", ascending=False)
    team_avg.round(3).to_csv(ROOT / "data" / "free_agency" / "team_avg_fit_score.csv", index=False)

    print(f"\nWrote data/free_agency/player_fit_vs_contract.csv ({len(full)} rows across "
          f"{full['SEASON'].nunique()} seasons), data/free_agency/team_avg_fit_score.csv")


if __name__ == "__main__":
    main()
