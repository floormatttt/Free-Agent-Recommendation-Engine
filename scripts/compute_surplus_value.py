"""
Surplus value: Realized Impact - Expected Impact Given Salary, checked against
the season AFTER each free agent signed -- not the contract's own first
season -- across every free-agency cycle where that following season has
actually been played.

Uses this repo's from-scratch BPM/VORP reconstruction (compute_bpm_vorp.py)
as the "actual impact" ground truth -- deliberately external to this repo's
own roster-vector/need/win% models, so scoring a contract's value this way
doesn't grade the model against itself (see docs/free_agency_2026.md's
Surplus/Regret eval design discussion). Impact is BPM, not VORP -- this
script has flip-flopped between the two at the user's request (VORP, then
BPM, then VORP, now BPM again); this is the current state. BPM is a rate
(points per 100 possessions, already anchored to league average), playing-
time-blind: a player who was excellent in limited minutes and a player who
was equally excellent across a full starter's workload score the same here.
VORP would have separated them (VORP = (BPM+2.0) * %possessions * games/82)
-- it answers "how much value did this player add to the team," where BPM
answers the narrower "how good were they when they played."

Tried and reverted: applying the same minutes-reliability gate built for the
contract model (`apply_minutes_gate.py`, trailing-3yr-minutes shrink toward
0) directly to the impact metric. Undone at the user's request -- values
below use raw, un-shrunk numbers. The market-curve fit still restricts to
GP >= 10 that season (LEAGUE_CURVE_MIN_GP) to keep single-game outliers from
distorting the regression; free agents below that threshold are flagged
(LOW_SAMPLE) rather than excluded or shrunk.

Two things:
  1. League-wide market curve, fit fresh per season (own-season BPM vs.
     own-season salary, min. 10 GP) for every season with realized BPM data
     (2019-20 through 2025-26). This is a same-season snapshot of the
     market's pay-for-production relationship, used for the illustrative
     scatter/context and as the source of each FOLLOWING season's curve
     below -- not itself the free-agent eval.
  2. Model check: every free agent in contract_model_v3_predictions.csv whose
     TARGET_SEASON (the season their new contract covers) has a *played
     following season* -- i.e. TARGET_SEASON 2019-20 through 2024-25, checked
     against realized BPM in 2020-21 through 2025-26 respectively.
     TARGET_SEASON 2025-26 is excluded here: its following season, 2026-27,
     hasn't been played, so there's no "did it pay off a year later" data yet
     (same reasoning that already excludes the current, still-unsigned
     2026-27 free-agent class). Real/predicted cap% still come from
     TARGET_SEASON (what the contract actually cost or was forecast to cost
     in its first year); SURPLUS is scored against the FOLLOWING season's own
     market curve, since that's the season whose BPM scale is being used.

"Winning" is who projected the player's value closer, not who priced them
cheaper. EXPECTED_BPM_NBA/EXPECTED_BPM_MODEL are what the market curve says
a player at the real/predicted cap% should produce; MODEL_CLOSER is true
when |BPM - EXPECTED_BPM_MODEL| < |BPM - EXPECTED_BPM_NBA| -- i.e. the
model's implied production forecast landed nearer the player's actual
following-season BPM than the real contract's implied forecast did. This
replaces an earlier win condition (SURPLUS_MODEL > SURPLUS_NBA) that, because
the impact metric is identical in both surplus terms, algebraically reduced
to nothing more than "was PREDICTED_SALARY_PCT_OF_CAP < ACTUAL_SALARY_PCT_OF_
CAP" -- a pure pricing-bias check with no dependence on the player's real
production at all. MODEL_CLOSER doesn't have that flaw.

Known limitation carried over from compute_bpm_vorp.py: BPM itself is
currently inflated (~1.1-1.6x, audited and documented in the BPM/VORP
artifact) by a still-open position-estimate distortion for stat-outlier
guards, on top of a partially-corrected team-shooting-context adjustment.
That inflation is roughly proportional across players, so it should mostly
wash out of *relative* surplus rankings, but the absolute BPM-per-cap-
percentage-point regression slope/intercept per season should not be read as
calibrated to real-world BPM scale until that's fixed.

Additional limitation specific to the following-season check: cap% for a
multi-year contract can drift between its first and second season (cap
growth, and most deals aren't flat), but ACTUAL_SALARY_PCT_OF_CAP /
PREDICTED_SALARY_PCT_OF_CAP are only available for TARGET_SEASON (the
contract's first year) -- so a player's price is being compared against a
different season's performance without a matching following-season price.

Rookie-scale contracts are filtered out entirely. They're fixed by draft
slot per the CBA, not market-negotiated, so a cheap rookie deal isn't the
market "finding value" -- it's the rule doing what it always does, and
including these rows would inflate every surplus number with contracts that
were never a market bet in the first place. Reuses the same threshold
already established for `contract_model_v3`'s training filter
(`docs/contract_model.md`'s "Rookie-scale contract contamination" section,
verified there against real data): `ROOKIE_CONTRACT` = `EXPERIENCE_YRS <= 4`,
sourced from `data/salary_prediction/player_season_features.csv`.

Output: data/player_value/surplus_value_<season>.csv (own-season league-wide
        market curve, one file per season with realized BPM)
        data/player_value/surplus_value_model_check_all_seasons.csv (every
        free agent whose following season has been played)

Usage:
    python scripts/compute_surplus_value.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from load_free_agent_class import normalize_name

ROOT = Path(__file__).resolve().parent.parent
PLAYER_VALUE_DIR = ROOT / "data" / "player_value"

# Every season with a realized BPM/VORP file (compute_bpm_vorp.py has been run
# for each of these).
REALIZED_SEASONS = [
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
LEAGUE_CURVE_MIN_GP = 10


def next_season(season: str) -> str:
    start = int(season[:4])
    return f"{start + 1}-{str(start + 2)[-2:]}"


def load_rookie_flags() -> pd.DataFrame:
    """(PLAYER_ID, SEASON) -> ROOKIE_CONTRACT, from the same EXPERIENCE_YRS <=
    4 threshold contract_model_v3 already established (see module docstring).
    Covers 2018-19 through 2025-26; a player/season not present here is
    treated as not rookie-scale (this file's own coverage of the player-
    vector pool, not evidence of rookie status)."""
    feats = pd.read_csv(ROOT / "data" / "salary_prediction" / "player_season_features.csv")
    return feats[["PLAYER_ID", "SEASON", "ROOKIE_CONTRACT"]].drop_duplicates(["PLAYER_ID", "SEASON"])


def fit_season_curve(season: str, rookie_flags: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    bpm = pd.read_csv(PLAYER_VALUE_DIR / f"bpm_vorp_{season}.csv")
    salaries = pd.read_csv(ROOT / "data" / "salaries" / f"salaries_{season}.csv")

    bpm = bpm.copy()
    bpm["NAME_KEY"] = bpm["PLAYER_NAME"].apply(normalize_name)
    salaries = salaries.copy()
    salaries["NAME_KEY"] = salaries["PLAYER_NAME"].apply(normalize_name)
    salaries = salaries.drop_duplicates("NAME_KEY")

    df = bpm.merge(salaries[["NAME_KEY", "SALARY", "SALARY_PCT_OF_CAP"]], on="NAME_KEY", how="inner")

    season_flags = rookie_flags[rookie_flags["SEASON"] == season][["PLAYER_ID", "ROOKIE_CONTRACT"]]
    df = df.merge(season_flags, on="PLAYER_ID", how="left")
    n_before = len(df)
    df = df[df["ROOKIE_CONTRACT"] != True].copy()  # noqa: E712 -- keeps NaN (unmatched) rows
    print(f"  {season}: dropped {n_before - len(df)} rookie-scale player-seasons "
          f"(EXPERIENCE_YRS <= 4) of {n_before} matched")

    fit_df = df[df["GP"] >= LEAGUE_CURVE_MIN_GP].copy()

    slope, intercept = np.polyfit(fit_df["SALARY_PCT_OF_CAP"], fit_df["BPM"], 1)
    df["EXPECTED_BPM"] = slope * df["SALARY_PCT_OF_CAP"] + intercept
    df["SURPLUS_NBA"] = df["BPM"] - df["EXPECTED_BPM"]
    return df, slope, intercept


def main():
    rookie_flags = load_rookie_flags()

    curves = {}
    all_bpm = {}
    for season in REALIZED_SEASONS:
        df, slope, intercept = fit_season_curve(season, rookie_flags)
        curves[season] = (slope, intercept)
        all_bpm[season] = df
        df = df.copy()
        df["SEASON"] = season

        out_cols = ["SEASON", "PLAYER_ID", "PLAYER_NAME", "GP", "MIN_TOTAL", "BPM",
                    "SALARY", "SALARY_PCT_OF_CAP", "EXPECTED_BPM", "SURPLUS_NBA"]
        league = df[out_cols].round(4).sort_values("SURPLUS_NBA", ascending=False)
        league_path = PLAYER_VALUE_DIR / f"surplus_value_{season}.csv"
        league.to_csv(league_path, index=False)
        print(f"{season}: own-season curve BPM = {slope:.5f} * cap% + {intercept:.5f}  "
              f"({len(league)} players matched to a salary)")

    # ---- model check: free agents checked against the FOLLOWING season ----
    evaluable_targets = [s for s in REALIZED_SEASONS if next_season(s) in REALIZED_SEASONS]
    skipped = [s for s in REALIZED_SEASONS if s not in evaluable_targets]
    print(f"\nEvaluable TARGET_SEASONs (following season already played): {evaluable_targets}")
    print(f"Skipped -- no played following season yet: {skipped}")

    contracts = pd.read_csv(ROOT / "data" / "salary_prediction" / "contract_model_v3_predictions.csv")
    contracts = contracts[contracts["TARGET_SEASON"].isin(evaluable_targets)].copy()
    contracts = contracts.merge(
        rookie_flags.rename(columns={"SEASON": "TARGET_SEASON"}), on=["PLAYER_ID", "TARGET_SEASON"], how="left")
    n_before = len(contracts)
    contracts = contracts[contracts["ROOKIE_CONTRACT"] != True].copy()  # noqa: E712
    print(f"\nDropped {n_before - len(contracts)} rookie-scale free-agent signings "
          f"(EXPERIENCE_YRS <= 4 in their signed-for season) of {n_before}")

    rows = []
    for target_season, group in contracts.groupby("TARGET_SEASON"):
        following_season = next_season(target_season)
        bpm = all_bpm[following_season][["PLAYER_ID", "PLAYER_NAME", "GP", "MIN_TOTAL", "BPM"]]
        slope, intercept = curves[following_season]
        merged = group.merge(bpm, on="PLAYER_ID", how="inner", suffixes=("", "_BPM"))
        merged["SEASON"] = target_season
        merged["FOLLOWING_SEASON"] = following_season
        merged["EXPECTED_BPM_NBA"] = slope * merged["ACTUAL_SALARY_PCT_OF_CAP"] + intercept
        merged["SURPLUS_NBA"] = merged["BPM"] - merged["EXPECTED_BPM_NBA"]
        merged["EXPECTED_BPM_MODEL"] = slope * merged["PREDICTED_SALARY_PCT_OF_CAP"] + intercept
        merged["SURPLUS_MODEL"] = merged["BPM"] - merged["EXPECTED_BPM_MODEL"]
        merged["MODEL_CLOSER"] = merged["SURPLUS_MODEL"].abs() < merged["SURPLUS_NBA"].abs()
        merged["LOW_SAMPLE"] = merged["GP"] < LEAGUE_CURVE_MIN_GP
        rows.append(merged)
        print(f"{target_season} (checked against {following_season}): matched {len(merged)} of {len(group)} "
              f"free agents to a following-season realized BPM "
              f"({merged['LOW_SAMPLE'].sum()} flagged low-sample, <{LEAGUE_CURVE_MIN_GP} GP)")

    fa_all = pd.concat(rows, ignore_index=True)
    fa_cols = ["SEASON", "FOLLOWING_SEASON", "PLAYER_ID", "PLAYER_NAME", "GP", "BPM",
               "ACTUAL_SALARY_PCT_OF_CAP", "PREDICTED_SALARY_PCT_OF_CAP",
               "SURPLUS_NBA", "SURPLUS_MODEL", "MODEL_CLOSER", "LOW_SAMPLE"]
    fa_out = fa_all[fa_cols].round(4).sort_values(["SEASON", "SURPLUS_NBA"], ascending=[True, False])
    fa_path = PLAYER_VALUE_DIR / "surplus_value_model_check_all_seasons.csv"
    fa_out.to_csv(fa_path, index=False)
    print(f"\nWrote {len(fa_out)} rows (every free agent with a played following season) to {fa_path}")

    print("\n=== Per-season summary (stable sample, GP >= 10 in following season only) ===")
    stable = fa_out[~fa_out["LOW_SAMPLE"]]
    summary = stable.groupby("SEASON").apply(
        lambda g: pd.Series({
            "N": len(g),
            "MEAN_SURPLUS_NBA": g["SURPLUS_NBA"].mean(),
            "MEAN_SURPLUS_MODEL": g["SURPLUS_MODEL"].mean(),
            "MODEL_CLOSER_PCT": g["MODEL_CLOSER"].mean() * 100,
            "MAE_NBA": g["SURPLUS_NBA"].abs().mean(),
            "MAE_MODEL": g["SURPLUS_MODEL"].abs().mean(),
        }), include_groups=False
    )
    print(summary.round(3).to_string())
    print(f"\nPooled MODEL_CLOSER (per-signing head-to-head): {stable['MODEL_CLOSER'].mean() * 100:.1f}% "
          f"of {len(stable)} ({stable['MODEL_CLOSER'].sum()} of {len(stable)})")

    # ---- aggregate closeness: summed/mean absolute forecast error, not a
    # per-signing win count -- "in total, whose forecast missed by less" ----
    mae_nba = stable["SURPLUS_NBA"].abs().mean()
    mae_model = stable["SURPLUS_MODEL"].abs().mean()
    sum_nba = stable["SURPLUS_NBA"].abs().sum()
    sum_model = stable["SURPLUS_MODEL"].abs().sum()
    print(f"\nAggregate closeness (mean absolute forecast error, {len(stable)} signings):")
    print(f"  Real contracts: MAE {mae_nba:.3f}  (sum of |error| = {sum_nba:.1f})")
    print(f"  Model prices:   MAE {mae_model:.3f}  (sum of |error| = {sum_model:.1f})")
    closer = "Real contracts" if mae_nba < mae_model else "Model prices"
    print(f"  {closer} closer in aggregate")


if __name__ == "__main__":
    main()
