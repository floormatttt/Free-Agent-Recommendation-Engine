"""
Shifts each player's contract PRICE within their existing LOW/PRED/HIGH band
(player_fit_vs_contract.csv, see analyze_fit_vs_contract.py) depending on
which team is asking -- instead of one number per player, this gives one
number per (player, team).

This is a labeled heuristic, not a fitted model. Before building it, the
obvious approach was tried and rejected: regress the model's historical
RESIDUAL_PP (ACTUAL - PREDICTED) against TEAM_Z_SCORE at each free agent's
real signing team (826 historical (player, real-team) pairs across the 7
`historical_fit_data.json` cycles, 660 matched to a contract prediction).
Pooled, that regression looks real (r=0.114, p=0.003, slope +0.521pp per SD
of TEAM_Z -- fit correlates with a price *premium*, not a hometown
discount) -- but a leave-one-cycle-out check breaks it: the slope's sign
flips across 3 of the 7 cycles, and only one single cycle (2023-24,
r=0.392) is doing almost all of the pooled work; a slope fit on the other 6
cycles predicts the held-out cycle's residuals with an out-of-fold
correlation that ranges from -0.10 to +0.39 with no consistent direction.
That's noise, not signal -- there isn't a historically validated
coefficient to calibrate a shift from.

So instead of a fitted slope, this applies a hand-specified, saturating
squash of TEAM_Z_SCORE into a position within the *existing* (already
validated, from real residual quantiles) LOW/PRED/HIGH band:

    position = clip(TEAM_Z_SCORE / Z_SCALE, -1, 1)
    FIT_ADJUSTED_PCT = PRED + position * (HIGH - PRED)   if position >= 0
                      = PRED + position * (PRED - LOW)    if position <  0

TEAM_Z_SCORE = 0 (average fit among that player's evaluated teams) recovers
the original PRED exactly. Z_SCALE = 2.0 means a team at +-2 standard
deviations of fit for that player (a genuinely extreme fit, good or bad)
places the price at the band's edge; beyond that it saturates rather than
extrapolating past the empirically-grounded LOW/HIGH. This is a directional
design choice -- "better team-specific fit should plausibly command more of
a team's willingness to pay, within the range the model already considers
plausible" -- not a statistically demonstrated effect, and should be
labeled as such wherever it's shown.

Team-fit source: 2026-27 uses free_agent_team_fit_rankings.csv's
TEAM_Z_SCORE directly (already standardized within each team's 47-player
distribution -- see free_agency_2026.md's per-player team-fit ranking
section). The 7 historical cycles use historical_fit_data.json's per-row
TEAM_Z for every (player, team) pair, not just each player's real
destination -- so this also answers "what would this team have shifted to"
for teams a player didn't actually sign with.

Output: data/free_agency/team_fit_adjusted_contract_bands.csv -- one row
per (SEASON, PLAYER_NAME, TEAM): TEAM_Z_SCORE, PRED_LOW_PCT,
PREDICTED_SALARY_PCT_OF_CAP, PRED_HIGH_PCT, FIT_ADJUSTED_PCT, dollar
versions of all four (using that season's own cap), plus
ACTUAL_SALARY_PCT_OF_CAP/ACTUAL_DOLLARS where known.

Usage:
    python scripts/apply_team_fit_to_contract_band.py
"""
import io
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
Z_SCALE = 2.0


def normalize_name(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower().strip()


def load_2026_team_fit() -> pd.DataFrame:
    rankings = pd.read_csv(ROOT / "data" / "free_agency" / "free_agent_team_fit_rankings.csv")
    out = rankings[["PLAYER_NAME", "TEAM", "TEAM_Z_SCORE"]].copy()
    out.insert(0, "SEASON", "2026-27")
    return out


def load_historical_team_fit() -> pd.DataFrame:
    hist = json.loads((ROOT / "data" / "free_agency" / "historical_fit_data.json").read_text(encoding="utf-8"))
    rows = []
    for season_next, cyc in hist.items():
        names = cyc["player_names"]
        teams = cyc["team_codes"]
        for player_idx, team_idx, pct, team_z, is_real in cyc["rows"]:
            rows.append({
                "SEASON": season_next, "PLAYER_NAME": names[player_idx],
                "TEAM": teams[team_idx], "TEAM_Z_SCORE": team_z,
            })
    return pd.DataFrame(rows)


def apply_shift(df: pd.DataFrame) -> pd.DataFrame:
    position = (df["TEAM_Z_SCORE"] / Z_SCALE).clip(-1, 1)
    up = position >= 0
    shift = np.where(
        up,
        position * (df["PRED_HIGH_PCT"] - df["PREDICTED_SALARY_PCT_OF_CAP"]),
        position * (df["PREDICTED_SALARY_PCT_OF_CAP"] - df["PRED_LOW_PCT"]),
    )
    df["FIT_ADJUSTED_PCT"] = (df["PREDICTED_SALARY_PCT_OF_CAP"] + shift).clip(lower=0)
    return df


def main():
    contracts = pd.read_csv(ROOT / "data" / "salary_prediction" / "contract_model_v3_predictions.csv")
    resid_lo, resid_hi = contracts["RESIDUAL_PP"].quantile([0.25, 0.75])
    cap_by_season = pd.read_csv(ROOT / "data" / "salary_cap" / "cap_history.csv").set_index("SEASON")["SALARY_CAP"]

    contracts = contracts.copy()
    contracts["NAME_KEY"] = contracts["PLAYER_NAME"].apply(normalize_name)
    contracts["PRED_LOW_PCT"] = (contracts["PREDICTED_SALARY_PCT_OF_CAP"] + resid_lo).clip(lower=0)
    contracts["PRED_HIGH_PCT"] = (contracts["PREDICTED_SALARY_PCT_OF_CAP"] + resid_hi).clip(lower=0)

    team_fit = pd.concat([load_2026_team_fit(), load_historical_team_fit()], ignore_index=True)
    team_fit["NAME_KEY"] = team_fit["PLAYER_NAME"].apply(normalize_name)

    merged = team_fit.merge(
        contracts[["NAME_KEY", "TARGET_SEASON", "PREDICTED_SALARY_PCT_OF_CAP", "PRED_LOW_PCT", "PRED_HIGH_PCT",
                   "ACTUAL_SALARY_PCT_OF_CAP"]],
        left_on=["NAME_KEY", "SEASON"], right_on=["NAME_KEY", "TARGET_SEASON"], how="inner",
    ).drop(columns=["TARGET_SEASON", "NAME_KEY"])

    merged = apply_shift(merged)

    cap = merged["SEASON"].map(cap_by_season)
    merged["PRED_LOW_DOLLARS"] = merged["PRED_LOW_PCT"] / 100 * cap
    merged["PRED_DOLLARS"] = merged["PREDICTED_SALARY_PCT_OF_CAP"] / 100 * cap
    merged["PRED_HIGH_DOLLARS"] = merged["PRED_HIGH_PCT"] / 100 * cap
    merged["FIT_ADJUSTED_DOLLARS"] = merged["FIT_ADJUSTED_PCT"] / 100 * cap
    merged["ACTUAL_DOLLARS"] = merged["ACTUAL_SALARY_PCT_OF_CAP"] / 100 * cap

    merged = merged.sort_values(["SEASON", "PLAYER_NAME", "TEAM_Z_SCORE"], ascending=[True, True, False])
    out_path = ROOT / "data" / "free_agency" / "team_fit_adjusted_contract_bands.csv"
    merged.round(3).to_csv(out_path, index=False)

    print(f"Matched {len(merged)} (player, team, season) rows across {merged['SEASON'].nunique()} seasons "
          f"(of {len(team_fit)} team-fit rows -- some players have no contract prediction to attach to).")
    print(f"Z_SCALE={Z_SCALE}: a team at |TEAM_Z_SCORE| >= {Z_SCALE} places the price at the band edge.")

    c26 = merged[merged["SEASON"] == "2026-27"]
    example_name = c26["PLAYER_NAME"].iloc[0] if not c26.empty else None
    if example_name:
        ex = c26[c26["PLAYER_NAME"] == example_name].sort_values("TEAM_Z_SCORE", ascending=False)
        print(f"\nExample -- {example_name}, 2026-27 (band ${ex['PRED_LOW_DOLLARS'].iloc[0]:,.0f} - "
              f"${ex['PRED_HIGH_DOLLARS'].iloc[0]:,.0f}, PRED ${ex['PRED_DOLLARS'].iloc[0]:,.0f}):")
        print(ex.head(3)[["TEAM", "TEAM_Z_SCORE", "FIT_ADJUSTED_DOLLARS"]].to_string(index=False))
        print(ex.tail(3)[["TEAM", "TEAM_Z_SCORE", "FIT_ADJUSTED_DOLLARS"]].to_string(index=False))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
