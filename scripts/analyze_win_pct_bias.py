"""
Analyzes systematic bias in the win-pct ridge model's predictions across all
8 backtested seasons (2018-19 .. 2025-26), 240 team-seasons.

For each team-season, merges:
  - PRED / ACTUAL win_pct from data/win_pct_model/win_pct_model_results.json (ridge calibration)
  - NEED_VECTOR (12 playtype/defense/rebounding composite features) from
    data/team_vectors_json/team_vectors_<season>.json
  - Raw box/tracking stats (DEF_RATING, PACE, rim defense, etc.) from
    data/team_stats/team_stats_<season>.csv, with OFF_RATING derived as
    PTS / POSS * 100

Computes PRED_WINS / ACTUAL_WINS (win_pct * that season's GP), ERROR =
PRED_WINS - ACTUAL_WINS, ranks teams, correlates ERROR against every
feature, and compares over- vs under-predicted groups.

Usage:
    python scripts/analyze_win_pct_bias.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nba_api.stats.static import teams as static_teams

ROOT = Path(__file__).resolve().parent.parent
TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}

results = json.loads((ROOT / "data" / "win_pct_model" / "win_pct_model_results.json").read_text())
calibration = pd.DataFrame(results["calibration"]["ridge"])
seasons = results["seasons"]
categories = results["categories"]

rows = []
for season in seasons:
    need_path = ROOT / "data" / "team_vectors_json" / f"team_vectors_{season}.json"
    stats_path = ROOT / "data" / "team_stats" / f"team_stats_{season}.csv"
    need = json.loads(need_path.read_text())
    stat_df = pd.read_csv(stats_path)
    stat_df["TEAM"] = stat_df["TEAM_ID"].map(TEAM_ID_TO_ABBR)
    stat_df["OFF_RATING"] = stat_df["PTS"] / stat_df["POSS"] * 100
    stat_df["NET_RATING"] = stat_df["OFF_RATING"] - stat_df["DEF_RATING"]
    stat_df = stat_df.set_index("TEAM")

    for team, vec in need.items():
        if team not in stat_df.index:
            continue
        row = {"SEASON": season, "TEAM": team, "GP": int(stat_df.loc[team, "GP"])}
        row.update(vec["NEED_VECTOR"])
        for col in ["DEF_RATING", "OFF_RATING", "NET_RATING", "PACE",
                    "OREB_PCT", "DREB_PCT", "RIM_DIFF_PCT"]:
            row[col] = stat_df.loc[team, col]
        rows.append(row)

features = pd.DataFrame(rows)

df = calibration.merge(features, on=["SEASON", "TEAM"], how="inner")
df["PRED_WINS"] = df["PRED"] * df["GP"]
df["ACTUAL_WINS"] = df["ACTUAL"] * df["GP"]
df["ERROR"] = df["PRED_WINS"] - df["ACTUAL_WINS"]

df = df.sort_values("ERROR", ascending=False).reset_index(drop=True)
out_cols = ["SEASON", "TEAM", "PRED_WINS", "ACTUAL_WINS", "ERROR", "ACTUAL"] + categories + \
    ["DEF_RATING", "OFF_RATING", "NET_RATING", "PACE", "OREB_PCT", "DREB_PCT", "RIM_DIFF_PCT"]
df[out_cols].round(3).to_csv(ROOT / "data" / "win_pct_model" / "win_pct_bias_analysis.csv", index=False)

print(f"Merged {len(df)} team-seasons across {len(seasons)} seasons")
print(f"\nOverall ERROR stats (PRED_WINS - ACTUAL_WINS):")
print(f"  mean={df['ERROR'].mean():.3f}  std={df['ERROR'].std():.3f}  "
      f"MAE={df['ERROR'].abs().mean():.3f}")

print("\n=== TOP 15 MOST OVER-PREDICTED (team-seasons) ===")
print(df[["SEASON", "TEAM", "PRED_WINS", "ACTUAL_WINS", "ERROR"]].head(15).to_string(index=False))

print("\n=== TOP 15 MOST UNDER-PREDICTED (team-seasons) ===")
print(df[["SEASON", "TEAM", "PRED_WINS", "ACTUAL_WINS", "ERROR"]].tail(15).to_string(index=False))

# Team-level aggregation across seasons
team_agg = df.groupby("TEAM").agg(
    n=("ERROR", "size"),
    mean_error=("ERROR", "mean"),
    mean_actual_wins=("ACTUAL_WINS", "mean"),
).sort_values("mean_error", ascending=False)
print("\n=== TEAM-LEVEL MEAN ERROR ACROSS ALL SEASONS (most over-predicted first) ===")
print(team_agg.round(2).to_string())

# Correlation of ERROR with every feature
feature_cols = categories + ["DEF_RATING", "OFF_RATING", "NET_RATING", "PACE",
                              "OREB_PCT", "DREB_PCT", "RIM_DIFF_PCT", "ACTUAL"]
print("\n=== PEARSON CORRELATION: feature vs ERROR (sorted by |r|) ===")
corr_rows = []
for c in feature_cols:
    r, p = stats.pearsonr(df[c], df["ERROR"])
    corr_rows.append((c, r, p))
corr_df = pd.DataFrame(corr_rows, columns=["feature", "r", "p"]).sort_values(
    "r", key=lambda s: s.abs(), ascending=False)
print(corr_df.round(4).to_string(index=False))

# Over- vs under-predicted group comparison (top/bottom quartile by ERROR)
n = len(df)
q = n // 4
over = df.iloc[:q]
under = df.iloc[-q:]
print(f"\n=== GROUP COMPARISON: top quartile over-predicted (n={len(over)}) "
      f"vs top quartile under-predicted (n={len(under)}) ===")
comp_rows = []
for c in feature_cols:
    o_mean, u_mean = over[c].mean(), under[c].mean()
    t, p = stats.ttest_ind(over[c], under[c], equal_var=False)
    comp_rows.append((c, o_mean, u_mean, o_mean - u_mean, t, p))
comp_df = pd.DataFrame(comp_rows, columns=["feature", "over_mean", "under_mean", "diff", "t", "p"])
comp_df = comp_df.sort_values("p")
print(comp_df.round(4).to_string(index=False))

# Bias vs team quality (regression to the mean check)
print("\n=== ERROR vs ACTUAL WIN_PCT TERCILE (elite / mid / bottom) ===")
df["QUALITY_TERCILE"] = pd.qcut(df["ACTUAL"], 3, labels=["bottom", "mid", "elite"])
print(df.groupby("QUALITY_TERCILE", observed=True)["ERROR"].agg(["mean", "std", "size"]).round(3))

# Bias by season
print("\n=== MEAN ERROR BY SEASON ===")
print(df.groupby("SEASON")["ERROR"].agg(["mean", "std"]).round(3))

# Bias vs pace
print("\n=== ERROR vs PACE TERCILE ===")
df["PACE_TERCILE"] = pd.qcut(df["PACE"], 3, labels=["slow", "mid", "fast"])
print(df.groupby("PACE_TERCILE", observed=True)["ERROR"].agg(["mean", "std", "size"]).round(3))

df.to_csv(ROOT / "data" / "win_pct_model" / "win_pct_bias_analysis_full.csv", index=False)
print("\nWrote data/win_pct_model/win_pct_bias_analysis.csv and data/win_pct_model/win_pct_bias_analysis_full.csv")
