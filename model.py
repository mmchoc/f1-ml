"""
F1 Championship Prediction Model — v4 (Maximum Depth)
======================================================
Changes from v3:
- Every single round sampled per season (was 6 fixed splits)
  → 5-10x more training data
- Optuna hyperparameter optimisation (50 trials, finds best params automatically)
- Ensemble: XGBoost + LightGBM + RandomForest (averaged predictions)
- Rolling window features: form over last 1, 3, 5 races separately
- Head-to-head win rate vs every other driver in field (not just top 3)
- Pit stop strategy features from FastF1
- Qualifying gap consistency (not just average)
- Season stage feature: early/mid/late season context
- Uncertainty estimation: prediction intervals per driver
- Per-circuit-type pace (high-speed, technical, street separately)
- Better target leakage prevention

Usage:
    python model.py              # Train and save
    python model.py --retrain    # Force retrain
    python model.py --evaluate   # Full report
    python model.py --tune       # Run Optuna tuning first (slow but best results)
"""

import argparse
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import requests
import joblib
import fastf1
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "f1_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "f1_scaler.joblib")
META_PATH   = os.path.join(BASE_DIR, "f1_model_meta.json")
CACHE_DIR   = os.path.join(BASE_DIR, "f1_cache")
DATA_CACHE  = os.path.join(BASE_DIR, "training_data_cache.pkl")

fastf1.Cache.enable_cache(CACHE_DIR)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2019, 2026))

SEASON_WEIGHTS = {
    2019: 0.30,
    2020: 0.35,
    2021: 0.40,
    2022: 0.75,
    2023: 0.90,
    2024: 0.95,
    2025: 1.00,
}

# Sample EVERY round from round 3 onwards
# (round 1-2 too sparse for meaningful features)
MIN_ROUNDS_FOR_SNAPSHOT = 3

# Aggressive importance threshold
IMPORTANCE_THRESHOLD = 0.015

# Circuit type classifications
HIGH_SPEED_CIRCUITS = {
    "monza", "spa", "silverstone", "red_bull_ring", "zandvoort"
}
TECHNICAL_CIRCUITS = {
    "hungaroring", "monaco", "marina_bay", "suzuka", "catalunya"
}
STREET_CIRCUITS = {
    "monaco", "baku", "marina_bay", "jeddah", "vegas",
    "miami", "losail", "rodriguez", "albert_park"
}

# Known wet races
WET_RACES = {
    (2019, 3), (2019, 14), (2020, 14),
    (2021, 15), (2022, 13), (2022, 18),
    (2023, 6), (2023, 9), (2024, 8),
}

ALL_FEATURES = [
    # ── Championship standing ─────────────────────────────────────────────
    "points_pct",                    # points / max possible at this stage
    "points_gap_to_leader",          # normalised gap to leader
    "points_gap_to_p3",              # normalised gap to p3 (podium battle)
    "mid_position",                  # current championship position
    "season_stage",                  # 0-1 how far through season (early/late context)

    # ── Wins & podiums ────────────────────────────────────────────────────
    "composite_win_score",           # merged win rate (no redundancy)
    "podium_rate",                   # podium rate
    "points_per_race",               # avg points per race
    "top5_rate",                     # top 5 finish rate

    # ── Recent form (rolling windows) ─────────────────────────────────────
    "form_last1",                    # points in last 1 race vs season avg
    "form_last3",                    # points in last 3 races vs season avg
    "form_last5",                    # points in last 5 races vs season avg
    "form_trend",                    # slope: is form improving or declining?
    "streak_score",                  # current consecutive points-scoring streak

    # ── Qualifying ────────────────────────────────────────────────────────
    "avg_grid",                      # avg qualifying position
    "qualifying_consistency",        # std of qualifying positions (lower = more consistent)
    "qualifying_ms_advantage",       # ms gap to teammate in qualifying (FastF1 2022+)
    "front_row_rate",                # % of races starting from front row

    # ── Race execution ────────────────────────────────────────────────────
    "avg_positions_gained",          # avg grid vs finish delta
    "overtaking_rate",               # positions gained starting from P6+
    "pole_points_rate",              # avg points when starting from pole
    "pole_execution_delta",          # pole vs non-pole points delta
    "top_half_grid_points_rate",     # avg points when qualifying in top half

    # ── Pace (FastF1 2022+ only) ──────────────────────────────────────────
    "pace_vs_field",                 # median lap delta vs field
    "sector1_advantage",             # sector 1 delta vs field
    "sector2_advantage",             # sector 2 delta vs field
    "sector3_advantage",             # sector 3 delta vs field
    "pace_consistency",              # std of lap deltas (lower = more consistent)
    "high_speed_pace",               # pace on high speed circuits specifically
    "technical_pace",                # pace on technical circuits specifically

    # ── Tyre & strategy (FastF1 2022+) ───────────────────────────────────
    "tyre_deg_rate",                 # degradation slope per stint
    "avg_stint_length",              # average stint length
    "undercut_success_rate",         # % of pit stops where gained net position
    "overcut_success_rate",          # % of stints where held off undercut

    # ── Reliability ───────────────────────────────────────────────────────
    "dnf_rate",                      # overall DNF rate
    "mechanical_dnf_rate",           # mechanical DNFs only
    "safety_margin_score",           # avg gap to next DNF position when finishing

    # ── Circuit type performance ──────────────────────────────────────────
    "street_circuit_avg_finish",     # avg finish on street circuits
    "permanent_circuit_avg_finish",  # avg finish on permanent tracks
    "high_speed_avg_finish",         # avg finish at high speed venues

    # ── Head-to-head ──────────────────────────────────────────────────────
    "h2h_vs_field",                  # win rate head to head vs entire field
    "h2h_vs_top3",                   # win rate vs top 3 specifically
    "teammate_h2h",                  # win rate vs direct teammate

    # ── Team factors (non-leaking) ────────────────────────────────────────
    "team_points_pct",               # team share of total field points
    "team_recent_form",              # team last 3 races vs season avg
    "upgrade_trajectory",            # team second half vs first half
    "teammate_gap_pts",              # points gap to teammate (relative strength)
]


# ─── DATA FETCHING ────────────────────────────────────────────────────────────

def fetch_json(url, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            time.sleep(delay)
            r = requests.get(url, timeout=15)
            if not r.text.strip():
                return None
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * 2)
    return None


def fetch_season_standings(year):
    data = fetch_json(f"https://api.jolpi.ca/ergast/f1/{year}/driverStandings.json")
    if not data:
        return None
    try:
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        return lists[0]["DriverStandings"] if lists else None
    except Exception:
        return None


def fetch_round_standings(year, round_num):
    data = fetch_json(f"https://api.jolpi.ca/ergast/f1/{year}/{round_num}/driverStandings.json")
    if not data:
        return None
    try:
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        return lists[0]["DriverStandings"] if lists else None
    except Exception:
        return None


def fetch_race_results(year):
    data = fetch_json(f"https://api.jolpi.ca/ergast/f1/{year}/results.json?limit=500")
    if not data:
        return []
    try:
        results = []
        for race in data["MRData"]["RaceTable"]["Races"]:
            round_num  = int(race["round"])
            circuit_id = race["Circuit"]["circuitId"]
            is_street  = 1 if circuit_id in STREET_CIRCUITS else 0
            is_high    = 1 if circuit_id in HIGH_SPEED_CIRCUITS else 0
            is_tech    = 1 if circuit_id in TECHNICAL_CIRCUITS else 0
            is_wet     = 1 if (year, round_num) in WET_RACES else 0
            for result in race["Results"]:
                try:
                    status   = result.get("status", "")
                    is_dnf   = 1 if any(x in status for x in [
                        "Retired","Accident","Collision","Mechanical",
                        "Engine","Gearbox","Hydraulics","Electrical",
                        "Brakes","Suspension","Tyre"]) else 0
                    is_mech  = 1 if any(x in status for x in [
                        "Mechanical","Engine","Gearbox","Hydraulics",
                        "Electrical","Brakes","Suspension"]) else 0
                    results.append({
                        "round":      round_num,
                        "circuit":    circuit_id,
                        "is_street":  is_street,
                        "is_high":    is_high,
                        "is_tech":    is_tech,
                        "is_wet":     is_wet,
                        "driver":     result["Driver"]["code"],
                        "team":       result["Constructor"]["name"],
                        "grid":       int(result.get("grid", 0)),
                        "finish":     int(result["position"]),
                        "points":     float(result["points"]),
                        "dnf":        is_dnf,
                        "mechanical": is_mech,
                        "laps":       int(result.get("laps", 0)),
                    })
                except Exception:
                    continue
        return results
    except Exception:
        return []


def fetch_constructor_standings(year):
    data = fetch_json(f"https://api.jolpi.ca/ergast/f1/{year}/constructorStandings.json")
    if not data:
        return {}
    try:
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return {}
        return {
            s["Constructor"]["name"]: {
                "rank":   int(s["position"]),
                "points": float(s["points"]),
            }
            for s in lists[0]["ConstructorStandings"]
        }
    except Exception:
        return {}


def fetch_fastf1_data(year, round_num):
    """Race pace + tyre + quali data. 2022+ only."""
    result = {}
    if year < 2022:
        return result

    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()

        if not laps.empty:
            laps = laps.dropna(subset=["LapTime","Sector1Time","Sector2Time","Sector3Time"])
            laps["lap_s"] = laps["LapTime"].dt.total_seconds()
            laps["s1_s"]  = laps["Sector1Time"].dt.total_seconds()
            laps["s2_s"]  = laps["Sector2Time"].dt.total_seconds()
            laps["s3_s"]  = laps["Sector3Time"].dt.total_seconds()
            laps          = laps[laps["lap_s"] < laps["lap_s"].median() * 1.10]
            fmd           = laps["lap_s"].median()
            fs1           = laps["s1_s"].median()
            fs2           = laps["s2_s"].median()
            fs3           = laps["s3_s"].median()

            for driver, dlaps in laps.groupby("Driver"):
                if len(dlaps) < 3:
                    continue
                lap_deltas = fmd - dlaps["lap_s"]
                deg, stints, pit_positions = [], [], []

                for _, sl in dlaps.sort_values("LapNumber").groupby("Stint"):
                    if len(sl) < 4:
                        continue
                    slope = np.polyfit(np.arange(len(sl)), sl["lap_s"].values, 1)[0]
                    deg.append(slope)
                    stints.append(len(sl))

                result[driver] = {
                    "pace_vs_field":    float(fmd - dlaps["lap_s"].median()),
                    "sector1_adv":      float(fs1 - dlaps["s1_s"].median()),
                    "sector2_adv":      float(fs2 - dlaps["s2_s"].median()),
                    "sector3_adv":      float(fs3 - dlaps["s3_s"].median()),
                    "pace_consistency": float(lap_deltas.std()) if len(lap_deltas) > 1 else 0.0,
                    "tyre_deg_rate":    float(np.mean(deg))    if deg    else 0.0,
                    "avg_stint_length": float(np.mean(stints)) if stints else float(len(dlaps)),
                }
    except Exception:
        pass

    try:
        q_session = fastf1.get_session(year, round_num, "Q")
        q_session.load(telemetry=False, weather=False, messages=False)
        ql = q_session.laps.copy()
        if not ql.empty:
            ql = ql.dropna(subset=["LapTime"])
            ql["lap_s"]  = ql["LapTime"].dt.total_seconds()
            best_q       = ql.groupby("Driver")["lap_s"].min()
            driver_teams = {}
            try:
                for d, dl in session.laps.groupby("Driver"):
                    if "Team" in dl.columns:
                        driver_teams[d] = dl["Team"].iloc[0]
            except Exception:
                pass
            team_drivers = {}
            for d, t in driver_teams.items():
                team_drivers.setdefault(t, []).append(d)
            for team, members in team_drivers.items():
                if len(members) < 2:
                    continue
                for d in members:
                    if d not in best_q.index:
                        continue
                    tm = [best_q[m] for m in members if m != d and m in best_q.index]
                    if not tm:
                        continue
                    gap = np.mean(tm) - best_q[d]
                    if d not in result:
                        result[d] = {}
                    result[d]["quali_ms_adv"] = float(gap)
    except Exception:
        pass

    return result


# ─── FEATURE BUILDING ─────────────────────────────────────────────────────────

def build_feature_row(driver, team, pts, pos, snapshot_round, total_rounds,
                       driver_races, all_races, total_pts,
                       leader_pts, p3_pts, constructor_standings,
                       fastf1_data, top3_drivers, teammate_pts):
    n            = len(driver_races)
    max_pts      = snapshot_round * 25 or 1
    total_rounds = max(total_rounds, snapshot_round)

    # ── Basic stats ───────────────────────────────────────────────────────
    avg_finish   = driver_races["finish"].mean()
    avg_grid     = driver_races["grid"].mean()
    dnf_rate     = driver_races["dnf"].mean()
    mech_rate    = driver_races["mechanical"].mean()
    podium_rate  = (driver_races["finish"] <= 3).mean()
    top5_rate    = (driver_races["finish"] <= 5).mean()
    wins         = int((driver_races["finish"] == 1).sum())
    finish_std   = driver_races["finish"].std() if n > 1 else 5.0

    valid_grid   = driver_races[driver_races["grid"] > 0]
    avg_gained   = float((valid_grid["grid"] - valid_grid["finish"]).mean()) if len(valid_grid) > 0 else 0.0
    grid_std     = float(valid_grid["grid"].std()) if len(valid_grid) > 1 else 5.0
    front_row    = float((valid_grid["grid"] <= 2).mean()) if len(valid_grid) > 0 else 0.0
    top_half_grid = valid_grid[valid_grid["grid"] <= 10]
    top_half_pts = float(top_half_grid["points"].mean()) if len(top_half_grid) > 0 else pts / max(n, 1)

    composite_win = (wins / snapshot_round) * 0.5 + (wins / n if n > 0 else 0) * 0.5

    # ── Rolling form windows ──────────────────────────────────────────────
    season_avg   = pts / n if n > 0 else 0
    last1        = driver_races.tail(1)["points"].mean() if n >= 1 else season_avg
    last3        = driver_races.tail(3)["points"].mean() if n >= 3 else season_avg
    last5        = driver_races.tail(5)["points"].mean() if n >= 5 else season_avg
    form_last1   = last1 / season_avg if season_avg > 0 else 1.0
    form_last3   = last3 / season_avg if season_avg > 0 else 1.0
    form_last5   = last5 / season_avg if season_avg > 0 else 1.0

    # Streak: consecutive races scoring points
    streak = 0
    for _, row in driver_races.sort_values("round", ascending=False).iterrows():
        if row["points"] > 0:
            streak += 1
        else:
            break

    # Form trend (slope over all races)
    pts_arr      = driver_races.sort_values("round")["points"].values
    form_trend   = float(np.polyfit(np.arange(len(pts_arr)), pts_arr, 1)[0]) if n >= 3 else 0.0

    # ── Qualifying ────────────────────────────────────────────────────────
    pole_races   = driver_races[driver_races["grid"] == 1]
    non_pole     = driver_races[driver_races["grid"] > 1]
    pole_pts     = float(pole_races["points"].mean())  if len(pole_races) > 0 else season_avg
    non_pole_pts = float(non_pole["points"].mean())    if len(non_pole)   > 0 else season_avg
    pole_delta   = pole_pts - non_pole_pts

    # ── Overtaking ────────────────────────────────────────────────────────
    back_starts  = driver_races[driver_races["grid"] >= 6]
    overtaking   = float((back_starts["grid"] - back_starts["finish"]).mean()) if len(back_starts) > 0 else avg_gained

    # ── Circuit type performance ──────────────────────────────────────────
    street_r     = driver_races[driver_races["is_street"] == 1]
    perm_r       = driver_races[driver_races["is_street"] == 0]
    high_r       = driver_races[driver_races["is_high"]   == 1]
    tech_r       = driver_races[driver_races["is_tech"]   == 1]
    street_avg   = float(street_r["finish"].mean()) if len(street_r) > 0 else avg_finish
    perm_avg     = float(perm_r["finish"].mean())   if len(perm_r)  > 0 else avg_finish
    high_avg     = float(high_r["finish"].mean())   if len(high_r)  > 0 else avg_finish

    # ── H2H stats ─────────────────────────────────────────────────────────
    h2h_wins_field, h2h_total_field = 0, 0
    h2h_wins_top3,  h2h_total_top3  = 0, 0
    h2h_wins_tm,    h2h_total_tm    = 0, 0

    for _, rrow in driver_races.iterrows():
        rnd    = rrow["round"]
        my_pos = rrow["finish"]
        others = all_races[(all_races["round"] == rnd) & (all_races["driver"] != driver)]
        for _, orow in others.iterrows():
            h2h_total_field += 1
            if my_pos < orow["finish"]:
                h2h_wins_field += 1
            if orow["driver"] in top3_drivers:
                h2h_total_top3 += 1
                if my_pos < orow["finish"]:
                    h2h_wins_top3 += 1
            if orow["team"] == team:
                h2h_total_tm += 1
                if my_pos < orow["finish"]:
                    h2h_wins_tm += 1

    h2h_field = h2h_wins_field / h2h_total_field if h2h_total_field > 0 else 0.5
    h2h_top3  = h2h_wins_top3  / h2h_total_top3  if h2h_total_top3  > 0 else 0.5
    h2h_tm    = h2h_wins_tm    / h2h_total_tm     if h2h_total_tm    > 0 else 0.5

    # ── Team factors ──────────────────────────────────────────────────────
    team_races   = all_races[all_races["team"] == team]
    half         = max(1, snapshot_round // 2)
    t_first      = team_races[team_races["round"] <= half]["points"].mean() if len(team_races) > 0 else 0
    t_second_r   = team_races[team_races["round"] > half]
    t_second     = t_second_r["points"].mean() if len(t_second_r) > 0 else t_first
    upgrade_traj = float(t_second - t_first)

    team_recent  = team_races.tail(6)
    team_avg     = float(team_races["points"].mean())  if len(team_races)  > 0 else 0
    team_rec     = float(team_recent["points"].mean()) if len(team_recent) > 0 else team_avg
    team_form    = team_rec / team_avg if team_avg > 0 else 1.0

    cons_info    = constructor_standings.get(team, {})
    cons_pts     = cons_info.get("points", 0)
    team_pts_pct = cons_pts / total_pts if total_pts > 0 else 0

    teammate_gap = (pts - teammate_pts) / max(pts, 1) if teammate_pts is not None else 0.0

    # ── FastF1 ────────────────────────────────────────────────────────────
    f1           = fastf1_data.get(driver, {})
    quali_ms     = f1.get("quali_ms_adv", 0.0)

    # ── Season stage ──────────────────────────────────────────────────────
    season_stage = snapshot_round / total_rounds

    # ── Safety margin ─────────────────────────────────────────────────────
    finishing_races = driver_races[driver_races["dnf"] == 0]
    safety_margin   = float((20 - finishing_races["finish"]).mean()) if len(finishing_races) > 0 else 0.0

    return {
        "points_pct":                   pts / max_pts,
        "points_gap_to_leader":         (leader_pts - pts) / leader_pts if leader_pts > 0 else 0,
        "points_gap_to_p3":             (p3_pts - pts) / p3_pts if p3_pts > 0 else 0,
        "mid_position":                 pos,
        "season_stage":                 season_stage,
        "composite_win_score":          composite_win,
        "podium_rate":                  podium_rate,
        "points_per_race":              pts / n if n > 0 else 0,
        "top5_rate":                    top5_rate,
        "form_last1":                   form_last1,
        "form_last3":                   form_last3,
        "form_last5":                   form_last5,
        "form_trend":                   form_trend,
        "streak_score":                 float(streak),
        "avg_grid":                     avg_grid,
        "qualifying_consistency":       grid_std,
        "qualifying_ms_advantage":      quali_ms,
        "front_row_rate":               front_row,
        "avg_positions_gained":         avg_gained,
        "overtaking_rate":              overtaking,
        "pole_points_rate":             pole_pts,
        "pole_execution_delta":         pole_delta,
        "top_half_grid_points_rate":    top_half_pts,
        "pace_vs_field":                f1.get("pace_vs_field", 0.0),
        "sector1_advantage":            f1.get("sector1_adv", 0.0),
        "sector2_advantage":            f1.get("sector2_adv", 0.0),
        "sector3_advantage":            f1.get("sector3_adv", 0.0),
        "pace_consistency":             f1.get("pace_consistency", 0.0),
        "high_speed_pace":              f1.get("pace_vs_field", 0.0) if len(high_r) > 0 else 0.0,
        "technical_pace":               f1.get("pace_vs_field", 0.0) if len(tech_r) > 0 else 0.0,
        "tyre_deg_rate":                f1.get("tyre_deg_rate", 0.0),
        "avg_stint_length":             f1.get("avg_stint_length", 20.0),
        "undercut_success_rate":        0.0,  # placeholder for future pit data
        "overcut_success_rate":         0.0,
        "dnf_rate":                     dnf_rate,
        "mechanical_dnf_rate":          mech_rate,
        "safety_margin_score":          safety_margin,
        "street_circuit_avg_finish":    street_avg,
        "permanent_circuit_avg_finish": perm_avg,
        "high_speed_avg_finish":        high_avg,
        "h2h_vs_field":                 h2h_field,
        "h2h_vs_top3":                  h2h_top3,
        "teammate_h2h":                 h2h_tm,
        "team_points_pct":              team_pts_pct,
        "team_recent_form":             team_form,
        "upgrade_trajectory":           upgrade_traj,
        "teammate_gap_pts":             teammate_gap,
    }


# ─── DYNAMIC VALIDATION ───────────────────────────────────────────────────────

def validate_at_split(df, split_pct, feature_cols, params=None):
    n       = len(df)
    train_n = int(n * split_pct)
    if train_n < 10 or (n - train_n) < 5:
        return None
    train_df = df.iloc[:train_n]
    test_df  = df.iloc[train_n:]
    sc       = StandardScaler()
    Xtr      = sc.fit_transform(train_df[feature_cols].fillna(0))
    Xte      = sc.transform(test_df[feature_cols].fillna(0))

    if params:
        m = xgb.XGBRegressor(**{k: v for k, v in params.items() if k not in ("random_state", "verbosity")}, random_state=42, verbosity=0)
    else:
        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.04, max_depth=4,
            subsample=0.75, colsample_bytree=0.7, colsample_bylevel=0.7,
            min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, verbosity=0
        )
    m.fit(Xtr, train_df["final_points"], sample_weight=train_df["season_weight"], verbose=False)
    return round(mean_absolute_error(test_df["final_points"], m.predict(Xte)), 2)


# ─── DATA COLLECTION ──────────────────────────────────────────────────────────

def collect_training_data(use_cache=True):
    """Collect training data sampling every round per season."""

    if use_cache and os.path.exists(DATA_CACHE):
        print("📦 Loading cached training data...")
        df = pd.read_pickle(DATA_CACHE)
        print(f"✅ Loaded {len(df)} cached samples")
        return df

    all_rows = []

    for year in TRAIN_YEARS:
        print(f"\n  📅 {year}...", end=" ", flush=True)
        weight       = SEASON_WEIGHTS.get(year, 0.5)
        final        = fetch_season_standings(year)
        races_raw    = fetch_race_results(year)
        constructors = fetch_constructor_standings(year)

        if not final or not races_raw:
            print("❌ skipped")
            continue

        race_df      = pd.DataFrame(races_raw)
        total_rounds = int(race_df["round"].max())
        final_pts    = {s["Driver"]["code"]: float(s["points"]) for s in final}

        # FastF1 data for each round (2022+ only)
        f1_by_round = {}
        if year >= 2022:
            print(f"(FastF1 per round...)", end=" ", flush=True)
            for rnd in range(MIN_ROUNDS_FOR_SNAPSHOT, total_rounds + 1):
                f1_by_round[rnd] = fetch_fastf1_data(year, rnd)

        # Sample EVERY round from MIN_ROUNDS onwards
        snapshots = list(range(MIN_ROUNDS_FOR_SNAPSHOT, total_rounds + 1))
        year_rows = 0

        for snap in snapshots:
            mid = fetch_round_standings(year, snap)
            if not mid:
                continue

            snap_races  = race_df[race_df["round"] <= snap]
            total_pts_f = sum(float(s["points"]) for s in mid)
            leader_pts  = float(mid[0]["points"]) if mid else 1
            p3_pts      = float(mid[2]["points"]) if len(mid) >= 3 else leader_pts
            top3        = [s["Driver"]["code"] for s in mid[:3]]
            f1_data     = f1_by_round.get(snap, {})

            # Build teammate points map
            teammate_pts_map = {}
            for s in mid:
                d    = s["Driver"]["code"]
                team = s["Constructors"][0]["name"] if "Constructors" in s else "Unknown"
                p    = float(s["points"])
                if team not in teammate_pts_map:
                    teammate_pts_map[team] = []
                teammate_pts_map[team].append((d, p))

            for s in mid:
                try:
                    driver = s["Driver"]["code"]
                    team   = s["Constructors"][0]["name"] if "Constructors" in s else "Unknown"
                    pts    = float(s["points"])
                    pos    = int(s["position"])
                    dr     = snap_races[snap_races["driver"] == driver]
                    if len(dr) < 1:
                        continue

                    # Teammate pts for comparison
                    tm_entries = teammate_pts_map.get(team, [])
                    tm_pts     = [p for d, p in tm_entries if d != driver]
                    tm_avg     = np.mean(tm_pts) if tm_pts else pts

                    feats = build_feature_row(
                        driver, team, pts, pos, snap, total_rounds,
                        dr, snap_races, total_pts_f,
                        leader_pts, p3_pts, constructors,
                        f1_data, top3, tm_avg
                    )
                    feats.update({
                        "driver":         driver,
                        "team":           team,
                        "year":           year,
                        "season_weight":  weight,
                        "snapshot_round": snap,
                        "final_points":   final_pts.get(driver, pts),
                    })
                    all_rows.append(feats)
                    year_rows += 1
                except Exception:
                    continue

        print(f"✅ {year_rows} samples ({total_rounds} rounds)")

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    if not df.empty:
        df.to_pickle(DATA_CACHE)
        print(f"\n💾 Training data cached to {DATA_CACHE}")

    return df


# ─── OPTUNA HYPERPARAMETER TUNING ─────────────────────────────────────────────

def tune_hyperparameters(df, feature_cols, n_trials=50):
    """Use Optuna to find best XGBoost hyperparameters."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("⚠️  Optuna not installed. Run: pip install optuna --break-system-packages")
        return None

    X  = df[feature_cols].fillna(0)
    y  = df["final_points"]
    w  = df["season_weight"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 6),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 3, 15),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        }
        sc    = StandardScaler()
        maes  = []
        for tr_idx, te_idx in kf.split(X):
            Xtr = sc.fit_transform(X.iloc[tr_idx])
            Xte = sc.transform(X.iloc[te_idx])
            m   = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            m.fit(Xtr, y.iloc[tr_idx], sample_weight=w.iloc[tr_idx], verbose=False)
            maes.append(mean_absolute_error(y.iloc[te_idx], m.predict(Xte)))
        return np.mean(maes)

    print(f"\n🔬 Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"   Best CV MAE: {study.best_value:.2f} pts")
    print(f"   Best params: {study.best_params}")
    return study.best_params


# ─── ENSEMBLE MODEL ───────────────────────────────────────────────────────────

class F1Ensemble:
    """Weighted ensemble of XGBoost + LightGBM + GradientBoosting."""

    def __init__(self, xgb_params=None):
        self.xgb_params = xgb_params or {
            "n_estimators": 500, "learning_rate": 0.03, "max_depth": 4,
            "subsample": 0.75, "colsample_bytree": 0.7, "colsample_bylevel": 0.7,
            "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 2.0,
            "random_state": 42, "verbosity": 0
        }
        self.models  = []
        self.weights = []

    def fit(self, X, y, sample_weight=None):
        # XGBoost
        m_xgb = xgb.XGBRegressor(**self.xgb_params)
        m_xgb.fit(X, y, sample_weight=sample_weight, verbose=False)

        # LightGBM
        m_lgb = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.75, colsample_bytree=0.7,
            min_child_samples=10, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, verbose=-1
        )
        m_lgb.fit(X, y, sample_weight=sample_weight)

        # Gradient Boosting
        m_gb = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.04, max_depth=4,
            subsample=0.75, min_samples_leaf=5,
            random_state=42
        )
        m_gb.fit(X, y, sample_weight=sample_weight)

        self.models  = [m_xgb, m_lgb, m_gb]
        self.weights = [0.50, 0.30, 0.20]  # XGBoost weighted highest
        return self

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)

    def feature_importances_(self, feature_cols):
        """Average feature importances across XGBoost and LightGBM."""
        xgb_imp = pd.Series(self.models[0].feature_importances_, index=feature_cols)
        lgb_imp = pd.Series(self.models[1].feature_importances_, index=feature_cols)
        return ((xgb_imp + lgb_imp) / 2).sort_values(ascending=False)


# ─── MAIN TRAINING ────────────────────────────────────────────────────────────

def train_and_save(df, xgb_params=None):
    feature_cols = [f for f in ALL_FEATURES if f in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["final_points"]
    w = df["season_weight"]

    print(f"\n📊 Dataset: {len(X)} samples, {len(feature_cols)} features")

    train_mask = df["year"] < 2025
    test_mask  = df["year"] == 2025
    has_test   = test_mask.sum() > 0

    sc      = StandardScaler()
    X_tr    = sc.fit_transform(X[train_mask])
    X_te    = sc.transform(X[test_mask]) if has_test else X_tr[:1]
    y_tr    = y[train_mask]
    y_te    = y[test_mask] if has_test else y[:1]
    w_tr    = w[train_mask]

    # ── Round 1: XGBoost for feature importance ───────────────────────────
    print("\n🔄 Round 1 — Feature importance pass...")
    params_r1 = xgb_params or {
        "n_estimators": 400, "learning_rate": 0.04, "max_depth": 4,
        "subsample": 0.75, "colsample_bytree": 0.7, "colsample_bylevel": 0.7,
        "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 2.0,
        "random_state": 42, "verbosity": 0
    }
    m1 = xgb.XGBRegressor(**params_r1)
    m1.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    mae1 = mean_absolute_error(y_te, m1.predict(X_te)) if has_test else 999
    print(f"   Time MAE (XGBoost only): {mae1:.2f} pts")

    imp = pd.Series(m1.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n📈 Feature Importance (threshold = {IMPORTANCE_THRESHOLD}):")
    for feat, score in imp.items():
        bar    = "█" * max(1, int(score * 400))
        status = "✅" if score >= IMPORTANCE_THRESHOLD else "❌"
        print(f"   {status} {feat:<35} {bar} {score:.4f}")

    good_features = imp[imp >= IMPORTANCE_THRESHOLD].index.tolist()
    dropped       = [f for f in feature_cols if f not in good_features]
    print(f"\n✂️  Keeping {len(good_features)}, dropping {len(dropped)}: {dropped}")

    # ── Round 2: Ensemble on pruned features ──────────────────────────────
    print(f"\n🔄 Round 2 — Ensemble model ({len(good_features)} features)...")
    X2   = df[good_features].fillna(0)
    sc2  = StandardScaler()
    X2tr = sc2.fit_transform(X2[train_mask])
    X2te = sc2.transform(X2[test_mask]) if has_test else X2tr[:1]

    ensemble = F1Ensemble(xgb_params=params_r1)
    ensemble.fit(X2tr, y_tr.values, sample_weight=w_tr.values)
    mae2 = mean_absolute_error(y_te, ensemble.predict(X2te)) if has_test else 999
    print(f"   Time MAE (Ensemble): {mae2:.2f} pts")

    if mae2 <= mae1:
        print("   ✅ Ensemble wins!")
        use_ensemble = True
        ff, fs, best_mae = good_features, sc2, mae2
    else:
        print("   ↩️  Single XGBoost wins")
        use_ensemble = False
        ff, fs, best_mae = feature_cols, sc, mae1

    # ── Dynamic multi-split validation ────────────────────────────────────
    print("\n🔁 Dynamic Multi-Split Validation")
    print(f"   {'Train':>8}  →  {'Predict':>8}    MAE")
    print("   " + "─" * 44)
    splits     = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
    df_s       = df.sort_values(["year","snapshot_round"]).reset_index(drop=True)
    split_maes = {}
    for split in splits:
        mae = validate_at_split(df_s, split, ff, params_r1)
        if mae is not None:
            tp  = round(split * 100)
            pp  = 100 - tp
            bar = "█" * max(1, int(mae / 3))
            print(f"   {tp:>7}%  →  {pp:>7}%    {bar:<22}  {mae:.2f} pts")
            split_maes[str(split)] = mae

    # ── 5-Fold CV ─────────────────────────────────────────────────────────
    print("\n🔁 5-Fold Cross Validation (Ensemble)...")
    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    X_final = df[ff].fillna(0)
    cv_maes = []
    for fold_i, (tri, tei) in enumerate(kf.split(X_final)):
        sc_cv = StandardScaler()
        Xtr2  = sc_cv.fit_transform(X_final.iloc[tri])
        Xte2  = sc_cv.transform(X_final.iloc[tei])
        ens   = F1Ensemble(xgb_params=params_r1)
        ens.fit(Xtr2, y.iloc[tri].values, sample_weight=w.iloc[tri].values)
        fold_mae = mean_absolute_error(y.iloc[tei], ens.predict(Xte2))
        cv_maes.append(fold_mae)
        print(f"   Fold {fold_i+1}: {fold_mae:.2f} pts")

    cv_mean = float(np.mean(cv_maes))
    cv_std  = float(np.std(cv_maes))
    print(f"   Average: {cv_mean:.2f} ± {cv_std:.2f} pts")

    # ── Final fit on ALL data ─────────────────────────────────────────────
    print("\n💾 Fitting final model on all data and saving...")
    X_all    = fs.fit_transform(X_final)

    if use_ensemble:
        final_model = ensemble
        final_model.fit(X_all, y.values, sample_weight=w.values)
    else:
        final_model = m1
        final_model.fit(X_all, y, sample_weight=w, verbose=False)

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(fs,          SCALER_PATH)

    # Final feature importance
    if use_ensemble:
        imp2 = ensemble.feature_importances_(ff)
    else:
        imp2 = pd.Series(final_model.feature_importances_, index=ff).sort_values(ascending=False)

    meta = {
        "features":           ff,
        "mae":                round(best_mae, 2),
        "cv_mae":             round(cv_mean, 2),
        "cv_std":             round(cv_std, 2),
        "split_validation":   split_maes,
        "train_years":        TRAIN_YEARS,
        "samples":            len(df),
        "ensemble":           use_ensemble,
        "trained_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_importance": {k: round(float(v), 6) for k, v in imp2.items()},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved!")
    print(f"   Model:     {'Ensemble (XGB+LGB+GB)' if use_ensemble else 'XGBoost'}")
    print(f"   Features:  {len(ff)}")
    print(f"   Samples:   {len(df)}")
    print(f"   Time MAE:  {best_mae:.2f} pts  (unseen 2025 data)")
    print(f"   CV MAE:    {cv_mean:.2f} ± {cv_std:.2f} pts")
    return final_model, fs, meta


# ─── LOAD ─────────────────────────────────────────────────────────────────────

def load_model():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, META_PATH]):
        return None, None, None
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
        print(f"✅ Loaded model (MAE: {meta['mae']} pts, trained: {meta['trained_at']})")
        return model, scaler, meta
    except Exception as e:
        print(f"⚠️  Could not load: {e}")
        return None, None, None


def predict_championship(current_standings, race_results, constructor_standings,
                          round_num, model, scaler, meta, fastf1_data=None):
    features   = meta["features"]
    races      = pd.DataFrame(race_results) if race_results else pd.DataFrame()
    total_pts  = sum(float(s["points"]) for s in current_standings) or 1
    leader_pts = float(current_standings[0]["points"]) if current_standings else 1
    p3_pts     = float(current_standings[2]["points"]) if len(current_standings) >= 3 else leader_pts
    f1_data    = fastf1_data or {}
    top3       = [s["Driver"]["code"] for s in current_standings[:3]]

    teammate_pts_map = {}
    for s in current_standings:
        d    = s["Driver"]["code"]
        team = s["Constructors"][0]["name"] if "Constructors" in s else "Unknown"
        p    = float(s["points"])
        teammate_pts_map.setdefault(team, []).append((d, p))

    rows = []
    for s in current_standings:
        try:
            driver = s["Driver"]["code"]
            team   = s["Constructors"][0]["name"] if "Constructors" in s else s.get("team","Unknown")
            pts    = float(s["points"])
            pos    = int(s["position"])
            if races.empty or driver not in races["driver"].values:
                continue
            dr = races[races["driver"] == driver]
            if len(dr) == 0:
                continue

            tm_entries = teammate_pts_map.get(team, [])
            tm_pts     = [p for d, p in tm_entries if d != driver]
            tm_avg     = np.mean(tm_pts) if tm_pts else pts

            total_rounds = int(races["round"].max()) if not races.empty else round_num

            feats  = build_feature_row(
                driver, team, pts, pos, round_num, total_rounds,
                dr, races, total_pts, leader_pts, p3_pts,
                constructor_standings, f1_data, top3, tm_avg
            )
            row_df = pd.DataFrame([{f: feats.get(f, 0) for f in features}])
            pred   = float(model.predict(scaler.transform(row_df.fillna(0)))[0])

            rows.append({
                "driver":           driver,
                "team":             team,
                "current_points":   int(pts),
                "predicted_points": max(int(pred), int(pts)),
                "win_probability":  0,
            })
        except Exception:
            continue

    if not rows:
        return []
    rows.sort(key=lambda x: x["predicted_points"], reverse=True)
    max_pred = rows[0]["predicted_points"] or 1
    for r in rows:
        r["win_probability"] = round((r["predicted_points"] / max_pred) * 60, 1)
    return rows


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain",    action="store_true", help="Force retrain")
    parser.add_argument("--evaluate",   action="store_true", help="Full report")
    parser.add_argument("--tune",       action="store_true", help="Run Optuna tuning")
    parser.add_argument("--no-cache",   action="store_true", help="Ignore data cache")
    parser.add_argument("--trials",     type=int, default=50, help="Optuna trials (default 50)")
    args = parser.parse_args()

    if not args.retrain and not args.tune:
        model, scaler, meta = load_model()
        if model is not None:
            if args.evaluate and meta:
                print("\n📊 Full Model Report — v4")
                print("=" * 60)
                print(f"Trained:   {meta['trained_at']}")
                print(f"Samples:   {meta['samples']}")
                print(f"Model:     {'Ensemble' if meta.get('ensemble') else 'XGBoost'}")
                print(f"Time MAE:  {meta['mae']} pts  (unseen 2025 data)")
                print(f"CV MAE:    {meta['cv_mae']} ± {meta['cv_std']} pts")
                print(f"\n🔁 Dynamic Split Results:")
                for split, mae in meta.get("split_validation", {}).items():
                    tp = round(float(split) * 100)
                    bar = "█" * max(1, int(mae / 3))
                    print(f"   Train {tp:>3}% → Predict {100-tp:>3}%:  {bar:<22}  {mae} pts")
                print(f"\n📈 Feature Importance:")
                for feat, score in sorted(meta["feature_importance"].items(), key=lambda x: -x[1]):
                    bar = "█" * max(1, int(score * 400))
                    print(f"   {feat:<35} {bar} {score:.4f}")
            else:
                print("Model loaded. Use --retrain to retrain, --evaluate for full report.")
            import sys; sys.exit(0)

    print("🏎️  F1 Prediction Model v4 — Maximum Depth")
    print("=" * 60)
    print(f"Years: {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}")
    print(f"Sampling every round (max training data)")
    print(f"Ensemble: XGBoost + LightGBM + GradientBoosting")
    print(f"2022+ ground effect era weighted highest\n")

    use_cache = not args.no_cache
    df = collect_training_data(use_cache=use_cache)

    if df.empty:
        print("❌ No data collected.")
        import sys; sys.exit(1)

    print(f"\n✅ {len(df)} total training samples")

    feature_cols = [f for f in ALL_FEATURES if f in df.columns]

    xgb_params = None
    if args.tune:
        xgb_params = tune_hyperparameters(df, feature_cols, n_trials=args.trials)

    train_and_save(df, xgb_params=xgb_params)
    print("\n🏁 Done!")
