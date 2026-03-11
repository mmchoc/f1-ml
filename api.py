from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import fastf1
import os
import warnings
warnings.filterwarnings('ignore')

# ─── SETUP ────────────────────────────────────────────────────────────────────
app = FastAPI()

# Allow your React app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastF1 cache
cache_dir = os.path.join(os.path.dirname(__file__), "f1_cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# ─── COPY ALL YOUR FUNCTIONS FROM model.py ────────────────────────────────────
def fetch_season_standings(year):
    url = f"https://api.jolpi.ca/ergast/f1/{year}/driverStandings.json"
    response = requests.get(url)
    data = response.json()
    standings = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings:
        return None
    results = []
    for entry in standings[0]["DriverStandings"]:
        try:
            position_raw = entry.get("position", "99")
            position = int(position_raw) if str(position_raw).isdigit() else 99
            wins_raw = entry.get("wins", "0")
            wins = int(wins_raw) if str(wins_raw).isdigit() else 0
            points_raw = entry.get("points", "0")
            points = float(points_raw) if points_raw else 0
            results.append({
                "year": year,
                "driver": entry["Driver"]["code"],
                "points": points,
                "wins": wins,
                "position": position,
                "team": entry["Constructors"][0]["name"],
                "champion": 1 if position == 1 else 0
            })
        except Exception:
            continue
    return results

def fetch_race_results(year):
    url = f"https://api.jolpi.ca/ergast/f1/{year}/results.json?limit=500"
    response = requests.get(url)
    data = response.json()
    races = data["MRData"]["RaceTable"]["Races"]
    results = []
    for race in races:
        for result in race["Results"]:
            try:
                driver_code = result["Driver"]["code"]
                finish_pos = int(result["position"])
                grid_pos = int(result.get("grid", 0))
                status = result.get("status", "")
                points = float(result.get("points", 0))
                dnf = 0 if "Finished" in status or "Lap" in status else 1
                results.append({
                    "year": year,
                    "round": int(race["round"]),
                    "driver": driver_code,
                    "finish_pos": finish_pos,
                    "grid_pos": grid_pos,
                    "positions_gained": grid_pos - finish_pos,
                    "points": points,
                    "dnf": dnf,
                    "podium": 1 if finish_pos <= 3 else 0,
                    "team": result["Constructor"]["name"],
                })
            except Exception:
                continue
    return results

def fetch_mid_season_standings(year, after_round):
    url = f"https://api.jolpi.ca/ergast/f1/{year}/{after_round}/driverStandings.json"
    response = requests.get(url)
    data = response.json()
    standings = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings:
        return None
    results = []
    for entry in standings[0]["DriverStandings"]:
        try:
            position_raw = entry.get("position", "99")
            position = int(position_raw) if str(position_raw).isdigit() else 99
            wins_raw = entry.get("wins", "0")
            wins = int(wins_raw) if str(wins_raw).isdigit() else 0
            points_raw = entry.get("points", "0")
            points = float(points_raw) if points_raw else 0
            results.append({
                "year": year,
                "driver": entry["Driver"]["code"],
                "mid_points": points,
                "mid_wins": wins,
                "mid_position": position,
                "team": entry["Constructors"][0]["name"],
            })
        except Exception:
            continue
    return results

def fetch_car_pace(year, round_num):
    try:
        session = fastf1.get_session(year, round_num, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.pick_quicklaps()
        if laps.empty:
            return {}
        driver_pace = laps.groupby('Driver')['LapTime'].median()
        driver_pace_seconds = driver_pace.dt.total_seconds()
        min_pace = driver_pace_seconds.min()
        pace_rating = (min_pace / driver_pace_seconds) * 100
        return pace_rating.to_dict()
    except Exception as e:
        return {}

def build_features(mid_standings_df, race_df, final_standings_df=None, use_fastf1=False, year_races=None):
    enriched = []
    for _, row in mid_standings_df.iterrows():
        year = row["year"]
        driver = row["driver"]
        driver_races = race_df[(race_df["year"] == year) & (race_df["driver"] == driver)]
        if len(driver_races) == 0:
            continue
        avg_finish = driver_races["finish_pos"].mean()
        avg_grid = driver_races["grid_pos"].mean()
        avg_positions_gained = driver_races["positions_gained"].mean()
        dnf_rate = driver_races["dnf"].mean()
        podium_rate = driver_races["podium"].mean()
        points_per_race = driver_races["points"].mean()
        win_rate = (driver_races["finish_pos"] == 1).mean()
        consistency = driver_races["finish_pos"].std()
        year_mid = mid_standings_df[mid_standings_df["year"] == year]
        max_pts = year_mid["mid_points"].max()
        points_pct = row["mid_points"] / max_pts if max_pts > 0 else 0
        max_wins = year_mid["mid_wins"].max()
        wins_pct = row["mid_wins"] / max_wins if max_wins > 0 else 0
        pace_rating = 95.0
        if use_fastf1 and year_races:
            for round_num, pace_data in year_races.items():
                if driver in pace_data:
                    pace_rating = pace_data[driver]
                    break
        record = {
            "year": year,
            "driver": driver,
            "team": row["team"],
            "mid_points": row["mid_points"],
            "mid_position": row["mid_position"],
            "points_pct": points_pct,
            "wins_pct": wins_pct,
            "avg_finish": avg_finish,
            "avg_grid": avg_grid,
            "avg_positions_gained": avg_positions_gained,
            "dnf_rate": dnf_rate,
            "podium_rate": podium_rate,
            "points_per_race": points_per_race,
            "win_rate": win_rate,
            "consistency": consistency if not np.isnan(consistency) else 10,
            "pace_rating": pace_rating,
        }
        if final_standings_df is not None:
            final = final_standings_df[
                (final_standings_df["year"] == year) &
                (final_standings_df["driver"] == driver)
            ]
            if len(final) > 0:
                record["final_points"] = final.iloc[0]["points"]
                record["final_position"] = final.iloc[0]["position"]
            else:
                continue
        enriched.append(record)
    return pd.DataFrame(enriched)

# ─── TRAIN MODEL ON STARTUP ───────────────────────────────────────────────────
print("Training model on startup...")
all_mid_standings = []
all_final_standings = []
all_races = []

for year in range(2010, 2026):
    print(f"  Loading {year}...")
    mid = fetch_mid_season_standings(year, 1)
    final = fetch_season_standings(year)
    races = fetch_race_results(year)
    if mid: all_mid_standings.extend(mid)
    if final: all_final_standings.extend(final)
    if races: all_races.extend(races)

mid_df = pd.DataFrame(all_mid_standings)
final_df = pd.DataFrame(all_final_standings)
race_df = pd.DataFrame(all_races)
df = build_features(mid_df, race_df, final_df)

FEATURES = [
    "points_pct", "wins_pct", "mid_position",
    "avg_finish", "avg_grid", "avg_positions_gained",
    "dnf_rate", "podium_rate", "points_per_race",
    "win_rate", "consistency", "pace_rating"
]

X = df[FEATURES]
y = df["final_points"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
gb.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error
rf_mae = mean_absolute_error(y_test, rf.predict(X_test))
gb_mae = mean_absolute_error(y_test, gb.predict(X_test))
model = rf if rf_mae <= gb_mae else gb
print(f"Model ready! Using {'Random Forest' if rf_mae <= gb_mae else 'Gradient Boosting'}")

# ─── API ENDPOINTS ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "F1 Prediction API is running"}

@app.get("/api/championship")
def predict_championship():
    current_round = 1
    mid_2026 = fetch_mid_season_standings(2026, current_round)
    races_2026 = fetch_race_results(2026)

    if not mid_2026 or not races_2026:
        return {"error": "Could not fetch 2026 data"}

    mid_2026_df = pd.DataFrame(mid_2026)
    races_2026_df = pd.DataFrame(races_2026)

    pace_2026 = {}
    pace_data = fetch_car_pace(2026, current_round)
    if pace_data:
        pace_2026[current_round] = pace_data

    current_df = build_features(mid_2026_df, races_2026_df, use_fastf1=True, year_races=pace_2026)
    current_df["predicted_points"] = model.predict(current_df[FEATURES])
    current_df = current_df.sort_values("predicted_points", ascending=False)

    max_pred = current_df["predicted_points"].max()
    current_df["win_probability"] = (current_df["predicted_points"] / max_pred) ** 3
    total_prob = current_df["win_probability"].sum()
    current_df["win_probability"] = current_df["win_probability"] / total_prob

    return {
        "round": current_round,
        "predictions": [
            {
                "driver": row["driver"],
                "team": row["team"],
                "predicted_points": round(row["predicted_points"], 1),
                "win_probability": round(row["win_probability"] * 100, 1),
                "pace_rating": round(row["pace_rating"], 1),
                "current_points": row["mid_points"],
            }
            for _, row in current_df.iterrows()
        ]
    }

@app.get("/api/standings")
def get_standings():
    standings = fetch_mid_season_standings(2026, 1)
    if not standings:
        return {"error": "Could not fetch standings"}
    return {"standings": standings}
