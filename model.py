import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── STEP 1: FETCH REAL F1 DATA ───────────────────────────────────────────────
def fetch_season_data(year):
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
        except Exception as e:
            print(f"  Skipping entry due to error: {e}")
            continue
    return results

# ─── STEP 2: BUILD DATASET FROM 2010-2025 ─────────────────────────────────────
print("Fetching historical F1 data...")
all_data = []
for year in range(2010, 2026):
    print(f"  Loading {year}...")
    season = fetch_season_data(year)
    if season:
        all_data.extend(season)

df = pd.DataFrame(all_data)
print(f"Loaded {len(df)} driver-season records")
print(df.head())

# ─── STEP 3: PREPARE FEATURES FOR THE MODEL ───────────────────────────────────
# These are the things the model learns from
df["points_pct"] = df["points"] / (df.groupby("year")["points"].transform("max"))
df["wins_pct"] = df["wins"] / (df.groupby("year")["wins"].transform("max").replace(0, 1))
features = ["points_pct", "wins_pct", "position"]
target = "champion"

X = df[features]
y = df[target]

# ─── STEP 4: SPLIT DATA INTO TRAINING AND TESTING ─────────────────────────────
# 80% of data trains the model, 20% tests how accurate it is
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── STEP 5: TRAIN THE RANDOM FOREST MODEL ────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ─── STEP 6: TEST THE MODEL ACCURACY ──────────────────────────────────────────
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.1f}%")

# ─── STEP 7: PREDICT 2026 CHAMPIONSHIP ────────────────────────────────────────
print("\nFetching 2026 current standings...")
current_season = fetch_season_data(2026)
current_df = pd.DataFrame(current_season)

current_df["win_probability"] = model.predict_proba(current_df[features])[:, 1]
current_df = current_df.sort_values("win_probability", ascending=False)

print("\n🏆 2026 Championship Predictions:")
print("─" * 40)
for _, row in current_df.iterrows():
    bar = "█" * int(row["win_probability"] * 20)
    print(f"{row['driver']:<6} {bar:<20} {row['win_probability']*100:.1f}%")
