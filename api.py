"""
F1 Prediction API
=================
Uses the trained model from model.py (f1_model.joblib)
Endpoints:
  GET /                                    - status
  GET /api/standings                       - current 2026 driver standings
  GET /api/constructors                    - current 2026 constructor standings
  GET /api/championship                    - ML championship predictions
  GET /api/schedule                        - 2026 race schedule
  GET /api/race/{round_num}                - pre-race win prediction
  GET /api/race/live/{round_num}           - live race prediction (OpenF1)
  GET /api/race/result/{round_num}         - store/get race result after finish
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import prediction helpers from model.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import (
    load_model, predict_championship,
    fetch_race_results, fetch_season_standings,
    fetch_constructor_standings, fetch_fastf1_data,
    STREET_CIRCUITS, HIGH_SPEED_CIRCUITS, TECHNICAL_CIRCUITS
)

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_YEAR     = 2026
COMPLETED_ROUNDS = 1   # UPDATE after each race

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
print("Loading ML model...")
model, scaler, meta = load_model()
if model is None:
    print("WARNING: No trained model found. Run python model.py --retrain first.")

# ─── SIMPLE CACHE ─────────────────────────────────────────────────────────────
_cache     = {}
CACHE_FILE = os.path.join(os.path.dirname(__file__), "api_cache.json")

def cache_get(key, max_age=300):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < max_age:
            return val
    return None

def cache_set(key, val):
    _cache[key] = (val, time.time())

# ─── DATA HELPERS ─────────────────────────────────────────────────────────────

def get_standings_2026():
    cached = cache_get("standings_2026", max_age=120)
    if cached:
        return cached
    try:
        r = requests.get(
            f"https://api.jolpi.ca/ergast/f1/{CURRENT_YEAR}/{COMPLETED_ROUNDS}/driverStandings.json",
            timeout=10
        )
        data  = r.json()
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return []
        result = lists[0]["DriverStandings"]
        cache_set("standings_2026", result)
        return result
    except Exception:
        return []

def get_constructors_2026():
    cached = cache_get("constructors_2026", max_age=120)
    if cached:
        return cached
    try:
        r = requests.get(
            f"https://api.jolpi.ca/ergast/f1/{CURRENT_YEAR}/{COMPLETED_ROUNDS}/constructorStandings.json",
            timeout=10
        )
        data  = r.json()
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return []
        result = lists[0]["ConstructorStandings"]
        cache_set("constructors_2026", result)
        return result
    except Exception:
        return []

def get_races_2026():
    cached = cache_get("races_2026", max_age=300)
    if cached:
        return cached
    result = fetch_race_results(CURRENT_YEAR)
    if result:
        cache_set("races_2026", result)
    return result

def get_qualifying(round_num):
    key    = f"quali_{round_num}"
    cached = cache_get(key, max_age=3600)
    if cached:
        return cached
    try:
        r = requests.get(
            f"https://api.jolpi.ca/ergast/f1/{CURRENT_YEAR}/{round_num}/qualifying.json",
            timeout=10
        )
        data  = r.json()
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return {}
        result = {
            res["Driver"]["code"]: int(res["position"])
            for res in races[0]["QualifyingResults"]
        }
        cache_set(key, result)
        return result
    except Exception:
        return {}


# ─── OPENF1 LIVE DATA ─────────────────────────────────────────────────────────

OPENF1 = "https://api.openf1.org/v1"

def get_live_session():
    """Get the current or most recent session from OpenF1."""
    try:
        r = requests.get(f"{OPENF1}/sessions?session_type=Race&year={CURRENT_YEAR}", timeout=8)
        sessions = r.json()
        if not sessions:
            return None
        # Return most recent
        return sorted(sessions, key=lambda s: s.get("date_start",""), reverse=True)[0]
    except Exception:
        return None

def get_live_positions(session_key):
    """Current race positions from OpenF1."""
    try:
        r = requests.get(f"{OPENF1}/position?session_key={session_key}", timeout=8)
        data = r.json()
        if not data:
            return {}
        # Get latest position for each driver
        latest = {}
        for entry in data:
            drv = entry.get("driver_number")
            if drv and (drv not in latest or entry["date"] > latest[drv]["date"]):
                latest[drv] = entry
        return {v["driver_number"]: v["position"] for v in latest.values()}
    except Exception:
        return {}

def get_live_intervals(session_key):
    """Gap to leader in seconds for each driver."""
    try:
        r = requests.get(f"{OPENF1}/intervals?session_key={session_key}", timeout=8)
        data = r.json()
        if not data:
            return {}
        latest = {}
        for entry in data:
            drv = entry.get("driver_number")
            if drv and (drv not in latest or entry["date"] > latest[drv]["date"]):
                latest[drv] = entry
        return {
            v["driver_number"]: v.get("gap_to_leader", 0) or 0
            for v in latest.values()
        }
    except Exception:
        return {}

def get_live_tyres(session_key):
    """Current tyre compound and age for each driver."""
    try:
        r = requests.get(f"{OPENF1}/stints?session_key={session_key}", timeout=8)
        data = r.json()
        if not data:
            return {}
        latest = {}
        for entry in data:
            drv = entry.get("driver_number")
            if drv and (drv not in latest or
                        entry.get("stint_number", 0) > latest[drv].get("stint_number", 0)):
                latest[drv] = entry
        return {
            v["driver_number"]: {
                "compound":  v.get("compound", "UNKNOWN"),
                "tyre_age":  v.get("tyre_age_at_start", 0) + v.get("lap_start", 0),
            }
            for v in latest.values()
        }
    except Exception:
        return {}

def get_live_laps(session_key):
    """Latest lap number and lap time for each driver."""
    try:
        r = requests.get(f"{OPENF1}/laps?session_key={session_key}", timeout=8)
        data = r.json()
        if not data:
            return {}
        latest = {}
        for entry in data:
            drv = entry.get("driver_number")
            if drv and (drv not in latest or
                        entry.get("lap_number", 0) > latest[drv].get("lap_number", 0)):
                latest[drv] = entry
        return {
            v["driver_number"]: {
                "lap":      v.get("lap_number", 0),
                "lap_time": v.get("lap_duration", None),
            }
            for v in latest.values()
        }
    except Exception:
        return {}

def get_driver_numbers(session_key):
    """Map driver number -> 3-letter code."""
    try:
        r = requests.get(f"{OPENF1}/drivers?session_key={session_key}", timeout=8)
        data = r.json()
        return {
            str(d["driver_number"]): d.get("name_acronym", str(d["driver_number"]))
            for d in data
        }
    except Exception:
        return {}

def get_race_control(session_key):
    """Check for safety car / VSC / red flag."""
    try:
        r = requests.get(f"{OPENF1}/race_control?session_key={session_key}", timeout=8)
        data = r.json()
        if not data:
            return "NONE"
        # Get latest flag/message
        latest = sorted(data, key=lambda x: x.get("date",""), reverse=True)
        for msg in latest[:10]:
            cat = msg.get("category","")
            flag = msg.get("flag","")
            if "SAFETY CAR" in cat.upper() or "SAFETY CAR" in flag.upper():
                return "SAFETY_CAR"
            if "VIRTUAL" in cat.upper() or "VSC" in flag.upper():
                return "VSC"
            if "RED" in flag.upper():
                return "RED_FLAG"
        return "NONE"
    except Exception:
        return "NONE"


# ─── LIVE SCORING ENGINE ──────────────────────────────────────────────────────

def compute_live_scores(
    pre_race_scores,   # dict: driver_code -> float (0-100 from pre-race ML)
    positions,         # dict: driver_number -> position
    intervals,         # dict: driver_number -> gap_to_leader (secs)
    tyres,             # dict: driver_number -> {compound, tyre_age}
    laps,              # dict: driver_number -> {lap, lap_time}
    driver_numbers,    # dict: driver_number -> driver_code
    total_laps,        # int: total laps in race
    race_control,      # str: NONE / SAFETY_CAR / VSC / RED_FLAG
):
    """
    Blend pre-race ML prediction with live race data.
    As race progresses, live data gets weighted more heavily.
    """
    if not positions or not driver_numbers:
        return pre_race_scores

    # Work out how far through the race we are
    laps_done   = max((v["lap"] for v in laps.values()), default=0) if laps else 0
    race_pct    = min(laps_done / max(total_laps, 1), 1.0)
    live_weight = race_pct         # 0% at start → 100% at end
    pre_weight  = 1.0 - race_pct

    # Median gap for normalisation
    gaps = [v for v in intervals.values() if isinstance(v, (int, float))]
    max_gap = max(gaps) if gaps else 60.0

    scores = {}
    for drv_num, drv_code in driver_numbers.items():
        pos      = positions.get(drv_num, 20)
        gap      = intervals.get(drv_num, max_gap)
        tyre_inf = tyres.get(drv_num, {})
        lap_inf  = laps.get(drv_num, {})

        # Position score — 1st = 100, 20th = 5
        pos_score = max(0, (21 - pos) / 20 * 100)

        # Gap score — closer to leader = better
        gap_score = max(0, (1 - gap / max(max_gap, 1)) * 100)

        # Tyre score — fresher tyres on faster compounds score higher
        compound_bonus = {"SOFT": 10, "MEDIUM": 5, "HARD": 0, "INTER": 3, "WET": 3}
        tyre_age       = tyre_inf.get("tyre_age", 20)
        compound       = tyre_inf.get("compound", "MEDIUM")
        tyre_score     = max(0, compound_bonus.get(compound, 5) - tyre_age * 0.3)

        # Safety car flattens the field (reduces gap advantage)
        if race_control in ("SAFETY_CAR", "VSC"):
            gap_score *= 0.3

        live_score = pos_score * 0.6 + gap_score * 0.3 + tyre_score * 0.1

        # Blend with pre-race ML prediction
        pre_score = pre_race_scores.get(drv_code, 30.0)
        scores[drv_code] = round(
            pre_weight * pre_score + live_weight * live_score, 2
        )

    # Normalise to 0-100
    if scores:
        mx = max(scores.values()) or 1
        scores = {k: round(v / mx * 100, 1) for k, v in scores.items()}

    return scores


# ─── PRE-RACE PREDICTION ──────────────────────────────────────────────────────

def build_prerace_prediction(round_num):
    """
    Uses the ML championship model + qualifying grid to predict race winner.
    Returns list of dicts sorted by win_probability.
    """
    if model is None:
        return []

    standings    = get_standings_2026()
    races        = get_races_2026()
    constructors = fetch_constructor_standings(CURRENT_YEAR)
    qualifying   = get_qualifying(round_num)

    if not standings or not races:
        return []

    # Get ML championship predictions as base scores
    champ_preds = predict_championship(
        standings, races, constructors,
        COMPLETED_ROUNDS, model, scaler, meta
    )
    if not champ_preds:
        return []

    # Score map: driver -> predicted championship points
    ml_scores = {p["driver"]: p["predicted_points"] for p in champ_preds}
    max_ml    = max(ml_scores.values()) or 1

    # Circuit type
    race_df      = pd.DataFrame(races)
    circuit_ids  = race_df["circuit"].unique()
    circuit_id   = circuit_ids[round_num - 1] if round_num <= len(circuit_ids) else ""
    is_street    = 1 if circuit_id in STREET_CIRCUITS    else 0
    is_high      = 1 if circuit_id in HIGH_SPEED_CIRCUITS else 0

    predictions = []
    for p in champ_preds:
        driver    = p["driver"]
        ml_norm   = ml_scores.get(driver, 0) / max_ml * 100

        # Qualifying boost: pole = +20, P2 = +15, P3 = +10 ... P20 = 0
        quali_pos   = qualifying.get(driver, 15)
        quali_boost = max(0, (21 - quali_pos) * 1.0)

        raw_score = ml_norm * 0.7 + quali_boost * 0.3
        predictions.append({
            "driver":             driver,
            "team":               p["team"],
            "current_points":     p["current_points"],
            "qualifying_position": quali_pos,
            "raw_score":          raw_score,
        })

    predictions.sort(key=lambda x: x["raw_score"], reverse=True)
    total = sum(p["raw_score"] for p in predictions) or 1
    for p in predictions:
        p["win_probability"] = round(p["raw_score"] / total * 100, 1)
        del p["raw_score"]

    return predictions


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    model_info = {
        "mae":      meta["mae"]      if meta else None,
        "cv_mae":   meta["cv_mae"]   if meta else None,
        "samples":  meta["samples"]  if meta else None,
        "trained":  meta["trained_at"] if meta else None,
    } if meta else {}
    return {"status": "F1 Prediction API running", "model": model_info}


@app.get("/api/standings")
def get_standings():
    standings = get_standings_2026()
    if not standings:
        return {"error": "Could not fetch standings"}
    return {
        "round": COMPLETED_ROUNDS,
        "standings": [
            {
                "position":  int(s["position"]),
                "driver":    s["Driver"]["code"],
                "name":      f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                "team":      s["Constructors"][0]["name"] if s.get("Constructors") else "",
                "points":    float(s["points"]),
                "wins":      int(s["wins"]),
            }
            for s in standings
        ]
    }


@app.get("/api/constructors")
def get_constructors():
    constructors = get_constructors_2026()
    if not constructors:
        return {"error": "Could not fetch constructor standings"}
    return {
        "round": COMPLETED_ROUNDS,
        "standings": [
            {
                "position":     int(c["position"]),
                "team":         c["Constructor"]["name"],
                "nationality":  c["Constructor"].get("nationality",""),
                "points":       float(c["points"]),
                "wins":         int(c["wins"]),
            }
            for c in constructors
        ]
    }


@app.get("/api/championship")
def championship_prediction():
    if model is None:
        return {"error": "Model not loaded. Run python model.py --retrain first."}

    standings    = get_standings_2026()
    races        = get_races_2026()
    constructors = fetch_constructor_standings(CURRENT_YEAR)

    if not standings or not races:
        return {"error": "Could not fetch 2026 data"}

    preds = predict_championship(
        standings, races, constructors,
        COMPLETED_ROUNDS, model, scaler, meta
    )

    return {
        "round":       COMPLETED_ROUNDS,
        "model_mae":   meta["mae"]    if meta else None,
        "model_cv":    meta["cv_mae"] if meta else None,
        "predictions": preds,
    }


@app.get("/api/schedule")
def get_schedule():
    try:
        r = requests.get(
            f"https://api.jolpi.ca/ergast/f1/{CURRENT_YEAR}.json",
            timeout=10
        )
        races = r.json()["MRData"]["RaceTable"]["Races"]
        return {
            "season": CURRENT_YEAR,
            "races": [
                {
                    "round":      int(race["round"]),
                    "name":       race["raceName"],
                    "circuit":    race["Circuit"]["circuitName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "country":    race["Circuit"]["Location"]["country"],
                    "date":       race["date"],
                    "completed":  int(race["round"]) <= COMPLETED_ROUNDS,
                }
                for race in races
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/race/{round_num}")
def prerace_prediction(round_num: int):
    """Pre-race win probability using ML model + qualifying grid."""
    preds = build_prerace_prediction(round_num)
    if not preds:
        return {"error": "Could not build prediction"}
    return {
        "round":       round_num,
        "state":       "pre_race",
        "predictions": preds,
    }


@app.get("/api/race/live/{round_num}")
def live_race_prediction(round_num: int):
    """
    Live race win prediction. Blends pre-race ML with OpenF1 live data.
    If race hasn't started yet, returns pre-race prediction.
    If race is finished, returns final result.
    """
    # Always build pre-race as baseline
    pre_preds = build_prerace_prediction(round_num)
    pre_scores = {p["driver"]: p["win_probability"] for p in pre_preds}

    # Try to get live session
    session = get_live_session()
    if not session:
        return {
            "round":       round_num,
            "state":       "pre_race",
            "live":        False,
            "predictions": pre_preds,
        }

    session_key  = session.get("session_key")
    total_laps   = session.get("total_laps", 57)
    session_name = session.get("session_name", "")

    # Get all live data in parallel-ish
    drv_numbers  = get_driver_numbers(session_key)
    positions    = get_live_positions(session_key)
    intervals    = get_live_intervals(session_key)
    tyres        = get_live_tyres(session_key)
    laps         = get_live_laps(session_key)
    race_control = get_race_control(session_key)

    laps_done = max((v["lap"] for v in laps.values()), default=0) if laps else 0
    race_pct  = min(laps_done / max(total_laps, 1), 1.0)

    if laps_done == 0:
        state = "pre_race"
    elif race_pct >= 0.99:
        state = "finished"
    else:
        state = "live"

    # Compute blended scores
    live_scores = compute_live_scores(
        pre_scores, positions, intervals,
        tyres, laps, drv_numbers, total_laps, race_control
    )

    # Build response — merge live scores with driver info from pre-race
    driver_info = {p["driver"]: p for p in pre_preds}
    predictions = []
    for drv_num, drv_code in drv_numbers.items():
        info     = driver_info.get(drv_code, {})
        lap_info = laps.get(drv_num, {})
        tyre_inf = tyres.get(drv_num, {})
        predictions.append({
            "driver":             drv_code,
            "team":               info.get("team", ""),
            "position":           positions.get(drv_num, 99),
            "gap_to_leader":      intervals.get(drv_num, 0),
            "lap":                lap_info.get("lap", 0),
            "tyre_compound":      tyre_inf.get("compound", ""),
            "tyre_age":           tyre_inf.get("tyre_age", 0),
            "win_probability":    live_scores.get(drv_code, 0),
            "pre_race_prob":      pre_scores.get(drv_code, 0),
            "current_points":     info.get("current_points", 0),
        })

    predictions.sort(key=lambda x: x["position"])

    return {
        "round":          round_num,
        "state":          state,
        "live":           state == "live",
        "laps_done":      laps_done,
        "total_laps":     total_laps,
        "race_control":   race_control,
        "race_progress":  round(race_pct * 100, 1),
        "predictions":    predictions,
    }


@app.get("/api/race/result/{round_num}")
def race_result(round_num: int):
    """Get actual race result after the race finishes."""
    try:
        r = requests.get(
            f"https://api.jolpi.ca/ergast/f1/{CURRENT_YEAR}/{round_num}/results.json",
            timeout=10
        )
        races = r.json()["MRData"]["RaceTable"]["Races"]
        if not races:
            return {"error": "No result yet"}
        results = races[0]["Results"]
        return {
            "round":  round_num,
            "name":   races[0]["raceName"],
            "result": [
                {
                    "position": int(r["position"]),
                    "driver":   r["Driver"]["code"],
                    "team":     r["Constructor"]["name"],
                    "points":   float(r["points"]),
                    "status":   r["status"],
                    "fastest_lap": r.get("FastestLap", {}).get("Time", {}).get("time", ""),
                }
                for r in results[:10]
            ]
        }
    except Exception as e:
        return {"error": str(e)}
        