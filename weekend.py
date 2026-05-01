"""
Progressive Race Weekend Prediction Pipeline
============================================
Builds increasingly accurate predictions as the weekend unfolds.
Each completed session adds new data and updates the blended prediction.
"""

import os
import json
import time
import requests
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

OPENF1  = "https://api.openf1.org/v1"
JOLPICA = "https://api.jolpi.ca/ergast/f1"

WEEKEND_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weekend_cache")
os.makedirs(WEEKEND_CACHE_DIR, exist_ok=True)

# OpenF1 session_name → internal key
OPENF1_NAME_TO_KEY = {
    "Practice 1":        "fp1",
    "Practice 2":        "fp2",
    "Practice 3":        "fp3",
    "Sprint Qualifying": "sprint_quali",
    "Sprint":            "sprint",
    "Qualifying":        "qualifying",
}

SESSION_DISPLAY = {
    "fp1":          "Practice 1",
    "fp2":          "Practice 2",
    "fp3":          "Practice 3",
    "sprint_quali": "Sprint Qualifying",
    "sprint":       "Sprint Race",
    "qualifying":   "Qualifying",
    "base":         "Base ML",
}

# Order sessions chronologically for a race weekend
SESSION_ORDER = ["fp1", "fp2", "fp3", "sprint_quali", "sprint", "qualifying"]

CONFIDENCE_BY_SESSION = {
    "base":         0.45,
    "fp1":          0.55,
    "fp2":          0.65,
    "fp3":          0.72,
    "sprint_quali": 0.78,
    "sprint":       0.82,
    "qualifying":   0.92,
}

# Blend weights per stage. Weights in each list must sum to 1.0.
# Sources: "ml", "fp1", "fp2", "fp3", "practice" (avg of all fp), "sprint_quali", "sprint", "qualifying"
BLEND_CONFIG: Dict[str, List[Tuple[str, float]]] = {
    "base":         [("ml", 1.00)],
    "fp1":          [("ml", 0.70), ("fp1", 0.30)],
    "fp2":          [("ml", 0.60), ("fp1", 0.20), ("fp2", 0.20)],
    "fp3":          [("ml", 0.50), ("fp1", 0.10), ("fp2", 0.20), ("fp3", 0.20)],
    "sprint_quali": [("ml", 0.40), ("practice", 0.20), ("sprint_quali", 0.40)],
    "sprint":       [("ml", 0.30), ("practice", 0.15), ("sprint_quali", 0.20), ("sprint", 0.35)],
    "qualifying":   [("ml", 0.30), ("practice", 0.10), ("qualifying", 0.60)],
}


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 15) -> Optional[list]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_past(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt < datetime.now(timezone.utc)
    except Exception:
        return False


def normalize_to_100(scores: Dict[str, float]) -> Dict[str, float]:
    """Linearly rescale a dict of floats so min→0 and max→100."""
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 50.0 for k in scores}
    return {k: round((v - mn) / (mx - mn) * 100, 2) for k, v in scores.items()}


# ─── OPENF1 SESSION DISCOVERY ─────────────────────────────────────────────────

def get_driver_map(session_key: int) -> Dict[str, str]:
    """Returns {driver_number_str: driver_code (3-letter acronym)}."""
    data = _get(f"{OPENF1}/drivers?session_key={session_key}")
    if not data:
        return {}
    return {
        str(d["driver_number"]): d.get("name_acronym", str(d["driver_number"]))
        for d in data
    }


def get_meeting_key(year: int, round_num: int) -> Optional[int]:
    """Resolve a Jolpica round number to an OpenF1 meeting_key."""
    # Fetch race info from Jolpica to get country/locality for matching
    schedule = _get(f"{JOLPICA}/{year}/{round_num}.json")
    if not schedule:
        return None
    try:
        race    = schedule["MRData"]["RaceTable"]["Races"][0]
        country = race["Circuit"]["Location"]["country"].lower()
        locality = race["Circuit"]["Location"]["locality"].lower()
        race_name = race["raceName"].lower()
    except (KeyError, IndexError):
        return None

    meetings = _get(f"{OPENF1}/meetings?year={year}")
    if not meetings:
        return None

    # Try exact country match
    for m in meetings:
        if m.get("country_name", "").lower() == country:
            return m["meeting_key"]

    # Try locality match
    for m in meetings:
        if m.get("location", "").lower() == locality:
            return m["meeting_key"]

    # Try partial name match on significant words (>4 chars)
    words = [w for w in race_name.split() if len(w) > 4]
    for m in meetings:
        m_name = m.get("meeting_name", "").lower()
        if any(w in m_name for w in words):
            return m["meeting_key"]

    # Fallback: sort by date and use round index
    sorted_m = sorted(meetings, key=lambda m: m.get("date_start", ""))
    if 1 <= round_num <= len(sorted_m):
        return sorted_m[round_num - 1]["meeting_key"]

    return None


def get_race_name(year: int, round_num: int) -> str:
    """Get the race name for a given year/round from Jolpica."""
    data = _get(f"{JOLPICA}/{year}/{round_num}.json")
    if not data:
        return f"Round {round_num}"
    try:
        return data["MRData"]["RaceTable"]["Races"][0]["raceName"]
    except (KeyError, IndexError):
        return f"Round {round_num}"


def get_completed_sessions(year: int, round_num: int) -> Tuple[Optional[int], Dict[str, dict]]:
    """
    Returns (meeting_key, {session_type: {session_key, date_start, date_end}})
    for all sessions whose date_end is in the past.
    """
    meeting_key = get_meeting_key(year, round_num)
    if not meeting_key:
        return None, {}

    sessions = _get(f"{OPENF1}/sessions?meeting_key={meeting_key}") or []
    completed: Dict[str, dict] = {}

    for s in sessions:
        name = s.get("session_name", "")
        key  = OPENF1_NAME_TO_KEY.get(name)
        if not key:
            continue
        date_end = s.get("date_end") or s.get("date_start")
        if date_end and _is_past(date_end):
            completed[key] = {
                "session_key": s["session_key"],
                "date_start":  s.get("date_start", ""),
                "date_end":    date_end,
            }

    return meeting_key, completed


# ─── PRACTICE SESSION PACE EXTRACTION ─────────────────────────────────────────

def fetch_practice_scores(session_key: int, driver_map: Dict[str, str]) -> Dict[str, float]:
    """
    Extract pace scores from a practice session.
    Filters outlaps, inlaps, and invalid laps.
    Returns normalized {driver_code: 0-100} (higher = faster relative to field).
    """
    laps_data   = _get(f"{OPENF1}/laps?session_key={session_key}")
    stints_data = _get(f"{OPENF1}/stints?session_key={session_key}") or []

    if not laps_data:
        return {}

    # Build set of (driver, lap_number) to exclude (outlaps and inlaps)
    excluded: set = set()
    for stint in stints_data:
        drv = stint.get("driver_number")
        if not drv:
            continue
        if stint.get("lap_start"):
            excluded.add((drv, stint["lap_start"]))   # outlap
        if stint.get("lap_end"):
            excluded.add((drv, stint["lap_end"]))      # inlap

    # Collect clean laps per driver
    driver_laps: Dict[int, List[float]] = {}
    for lap in laps_data:
        drv      = lap.get("driver_number")
        lap_num  = lap.get("lap_number")
        duration = lap.get("lap_duration")

        if not drv or not duration or duration <= 0:
            continue
        if lap.get("is_pit_out_lap"):
            continue
        if (drv, lap_num) in excluded:
            continue
        if duration > 300:   # ignore obviously invalid laps (>5 min)
            continue

        driver_laps.setdefault(drv, []).append(duration)

    if not driver_laps:
        return {}

    all_laps     = [t for laps in driver_laps.values() for t in laps]
    field_median = float(np.median(all_laps))

    # pace_score = (field_median - best_lap) / field_median * 100
    # Positive means faster than field average
    raw: Dict[str, float] = {}
    for drv_num, times in driver_laps.items():
        best = min(times)
        code = driver_map.get(str(drv_num), str(drv_num))
        raw[code] = (field_median - best) / field_median * 100

    return normalize_to_100(raw)


# ─── QUALIFYING DATA ──────────────────────────────────────────────────────────

def fetch_qualifying_scores(year: int, round_num: int) -> Dict[str, float]:
    """
    Grid positions from Jolpica qualifying results.
    Returns {driver_code: qualifying_boost} where Pole=100, P20=5.
    Formula: (21 - grid_pos) / 20 * 100
    """
    data = _get(f"{JOLPICA}/{year}/{round_num}/qualifying.json")
    if not data:
        return {}
    try:
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return {}
        scores: Dict[str, float] = {}
        for res in races[0]["QualifyingResults"]:
            code = res["Driver"]["code"]
            pos  = int(res["position"])
            scores[code] = (21 - pos) / 20 * 100
        return scores
    except (KeyError, IndexError, ValueError):
        return {}


# ─── SPRINT SESSION DATA ──────────────────────────────────────────────────────

def fetch_sprint_quali_scores(session_key: int, driver_map: Dict[str, str]) -> Dict[str, float]:
    """
    Sprint qualifying pace from OpenF1 lap data.
    Ranks drivers by best lap time, returns {driver_code: 0-100}.
    """
    laps_data = _get(f"{OPENF1}/laps?session_key={session_key}")
    if not laps_data:
        return {}

    best_laps: Dict[int, float] = {}
    for lap in laps_data:
        drv      = lap.get("driver_number")
        duration = lap.get("lap_duration")
        if not drv or not duration or duration <= 0 or duration > 300:
            continue
        if lap.get("is_pit_out_lap"):
            continue
        if drv not in best_laps or duration < best_laps[drv]:
            best_laps[drv] = duration

    if not best_laps:
        return {}

    sorted_drivers = sorted(best_laps.items(), key=lambda x: x[1])
    n = len(sorted_drivers)
    scores: Dict[str, float] = {}
    for rank, (drv_num, _) in enumerate(sorted_drivers, 1):
        code = driver_map.get(str(drv_num), str(drv_num))
        scores[code] = (n + 1 - rank) / n * 100

    return scores


def fetch_sprint_result_scores(year: int, round_num: int) -> Dict[str, float]:
    """
    Sprint race results from Jolpica.
    Returns {driver_code: sprint_score} where P1=100, P20=5.
    """
    data = _get(f"{JOLPICA}/{year}/{round_num}/sprint.json")
    if not data:
        return {}
    try:
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return {}
        scores: Dict[str, float] = {}
        for res in races[0].get("SprintResults", []):
            code = res["Driver"]["code"]
            pos  = int(res["position"])
            scores[code] = (21 - pos) / 20 * 100
        return scores
    except (KeyError, IndexError, ValueError):
        return {}


# ─── PREDICTION BLENDING ─────────────────────────────────────────────────────

def get_latest_session(completed: List[str]) -> str:
    """Return the last session in SESSION_ORDER that appears in completed, or 'base'."""
    for s in reversed(SESSION_ORDER):
        if s in completed:
            return s
    return "base"


def blend_predictions(
    ml_scores: Dict[str, float],
    session_scores: Dict[str, Dict[str, float]],
    latest_session: str,
    all_drivers: List[str],
) -> Dict[str, float]:
    """
    Weighted blend of ML base + session scores, all on 0-100 scale.
    Falls back to ML score for drivers missing from a session source.
    Returns blended scores normalized to 0-100.
    """
    config = BLEND_CONFIG.get(latest_session, BLEND_CONFIG["base"])

    # Pre-compute practice average (fp1/fp2/fp3 average)
    practice_keys     = [k for k in ("fp1", "fp2", "fp3") if k in session_scores]
    has_practice_avg  = "practice" in dict(config) and practice_keys

    blended: Dict[str, float] = {}
    for drv in all_drivers:
        total = 0.0
        for source, weight in config:
            if source == "ml":
                score = ml_scores.get(drv, 50.0)
            elif source == "practice":
                if practice_keys:
                    vals  = [session_scores[k].get(drv, ml_scores.get(drv, 50.0)) for k in practice_keys]
                    score = float(np.mean(vals))
                else:
                    score = ml_scores.get(drv, 50.0)
            else:
                score = session_scores.get(source, {}).get(drv, ml_scores.get(drv, 50.0))
            total += weight * score
        blended[drv] = total

    return normalize_to_100(blended)


# ─── CACHE HELPERS ────────────────────────────────────────────────────────────

def _cache_path(year: int, round_num: int) -> str:
    return os.path.join(WEEKEND_CACHE_DIR, f"{year}_{round_num:02d}.json")


def load_weekend_state(year: int, round_num: int) -> Optional[dict]:
    path = _cache_path(year, round_num)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_weekend_state(year: int, round_num: int, state: dict) -> None:
    path = _cache_path(year, round_num)
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def build_weekend_prediction(
    year: int,
    round_num: int,
    ml_base_scores: Dict[str, float],    # driver_code → predicted championship points
    driver_info: Dict[str, dict],         # driver_code → {team, current_points, ...}
    race_name: str = "",
    force_refresh: bool = False,
) -> dict:
    """
    Build the full progressive weekend prediction.

    1. Check cache (300 s TTL unless force_refresh).
    2. Normalize ML scores to 0-100.
    3. Discover completed sessions from OpenF1.
    4. Fetch and compute scores for each completed session.
    5. Blend all sources using stage-appropriate weights.
    6. Persist to cache and return.
    """
    if not force_refresh:
        state = load_weekend_state(year, round_num)
        if state:
            age = time.time() - state.get("_cache_ts", 0)
            if age < 300:
                return _strip_internal(state)

    if not ml_base_scores:
        return {"error": "ML model not available", "year": year, "round": round_num}

    # Normalize ML scores to 0-100
    max_ml = max(ml_base_scores.values()) or 1
    ml_normalized = {drv: round(v / max_ml * 100, 2) for drv, v in ml_base_scores.items()}
    all_drivers   = list(ml_base_scores.keys())

    # Discover completed sessions from OpenF1
    meeting_key, completed_info = get_completed_sessions(year, round_num)
    sessions_completed = [s for s in SESSION_ORDER if s in completed_info]
    latest_session     = get_latest_session(sessions_completed)

    # Fetch scores for each completed session
    session_scores: Dict[str, Dict[str, float]] = {}
    for sess_key in SESSION_ORDER:
        if sess_key not in completed_info:
            continue
        info = completed_info[sess_key]
        sk   = info["session_key"]

        if sess_key in ("fp1", "fp2", "fp3"):
            drv_map = get_driver_map(sk)
            scores  = fetch_practice_scores(sk, drv_map)
        elif sess_key == "qualifying":
            scores = fetch_qualifying_scores(year, round_num)
        elif sess_key == "sprint_quali":
            drv_map = get_driver_map(sk)
            scores  = fetch_sprint_quali_scores(sk, drv_map)
        elif sess_key == "sprint":
            scores = fetch_sprint_result_scores(year, round_num)
        else:
            scores = {}

        if scores:
            session_scores[sess_key] = scores

    # Blend all sources
    blended    = blend_predictions(ml_normalized, session_scores, latest_session, all_drivers)
    total_sc   = sum(blended.values()) or 1
    confidence = CONFIDENCE_BY_SESSION.get(latest_session, 0.45)

    # Build per-driver prediction rows
    predictions = []
    for drv in all_drivers:
        info     = driver_info.get(drv, {})
        b_score  = blended.get(drv, 0.0)
        per_sess = {"ml_score": round(ml_normalized.get(drv, 0), 1)}
        for sk, sscores in session_scores.items():
            per_sess[f"{sk}_score"] = round(sscores.get(drv, 0), 1)

        predictions.append({
            "driver":              drv,
            "team":                info.get("team", ""),
            "current_points":      info.get("current_points", 0),
            "qualifying_position": info.get("qualifying_position"),
            "win_probability":     round(b_score / total_sc * 100, 1),
            "blended_score":       round(b_score, 1),
            "session_scores":      per_sess,
        })

    predictions.sort(key=lambda x: x["win_probability"], reverse=True)

    state = {
        "year":                  year,
        "round":                 round_num,
        "race_name":             race_name or get_race_name(year, round_num),
        "meeting_key":           meeting_key,
        "sessions_completed":    sessions_completed,
        "latest_session":        latest_session,
        "latest_session_name":   SESSION_DISPLAY.get(latest_session, "Base ML"),
        "predictions":           predictions,
        "prediction_confidence": round(confidence, 2),
        "last_updated":          _now_utc(),
        "_cache_ts":             time.time(),
    }

    save_weekend_state(year, round_num, state)
    return _strip_internal(state)


def _strip_internal(state: dict) -> dict:
    """Remove internal fields before returning to the API caller."""
    return {k: v for k, v in state.items() if not k.startswith("_")}
