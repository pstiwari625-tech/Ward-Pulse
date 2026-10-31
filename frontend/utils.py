"""
Ward Pulse: Frontend Data Utilities (frontend/utils.py)
Safely loads backend contracts, GeoJSON boundaries, intervals, and model metrics.
"""

from pathlib import Path
import json
import csv
import config

def load_summary_data():
    """Loads ward_summary_latest.csv."""
    path = config.WARD_SUMMARY_FILE
    if not path.exists():
        return []
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_predictions_data():
    """Loads ward_predictions.csv (4,320 rows)."""
    path = config.WARD_PREDICTIONS_FILE
    if not path.exists():
        return []
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_intervals_data():
    """Loads ward_predictions_with_intervals.csv (4,320 rows)."""
    path = config.WARD_PREDICTIONS_INTERVALS_FILE
    if not path.exists():
        return []
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_summary_intervals_data():
    """Loads ward_summary_intervals.csv (40 wards)."""
    path = config.WARD_SUMMARY_INTERVALS_FILE
    if not path.exists():
        return []
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_metrics_data():
    """Loads metrics_latest.json (reconciled benchmarks)."""
    path = config.DATA_DIR / "metrics" / "metrics_latest.json"
    if not path.exists():
        path = config.MODEL_METRICS_FILE
    if not path.exists():
        return {}
    with open(path, mode="r", encoding="utf-8") as f:
        return json.load(f)

def load_priority_sensitivity():
    """Loads priority_sensitivity.json."""
    path = config.DATA_DIR / "metrics" / "priority_sensitivity.json"
    if not path.exists():
        return {}
    with open(path, mode="r", encoding="utf-8") as f:
        return json.load(f)

def load_geojson_data():
    """Loads local wards.geojson without network requests."""
    paths = [config.GEOJSON_FILE, Path("data/raw/wards.geojson"), Path("wards.geojson")]
    for p in paths:
        if p.exists():
            with open(p, mode="r", encoding="utf-8") as f:
                return json.load(f)
    return None
