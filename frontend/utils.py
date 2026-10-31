"""
Ward Pulse: Frontend Data Utilities
Safely loads backend contracts, GeoJSON boundaries, and model metrics.
"""

from pathlib import Path
import json
import csv

def load_summary_data():
    """Loads ward_summary_latest.csv."""
    path = Path("data/predictions/ward_summary_latest.csv")
    if not path.exists():
        return []
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_metrics_data():
    """Loads model_metrics.json."""
    path = Path("models/model_metrics.json")
    if not path.exists():
        return {}
    with open(path, mode="r", encoding="utf-8") as f:
        return json.load(f)

def load_metadata_data():
    """Loads model_metadata.json."""
    path = Path("models/model_metadata.json")
    if not path.exists():
        return {}
    with open(path, mode="r", encoding="utf-8") as f:
        return json.load(f)

def load_geojson_data():
    """Loads wards.geojson."""
    paths = [Path("data/raw/wards.geojson"), Path("wards.geojson")]
    for p in paths:
        if p.exists():
            with open(p, mode="r", encoding="utf-8") as f:
                return json.load(f)
    return None
