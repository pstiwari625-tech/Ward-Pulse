"""
Ward Pulse: Backend Ingestion Layer
Loads raw CSV and GeoJSON datasets with validation.
"""

from pathlib import Path
import json
import csv
import config

def load_ward_features(filepath: Path = config.WARD_FEATURES_FILE):
    """Loads ward_features.csv with standard dict reader or pandas if installed."""
    try:
        import pandas as pd
        return pd.read_csv(filepath)
    except ImportError:
        with open(filepath, mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

def load_grievances(filepath: Path = config.GRIEVANCES_FILE):
    """Loads historical grievance records."""
    try:
        import pandas as pd
        return pd.read_csv(filepath)
    except ImportError:
        with open(filepath, mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

def load_prediction_grid(filepath: Path = config.PREDICTION_GRID_FILE):
    """Loads prediction grid template."""
    try:
        import pandas as pd
        return pd.read_csv(filepath)
    except ImportError:
        with open(filepath, mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

def load_geojson(filepath: Path = config.GEOJSON_FILE):
    """Loads ward boundary GeoJSON."""
    try:
        import geopandas as gpd
        return gpd.read_file(filepath)
    except ImportError:
        with open(filepath, mode="r", encoding="utf-8") as f:
            return json.load(f)
