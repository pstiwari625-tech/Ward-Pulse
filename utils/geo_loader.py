"""
Ward Pulse - Geospatial Data Loader & Spatial Validator
Phase 3: Official GeoJSON Map Integration

Loads and validates official hackathon organizer datasets:
- data/geospatial/wards.geojson (198 Ward Polygons)
- data/geospatial/ward_features.csv (Ward socio-economic & infrastructure features)
- data/geospatial/prediction_grid_template.csv (40 prediction target wards template)

Implements Streamlit caching to avoid repeatedly re-parsing large GeoJSON files.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set, List
import json
import pandas as pd
import logging
import streamlit as st

from utils.validation import verify_geographical_location

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def find_file_path(relative_paths: List[str]) -> Optional[Path]:
    """Search multiple candidate locations for a dataset file."""
    root = get_project_root()
    for rel in relative_paths:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_geojson_boundaries() -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Load official ward boundaries from wards.geojson and perform strict spatial validation.
    Cached via @st.cache_data for instant interactive UI rendering.
    
    Validates:
    - File existence and valid JSON syntax
    - Type is FeatureCollection
    - Feature count (expects 198 official wards)
    - Presence and normalization of 'ward_id' in every feature
    - Uniqueness of ward_id
    - Coordinate system integrity (WGS84 lon ~77, lat ~12-13, no reversal)
    """
    validation: Dict[str, Any] = {
        "is_valid": False,
        "feature_count": 0,
        "unique_ward_ids": 0,
        "missing_ward_id_count": 0,
        "coords_valid": False,
        "bounds": None,
        "centroid": None,
        "errors": [],
        "file_path": None
    }

    geojson_path = find_file_path([
        "data/geospatial/wards.geojson",
        "data/raw/wards.geojson",
        "wards.geojson"
    ])

    if not geojson_path:
        validation["errors"].append(
            "wards.geojson not found in data/geospatial/, data/raw/, or project root."
        )
        return None, validation

    validation["file_path"] = str(geojson_path)

    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
    except Exception as exc:
        validation["errors"].append(f"Failed to parse wards.geojson: {exc}")
        return None, validation

    if not isinstance(geojson_data, dict) or geojson_data.get("type") != "FeatureCollection":
        validation["errors"].append("GeoJSON root is not a valid FeatureCollection.")
        return None, validation

    features = geojson_data.get("features", [])
    validation["feature_count"] = len(features)

    if len(features) != 198:
        validation["errors"].append(f"Expected 198 features, found {len(features)}.")

    # Validate ward_ids and extract coordinates
    ward_ids: List[int] = []
    missing_id_count = 0
    all_lons: List[float] = []
    all_lats: List[float] = []

    for idx, feat in enumerate(features):
        props = feat.get("properties", {})
        wid = props.get("ward_id")

        if wid is None:
            missing_id_count += 1
            continue

        try:
            norm_id = int(wid)
            props["ward_id"] = norm_id  # Normalize to integer in-place
            ward_ids.append(norm_id)
        except (ValueError, TypeError):
            missing_id_count += 1
            continue

        # Coordinate inspection
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])

        def extract_pts(c):
            if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
                all_lons.append(float(c[0]))
                all_lats.append(float(c[1]))
            elif isinstance(c, (list, tuple)):
                for sub in c:
                    extract_pts(sub)

        extract_pts(coords)

    validation["missing_ward_id_count"] = missing_id_count
    validation["unique_ward_ids"] = len(set(ward_ids))

    if missing_id_count > 0:
        validation["errors"].append(f"{missing_id_count} features missing 'ward_id'.")
    if len(set(ward_ids)) != len(ward_ids):
        validation["errors"].append("Duplicate ward_id detected in GeoJSON features.")

    # Validate coordinates (Bangalore municipal bounds: lon ~77.4 - 77.8, lat ~12.8 - 13.2)
    if all_lons and all_lats:
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)

        # Check for longitude/latitude reversal
        if 70.0 <= min_lon <= 85.0 and 10.0 <= min_lat <= 20.0:
            validation["coords_valid"] = True
            validation["bounds"] = [[min_lat, min_lon], [max_lat, max_lon]]
            validation["centroid"] = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
        else:
            validation["coords_valid"] = False
            validation["errors"].append(
                f"Coordinates out of expected bounds or reversed: Lon [{min_lon:.2f}, {max_lon:.2f}], Lat [{min_lat:.2f}, {max_lat:.2f}]"
            )

    validation["is_valid"] = len(validation["errors"]) == 0
    return geojson_data, validation


@st.cache_data(show_spinner=False)
def load_ward_features() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load official ward demographic, infrastructure, and socio-economic attributes.
    Cached via @st.cache_data.
    
    Normalizes:
    - ward_id to integer
    - Provides standardized aliases: zone, population, area_sq_km, road_length_kms,
      parks, govt_schools, streetlights.
    """
    validation: Dict[str, Any] = {
        "is_valid": False,
        "row_count": 0,
        "columns": [],
        "errors": [],
        "file_path": None
    }

    csv_path = find_file_path([
        "data/geospatial/ward_features.csv",
        "data/raw/ward_features.csv",
        "ward_features.csv"
    ])

    if not csv_path:
        validation["errors"].append("ward_features.csv not found.")
        return None, validation

    validation["file_path"] = str(csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        validation["errors"].append(f"Failed to read ward_features.csv: {exc}")
        return None, validation

    validation["row_count"] = len(df)
    validation["columns"] = list(df.columns)

    if "ward_id" not in df.columns:
        validation["errors"].append("ward_features.csv is missing mandatory 'ward_id' column.")
        return None, validation

    # Normalize ward_id to integer
    try:
        df["ward_id"] = df["ward_id"].astype(int)
    except Exception as exc:
        validation["errors"].append(f"Failed to cast ward_id to integer: {exc}")
        return None, validation

    # Validate 1-to-1 uniqueness
    if not df["ward_id"].is_unique:
        validation["errors"].append("ward_id in ward_features.csv contains duplicate entries.")

    # Provide clean standardized aliases for UI presentation
    if "bbmp_zone_name" in df.columns and "zone" not in df.columns:
        df["zone"] = df["bbmp_zone_name"]
    elif "zone" not in df.columns:
        df["zone"] = "Standard Zone"

    if "population_2011" in df.columns and "population" not in df.columns:
        df["population"] = df["population_2011"]

    if "area_sq_km" not in df.columns:
        df["area_sq_km"] = 0.0

    if "road_length_kms" not in df.columns:
        df["road_length_kms"] = 0.0

    if "parks" not in df.columns:
        df["parks"] = 0

    if "govt_schools" in df.columns and "schools" not in df.columns:
        df["schools"] = df["govt_schools"]
    elif "schools" not in df.columns:
        df["schools"] = 0

    if "streetlights" not in df.columns:
        df["streetlights"] = 0

    # Ensure ward_name is populated
    if "ward_name" not in df.columns:
        df["ward_name"] = df["ward_id"].apply(lambda wid: f"Ward {wid}")
    else:
        df["ward_name"] = df["ward_name"].fillna(df["ward_id"].apply(lambda wid: f"Ward {wid}"))

    validation["is_valid"] = len(validation["errors"]) == 0
    return df, validation


@st.cache_data(show_spinner=False)
def load_target_ward_ids() -> Tuple[Set[int], Dict[str, Any]]:
    """
    Load official prediction target wards from prediction_grid_template.csv.
    Identifies the 40 wards that have zero historical complaint records.
    Cached via @st.cache_data.
    """
    validation: Dict[str, Any] = {
        "target_count": 0,
        "is_valid": False,
        "target_ids": [],
        "errors": []
    }

    template_path = find_file_path([
        "data/geospatial/prediction_grid_template.csv",
        "data/raw/prediction_grid_template.csv",
        "prediction_grid_template.csv"
    ])

    if not template_path:
        validation["errors"].append("prediction_grid_template.csv not found.")
        return set(), validation

    try:
        df_grid = pd.read_csv(template_path)
        if "ward_id" in df_grid.columns:
            target_ids = sorted(df_grid["ward_id"].astype(int).unique().tolist())
            validation["target_count"] = len(target_ids)
            validation["target_ids"] = target_ids
            validation["is_valid"] = len(target_ids) == 40
            return set(target_ids), validation
    except Exception as exc:
        validation["errors"].append(f"Failed to read prediction_grid_template.csv: {exc}")

    return set(), validation


@st.cache_data(show_spinner=False)
def get_merged_geospatial_data() -> Dict[str, Any]:
    """
    Master cached loader function:
    1. Loads GeoJSON boundaries.
    2. Loads ward_features.csv.
    3. Identifies the 40 target wards from prediction_grid_template.csv.
    4. Enriches GeoJSON feature properties with ward attributes.
    5. Calculates bounds and matched/unmatched statistics.
    6. Performs critical geographical location audit (Bengaluru BBMP vs Lucknow).
    """
    geojson_data, geo_val = load_geojson_boundaries()
    df_features, feat_val = load_ward_features()
    target_ids, target_val = load_target_ward_ids()

    result: Dict[str, Any] = {
        "status": "ERROR",
        "geojson": None,
        "df_features": None,
        "target_ward_ids": set(),
        "total_boundaries": 0,
        "matched_count": 0,
        "unmatched_count": 0,
        "bounds": [[12.834, 77.460], [13.144, 77.784]],
        "centroid": [12.9888, 77.6221],
        "location_audit": {
            "detected_location": "Unknown",
            "is_mismatch_with_lucknow": False,
            "advisory": ""
        },
        "validation_report": {
            "geojson": geo_val,
            "features": feat_val,
            "targets": target_val,
            "matched_ids": [],
            "unmatched_ids": []
        },
        "error_message": None
    }

    if geojson_data is None:
        result["error_message"] = "Official GeoJSON boundaries could not be loaded: " + "; ".join(geo_val["errors"])
        return result

    if df_features is None:
        result["error_message"] = "Ward features CSV could not be loaded: " + "; ".join(feat_val["errors"])
        return result

    # Check matches between GeoJSON and ward_features
    geo_ids = {int(f["properties"]["ward_id"]) for f in geojson_data.get("features", []) if "ward_id" in f.get("properties", {})}
    feat_ids = set(df_features["ward_id"].tolist())

    matched_ids = sorted(list(geo_ids.intersection(feat_ids)))
    unmatched_ids = sorted(list(geo_ids.symmetric_difference(feat_ids)))

    result["total_boundaries"] = len(geojson_data.get("features", []))
    result["matched_count"] = len(matched_ids)
    result["unmatched_count"] = len(unmatched_ids)
    result["validation_report"]["matched_ids"] = matched_ids
    result["validation_report"]["unmatched_ids"] = unmatched_ids

    # Location Verification
    sample_names = df_features["ward_name"].dropna().tolist()
    loc_audit = verify_geographical_location(
        geo_val.get("bounds", [[12.834, 77.460], [13.144, 77.784]]),
        sample_names
    )
    result["location_audit"] = loc_audit

    # Create fast index lookup from ward_features
    features_by_id = df_features.set_index("ward_id").to_dict(orient="index")

    # Enrich GeoJSON properties
    for feat in geojson_data.get("features", []):
        wid = int(feat["properties"]["ward_id"])
        row = features_by_id.get(wid, {})

        # Extract attributes safely with zero / fallback
        ward_name = str(row.get("ward_name", f"Ward {wid}"))
        zone = str(row.get("zone", row.get("bbmp_zone_name", "General Zone")))
        population = int(row.get("population", row.get("population_2011", 0)))
        area_sq_km = float(row.get("area_sq_km", 0.0))
        road_length_kms = float(row.get("road_length_kms", 0.0))
        parks = int(row.get("parks", 0))
        schools = int(row.get("schools", row.get("govt_schools", 0)))
        streetlights = int(row.get("streetlights", 0))

        is_target = wid in target_ids
        target_badge = "Prediction target – no historical complaint records" if is_target else "Official Ward (Historical data present)"

        # Store in GeoJSON properties for Folium tooltips and popups
        feat["properties"]["ward_id"] = wid
        feat["properties"]["ward_name"] = ward_name
        feat["properties"]["zone"] = zone
        feat["properties"]["population"] = f"{population:,}"
        feat["properties"]["raw_population"] = population
        feat["properties"]["area_sq_km"] = f"{area_sq_km:.2f}"
        feat["properties"]["raw_area"] = area_sq_km
        feat["properties"]["road_length_kms"] = f"{road_length_kms:.1f}"
        feat["properties"]["parks"] = parks
        feat["properties"]["schools"] = schools
        feat["properties"]["streetlights"] = f"{streetlights:,}"
        feat["properties"]["is_target"] = is_target
        feat["properties"]["status_label"] = target_badge

    # Set bounds and centroid
    if geo_val.get("bounds"):
        result["bounds"] = geo_val["bounds"]
    if geo_val.get("centroid"):
        result["centroid"] = geo_val["centroid"]

    result["geojson"] = geojson_data
    result["df_features"] = df_features
    result["target_ward_ids"] = target_ids
    result["status"] = "SUCCESS"
    return result
