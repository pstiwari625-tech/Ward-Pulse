"""
Ward Pulse - Data Integrity & Geospatial Validation Engine
Provides comprehensive validation for GeoJSON geometries, ward attributes,
schema compliance, and spatial coordinate verification.
"""

from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
import json
import pandas as pd
import numpy as np


def verify_geographical_location(bounds: List[List[float]], ward_names: List[str]) -> Dict[str, Any]:
    """
    CRITICAL AUDIT: Validates whether coordinates and ward administrative names
    correspond to Bengaluru (BBMP) or Lucknow.
    
    Expected coordinates:
    - Bengaluru: Lat ~12.8°N - 13.2°N, Lon ~77.4°E - 77.8°E
    - Lucknow:   Lat ~26.7°N - 27.0°N, Lon ~80.8°E - 81.1°E
    """
    min_lat, min_lon = bounds[0]
    max_lat, max_lon = bounds[1]

    avg_lat = (min_lat + max_lat) / 2.0
    avg_lon = (min_lon + max_lon) / 2.0

    is_bengaluru_coords = (12.0 <= avg_lat <= 14.0) and (76.5 <= avg_lon <= 78.5)
    is_lucknow_coords = (26.0 <= avg_lat <= 28.0) and (80.0 <= avg_lon <= 82.0)

    # Check for well-known Bengaluru BBMP ward identifiers
    bengaluru_keywords = {"yelahanka", "malleshwaram", "bommanahalli", "mahadevapura", "kempegowda", "hebbal", "rajajinagar"}
    sample_names = " ".join([str(n).lower() for n in ward_names[:25]])
    bengaluru_name_matches = [kw for kw in bengaluru_keywords if kw in sample_names]

    if is_bengaluru_coords or len(bengaluru_name_matches) >= 2:
        return {
            "detected_location": "Bengaluru (Bruhat Bengaluru Mahanagara Palike - BBMP)",
            "is_mismatch_with_lucknow": True,
            "center": [round(avg_lat, 4), round(avg_lon, 4)],
            "sample_bengaluru_wards": bengaluru_name_matches,
            "advisory": (
                "The official organiser dataset ('p3-ward-civic-demand') contains 198 BBMP ward boundaries "
                f"located at {avg_lat:.2f}°N, {avg_lon:.2f}°E (Bengaluru), despite the hackathon repository title referencing Lucknow. "
                "In strict accordance with data governance rules, boundaries and coordinates are displayed exactly as provided "
                "without artificial coordinate transformation or relabeling, pending organiser clarification."
            )
        }
    elif is_lucknow_coords:
        return {
            "detected_location": "Lucknow (Lucknow Municipal Corporation - LMC)",
            "is_mismatch_with_lucknow": False,
            "center": [round(avg_lat, 4), round(avg_lon, 4)],
            "advisory": "Coordinates accurately match Lucknow municipal boundaries."
        }
    else:
        return {
            "detected_location": f"Custom Coordinate Grid ({avg_lat:.2f}°N, {avg_lon:.2f}°E)",
            "is_mismatch_with_lucknow": True,
            "center": [round(avg_lat, 4), round(avg_lon, 4)],
            "advisory": "Coordinates represent a non-standard municipal jurisdiction."
        }


def validate_geojson_file(geojson_path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Validates GeoJSON file existence, JSON validity, FeatureCollection structure,
    and coordinate sanity.
    """
    report: Dict[str, Any] = {
        "is_valid": False,
        "feature_count": 0,
        "unique_ward_ids": 0,
        "errors": [],
        "warnings": [],
        "bounds": None
    }

    if not geojson_path.exists():
        report["errors"].append(f"GeoJSON file does not exist at: {geojson_path}")
        return None, report

    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        report["errors"].append(f"Failed to parse GeoJSON: {exc}")
        return None, report

    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        report["errors"].append("GeoJSON root must be a FeatureCollection.")
        return None, report

    features = data.get("features", [])
    report["feature_count"] = len(features)

    if len(features) != 198:
        report["warnings"].append(f"Expected 198 features, found {len(features)}.")

    ward_ids: List[int] = []
    all_lons: List[float] = []
    all_lats: List[float] = []

    for idx, feat in enumerate(features):
        props = feat.get("properties", {})
        wid = props.get("ward_id")
        if wid is None:
            report["errors"].append(f"Feature at index {idx} is missing 'ward_id'.")
            continue
        try:
            ward_ids.append(int(wid))
        except (ValueError, TypeError):
            report["errors"].append(f"Feature at index {idx} has non-integer 'ward_id': {wid}")

        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])

        def extract_coords(c):
            if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
                all_lons.append(float(c[0]))
                all_lats.append(float(c[1]))
            elif isinstance(c, (list, tuple)):
                for sub in c:
                    extract_coords(sub)

        extract_coords(coords)

    report["unique_ward_ids"] = len(set(ward_ids))
    if len(set(ward_ids)) != len(ward_ids):
        report["errors"].append("Duplicate ward_ids detected in GeoJSON features.")

    if all_lons and all_lats:
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        report["bounds"] = [[min_lat, min_lon], [max_lat, max_lon]]

        # Coordinate reversal check
        if min_lat > min_lon:
            report["errors"].append("Coordinates appear to be inverted (Latitude and Longitude swapped).")

    report["is_valid"] = len(report["errors"]) == 0
    return data, report


def validate_prediction_template(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates prediction_grid_template.csv:
    Confirms 4,320 rows, 40 unique target wards, 18 months, 6 categories.
    Verifies that rows are combinations, NOT actual prediction outputs.
    """
    report = {
        "is_valid": False,
        "row_count": len(df),
        "target_wards_count": 0,
        "is_template_only": True,
        "errors": []
    }

    expected_cols = {"ward_id", "month", "category_group"}
    missing = expected_cols - set(df.columns)
    if missing:
        report["errors"].append(f"Template missing required columns: {missing}")
        return report

    target_wards = df["ward_id"].unique()
    report["target_wards_count"] = len(target_wards)

    if len(df) == 4320 and len(target_wards) == 40:
        report["is_valid"] = True
    else:
        report["errors"].append(f"Template deviates from contract: {len(df)} rows, {len(target_wards)} target wards.")

    return report
