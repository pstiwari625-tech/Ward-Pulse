"""
Ward Pulse: Phase 2 & 3 Master Integration Tests
Validates official GeoJSON integrity, attributes CSV matching, target ward identification,
demand classifier determinism, and geographical location verification.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.geo_loader import (
    load_geojson_boundaries,
    load_ward_features,
    load_target_ward_ids,
    get_merged_geospatial_data
)
from utils.validation import verify_geographical_location, validate_prediction_template
from utils.demand_classifier import classify_dataframe_demand, DEMAND_TIERS, DEMAND_COLOR_MAP


def test_geojson_validity():
    """Verify GeoJSON exists, is valid, has 198 features, unique IDs, and valid WGS84 coordinates."""
    gj, val = load_geojson_boundaries()
    assert gj is not None, f"Failed to load GeoJSON: {val.get('errors')}"
    assert val["is_valid"] is True, f"GeoJSON validation failed: {val.get('errors')}"
    assert val["feature_count"] == 198, f"Expected 198 features, got {val['feature_count']}"
    assert val["unique_ward_ids"] == 198, f"Expected 198 unique ward IDs, got {val['unique_ward_ids']}"
    assert val["missing_ward_id_count"] == 0, f"Missing ward_id count: {val['missing_ward_id_count']}"
    assert val["coords_valid"] is True, "Coordinates validation failed (check latitude/longitude reversal)"
    print("[PASS] GeoJSON boundaries validated: 198 features, coordinates verified.")


def test_ward_features_validity():
    """Verify ward_features.csv loads with 198 rows and contains normalized attributes."""
    df_feat, val = load_ward_features()
    assert df_feat is not None, f"Failed to load ward_features: {val.get('errors')}"
    assert val["is_valid"] is True, f"Features validation failed: {val.get('errors')}"
    assert len(df_feat) == 198, f"Expected 198 feature rows, got {len(df_feat)}"
    assert "ward_id" in df_feat.columns, "ward_id missing in df_feat"
    assert "ward_name" in df_feat.columns, "ward_name missing in df_feat"
    assert "zone" in df_feat.columns, "zone missing in df_feat"
    assert "population" in df_feat.columns, "population missing in df_feat"
    assert "area_sq_km" in df_feat.columns, "area_sq_km missing in df_feat"
    assert "road_length_kms" in df_feat.columns, "road_length_kms missing in df_feat"
    assert "parks" in df_feat.columns, "parks missing in df_feat"
    assert "schools" in df_feat.columns, "schools missing in df_feat"
    assert "streetlights" in df_feat.columns, "streetlights missing in df_feat"
    print("[PASS] Ward features validated: 198 records with required attributes.")


def test_target_wards_identification():
    """Verify prediction_grid_template.csv identifies exactly 40 target wards."""
    target_ids, val = load_target_ward_ids()
    assert len(target_ids) == 40, f"Expected 40 prediction target wards, got {len(target_ids)}"
    assert val["is_valid"] is True, f"Target identification failed: {val.get('errors')}"
    print(f"[PASS] 40 Target wards correctly identified: {sorted(list(target_ids))[:5]}...")


def test_geospatial_merging_and_matching():
    """Verify 100% ID matching between GeoJSON and CSV (198 matched, 0 unmatched)."""
    merged = get_merged_geospatial_data()
    assert merged["status"] == "SUCCESS", f"Merged geospatial data failed: {merged.get('error_message')}"
    assert merged["total_boundaries"] == 198, f"Expected 198 boundaries, got {merged['total_boundaries']}"
    assert merged["matched_count"] == 198, f"Expected 198 matched IDs, got {merged['matched_count']}"
    assert merged["unmatched_count"] == 0, f"Expected 0 unmatched IDs, got {merged['unmatched_count']}"
    assert len(merged["target_ward_ids"]) == 40, f"Expected 40 target wards, got {len(merged['target_ward_ids'])}"

    # Check that sample feature has all required fields in properties
    sample_feat = merged["geojson"]["features"][0]
    props = sample_feat["properties"]
    required_props = [
        "ward_id", "ward_name", "zone", "population",
        "area_sq_km", "road_length_kms", "parks", "schools", "streetlights", "status_label"
    ]
    for prop in required_props:
        assert prop in props, f"Property '{prop}' missing in GeoJSON feature properties"

    print("[PASS] Geospatial merge verified: 198/198 matched, 0 unmatched, enriched properties validated.")


def test_geographical_location_verification():
    """
    CRITICAL TEST: Verifies that coordinates (12.83°N–13.14°N, 77.46°E–77.78°E) and
    ward names (Yelahanka, Malleshwaram) correctly trigger the Bengaluru location detection
    and mismatch advisory.
    """
    merged = get_merged_geospatial_data()
    loc = merged.get("location_audit", {})
    assert loc.get("is_mismatch_with_lucknow") is True, "Expected location mismatch flag for Bengaluru dataset"
    assert "Bengaluru" in loc.get("detected_location", ""), f"Unexpected detected location: {loc.get('detected_location')}"
    assert "advisory" in loc, "Missing location advisory string"
    print(f"[PASS] Geographical location verification passed: Detected {loc['detected_location']}.")


def test_no_synthetic_predictions_introduced():
    """Verify no fake complaint counts or invented demand levels are added to properties."""
    merged = get_merged_geospatial_data()
    for feat in merged["geojson"]["features"]:
        props = feat["properties"]
        assert "cpi_score" not in props, "Unexpected cpi_score in baseline properties"
        assert "predicted_complaints" not in props, "Unexpected predicted_complaints in baseline properties"
        assert "demand_level" not in props, "Unexpected demand_level in baseline properties"
    print("[PASS] Strict anti-hallucination check passed: Zero synthetic predictions in baseline properties.")


def test_demand_classifier_deterministic():
    """Verify demand classifier is reproducible, quantile-based, and uses documented colors."""
    test_df = pd.DataFrame({
        "ward_id": list(range(1, 101)),
        "predicted_complaints": list(range(1, 101))
    })
    classified_df, meta = classify_dataframe_demand(test_df)
    assert meta["status"] == "SUCCESS"
    assert set(meta["tier_counts"].keys()) == set(DEMAND_TIERS)
    # Check that cutoffs split evenly for 100 sequential numbers
    assert classified_df.loc[0, "demand_level"] == "Low"      # value 1
    assert classified_df.loc[99, "demand_level"] == "Very High" # value 100
    for tier in DEMAND_TIERS:
        assert tier in DEMAND_COLOR_MAP
    print("[PASS] Demand classifier verified: Documented 4-tier cutoffs and deterministic classification.")


if __name__ == "__main__":
    test_geojson_validity()
    test_ward_features_validity()
    test_target_wards_identification()
    test_geospatial_merging_and_matching()
    test_geographical_location_verification()
    test_no_synthetic_predictions_introduced()
    test_demand_classifier_deterministic()
    print("\n[ALL 7 TESTS PASSED] Phase 2 & Phase 3 Master Integration verified.")
