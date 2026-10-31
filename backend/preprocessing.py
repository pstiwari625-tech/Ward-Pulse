"""
Ward Pulse: Backend Preprocessing Layer
Cleans numeric fields and aggregates raw complaints into a balanced monthly panel.
"""

import re
from typing import Dict, Any

def clean_ward_feature_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Cleans numeric artifacts (commas, percentages) from ward features."""
    cleaned = dict(row)

    # Clean population density: remove commas
    if "population_density_2011_persons_sq_km" in cleaned:
        raw_val = str(cleaned["population_density_2011_persons_sq_km"]).replace(",", "").strip()
        try:
            cleaned["population_density_2011_persons_sq_km"] = float(raw_val)
        except ValueError:
            cleaned["population_density_2011_persons_sq_km"] = 0.0

    # Clean percentages
    for pct_col in ["decadal_popn_growth_rate_2011_2001", "decadal_households_growth_rate_2011_2001"]:
        if pct_col in cleaned:
            val = str(cleaned[pct_col]).replace("%", "").strip()
            try:
                cleaned[pct_col] = float(val) / 100.0
            except ValueError:
                cleaned[pct_col] = 0.0

    # Numeric conversion for static counters
    numeric_cols = [
        "population_2011", "area_sq_km", "households_census_2011", "road_length_kms",
        "lakes", "lake_area_sq_m", "parks", "park_area_sq_m", "playgrounds",
        "playground_area_sq_m", "govt_schools", "police_stations", "fire_stations",
        "bus_stops", "bmtc_ttmc", "streetlights", "bus_routes", "centroid_lon", "centroid_lat"
    ]
    for col in numeric_cols:
        if col in cleaned:
            try:
                cleaned[col] = float(cleaned[col]) if cleaned[col] != "" else 0.0
            except ValueError:
                cleaned[col] = 0.0

    return cleaned
