"""
Ward Pulse: Backend Feature Engineering Layer
Builds spatial lag features, density ratios, and cyclical harmonic calendar encodings.
"""

import math
from typing import Dict, Any

def compute_cyclical_time_features(month_str: str) -> Dict[str, float]:
    """
    Computes sine and cosine harmonic features for a given YYYY-MM string.
    Ensures seamless continuity between December and January.
    """
    month_int = int(month_str.split("-")[1])
    radians = 2.0 * math.pi * (month_int - 1) / 12.0
    return {
        "sin_month": round(math.sin(radians), 4),
        "cos_month": round(math.cos(radians), 4),
        "is_monsoon": 1.0 if month_int in [6, 7, 8, 9] else 0.0,
    }

def compute_density_ratios(features: Dict[str, Any]) -> Dict[str, float]:
    """Computes area and population-normalized infrastructure ratios."""
    area = max(features.get("area_sq_km", 1.0), 0.1)
    pop = max(features.get("population_2011", 1000.0), 100.0)

    return {
        "road_density_km_per_sq_km": features.get("road_length_kms", 0.0) / area,
        "streetlight_density_per_km": features.get("streetlights", 0.0) / max(features.get("road_length_kms", 1.0), 0.1),
        "bus_stop_density_per_sq_km": features.get("bus_stops", 0.0) / area,
        "school_density_per_10k_pop": (features.get("govt_schools", 0.0) / pop) * 10000.0,
    }
