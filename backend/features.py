"""
Ward Pulse: Backend Feature Engineering Layer
Builds planar UTM coordinates, candidate asset exposures, and log-transformed covariates.
Strictly grounded in columns present in ward_features.csv.
"""

import math
from typing import Dict, Any, List, Tuple
import config

def to_utm43n_approx_km(lat: float, lon: float) -> Tuple[float, float]:
    """
    Projects WGS84 (lat, lon) to local planar coordinates in kilometers
    centered near Bengaluru (13.0°N, 77.6°E).
    """
    dlat_km = (lat - 13.0) * 110.574
    dlon_km = (lon - 77.6) * 111.320 * math.cos(math.radians(13.0))
    return dlon_km, dlat_km

def extract_ward_feature_vector(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts cleaned numeric values and 14 log-transformed covariates
    from a single ward_features record.
    """
    lat = float(row.get("centroid_lat", 13.0))
    lon = float(row.get("centroid_lon", 77.6))
    x_km, y_km = to_utm43n_approx_km(lat, lon)

    pop = float(row.get("population_2011", 40000.0))
    hh = float(row.get("households_census_2011", 10000.0))
    road_len = float(row.get("road_length_kms", 50.0))
    streetlights = float(row.get("streetlights", 2000.0))
    area = float(row.get("area_sq_km", 3.0))
    parks = float(row.get("parks", 5.0))
    park_area = float(row.get("park_area_sq_m", 10000.0))
    playgrounds = float(row.get("playgrounds", 1.0))
    schools = float(row.get("govt_schools", 4.0))
    bus_stops = float(row.get("bus_stops", 20.0))
    bus_routes = float(row.get("bus_routes", 50.0))
    density = float(row.get("population_density_2011_persons_sq_km", 10000.0))
    growth_pop = float(row.get("decadal_popn_growth_rate_2011_2001", 0.5))
    growth_hh = float(row.get("decadal_households_growth_rate_2011_2001", 0.6))

    # 14 log-transformed covariates
    features = [
        math.log(pop + 1.0),
        math.log(hh + 1.0),
        math.log(road_len + 1.0),
        math.log(streetlights + 1.0),
        math.log(area + 1.0),
        math.log(parks + 1.0),
        math.log(park_area + 1.0),
        math.log(playgrounds + 1.0),
        math.log(schools + 1.0),
        math.log(bus_stops + 1.0),
        math.log(bus_routes + 1.0),
        density / 10000.0,
        growth_pop,
        growth_hh
    ]

    return {
        "x": x_km,
        "y": y_km,
        "features": features,
        "raw": {
            "population_2011": pop,
            "households_census_2011": hh,
            "road_length_kms": road_len,
            "streetlights": streetlights,
            "area_sq_km": area,
            "parks": parks,
            "park_area_sq_m": park_area,
            "playgrounds": playgrounds,
            "govt_schools": schools,
            "bus_stops": bus_stops,
            "bus_routes": bus_routes,
            "population_density_2011_persons_sq_km": density,
            "decadal_popn_growth_rate_2011_2001": growth_pop,
            "decadal_households_growth_rate_2011_2001": growth_hh,
            "ward_name": row.get("ward_name", ""),
            "zone": row.get("bbmp_zone_name", "Unknown"),
        }
    }

def get_exposure_value(ward_data: Dict[str, Any], category: str) -> float:
    """Returns the category-specific asset exposure for a ward."""
    col_name = config.CATEGORY_EXPOSURE_MAP.get(category, "population_2011")
    val = ward_data["raw"].get(col_name, 1.0)
    return max(1.0, float(val))
