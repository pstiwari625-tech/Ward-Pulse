"""
Ward Pulse: Backend Prioritization Layer
Computes the deterministic Civic Priority Index (CPI).
Strictly separates policy heuristics from trained ML regression outputs.
"""

from typing import List, Dict, Any
import csv
import config

def compute_cpi(
    norm_predicted_demand: float,
    norm_density: float,
    norm_growth: float,
    norm_infra: float
) -> float:
    """
    Computes Civic Priority Index (CPI) in [0.0, 1.0]:
    CPI = 0.50 * Demand + 0.25 * Density + 0.15 * Growth + 0.10 * (1 - Infra)
    """
    cpi = (
        config.CPI_WEIGHT_PREDICTED_DEMAND * norm_predicted_demand +
        config.CPI_WEIGHT_POPULATION_DENSITY * norm_density +
        config.CPI_WEIGHT_GROWTH_RATE * norm_growth +
        config.CPI_WEIGHT_INFRA_DEFICIT * (1.0 - norm_infra)
    )
    return round(min(1.0, max(0.0, cpi)), 4)

def classify_tier(cpi_score: float) -> str:
    """Classifies CPI into operational tiers."""
    if cpi_score >= 0.70:
        return "CRITICAL"
    elif cpi_score >= 0.45:
        return "HIGH"
    elif cpi_score >= 0.25:
        return "MEDIUM"
    return "LOW"
