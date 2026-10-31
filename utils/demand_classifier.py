"""
Ward Pulse - Civic Demand Stratification & Threshold Classifier
Provides standardized, documented, and deterministic demand level categorization
across municipal wards. Strictly forbids random or synthetic category assignment.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


# Standardized 4-tier demand levels and accessible color scheme
DEMAND_TIERS = ["Low", "Medium", "High", "Very High"]

DEMAND_COLOR_MAP: Dict[str, str] = {
    "Low": "#10B981",        # Emerald Green
    "Medium": "#F59E0B",     # Amber Yellow
    "High": "#F97316",       # Vibrant Orange
    "Very High": "#EF4444",  # Crimson Red
    "Critical": "#EF4444",   # Backward-compatible alias
    "Standard": "#64748B"    # Muted Slate Baseline
}


def get_quantile_cutoffs(series: pd.Series) -> Dict[str, float]:
    """
    Computes documented 4-tier quartile cutoffs (25th, 50th, 75th percentiles).
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.empty or len(clean_series) < 4:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}

    return {
        "p25": float(clean_series.quantile(0.25)),
        "p50": float(clean_series.quantile(0.50)),
        "p75": float(clean_series.quantile(0.75)),
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "mean": float(clean_series.mean())
    }


def classify_demand(
    val: float,
    cutoffs: Dict[str, float]
) -> str:
    """
    Deterministically maps a scalar complaint value to a documented demand tier.
    """
    if pd.isna(val):
        return "Low"

    p25 = cutoffs.get("p25", 0.0)
    p50 = cutoffs.get("p50", 0.0)
    p75 = cutoffs.get("p75", 0.0)

    if val >= p75:
        return "Very High"
    elif val >= p50:
        return "High"
    elif val >= p25:
        return "Medium"
    else:
        return "Low"


def classify_dataframe_demand(
    df: pd.DataFrame,
    complaint_col: str = "predicted_complaints",
    target_col: str = "demand_level"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Adds a normalized, deterministic demand classification column to the DataFrame.
    Guarantees reproducible cutoffs and documented threshold metrics.
    """
    df_out = df.copy()

    if complaint_col not in df_out.columns:
        df_out[target_col] = "Standard"
        return df_out, {
            "status": "UNAVAILABLE",
            "message": f"Column '{complaint_col}' not found in DataFrame.",
            "cutoffs": {},
            "tier_counts": {}
        }

    numeric_series = pd.to_numeric(df_out[complaint_col], errors="coerce").fillna(0.0)
    cutoffs = get_quantile_cutoffs(numeric_series)

    df_out[target_col] = numeric_series.apply(lambda v: classify_demand(v, cutoffs))

    tier_counts = df_out[target_col].value_counts().to_dict()

    metadata = {
        "status": "SUCCESS",
        "method": "Quantile (P25, P50, P75)",
        "cutoffs": cutoffs,
        "tier_counts": tier_counts,
        "total_wards": len(df_out),
        "high_priority_count": int(tier_counts.get("High", 0) + tier_counts.get("Very High", 0))
    }

    return df_out, metadata
