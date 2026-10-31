"""
Ward Pulse: Central Configuration
Manages directory paths, schema definitions, model parameters, and CPI weights.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" if (DATA_DIR / "raw").exists() else BASE_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Raw Data Files
GRIEVANCES_FILE = RAW_DATA_DIR / "grievances_public_wards.csv" if (RAW_DATA_DIR / "grievances_public_wards.csv").exists() else BASE_DIR / "grievances_public_wards.csv"
WARD_FEATURES_FILE = RAW_DATA_DIR / "ward_features.csv" if (RAW_DATA_DIR / "ward_features.csv").exists() else BASE_DIR / "ward_features.csv"
GEOJSON_FILE = RAW_DATA_DIR / "wards.geojson" if (RAW_DATA_DIR / "wards.geojson").exists() else BASE_DIR / "wards.geojson"
PREDICTION_GRID_FILE = RAW_DATA_DIR / "prediction_grid_template.csv" if (RAW_DATA_DIR / "prediction_grid_template.csv").exists() else BASE_DIR / "prediction_grid_template.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv" if (RAW_DATA_DIR / "sample_submission.csv").exists() else BASE_DIR / "sample_submission.csv"

# Prediction Output Contracts
WARD_PREDICTIONS_FILE = PREDICTIONS_DIR / "ward_predictions.csv"
WARD_PREDICTIONS_INTERVALS_FILE = PREDICTIONS_DIR / "ward_predictions_with_intervals.csv"
WARD_SUMMARY_FILE = PREDICTIONS_DIR / "ward_summary_latest.csv"
WARD_SUMMARY_INTERVALS_FILE = PREDICTIONS_DIR / "ward_summary_intervals.csv"
MODEL_METRICS_FILE = MODELS_DIR / "model_metrics.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
CHAMPION_MODEL_FILE = MODELS_DIR / "champion_regressor.joblib"

# Confidence Label Thresholds (relative interval width = (upper - lower) / pred)
CONFIDENCE_HIGH_THRESHOLD = 0.25   # rel_width < 0.25 -> High confidence
CONFIDENCE_MEDIUM_THRESHOLD = 0.40 # rel_width < 0.40 -> Medium confidence, else Low

# 18-Month Target Evaluation Window
TARGET_MONTHS = [f"2024-{m:02d}" for m in range(1, 13)] + [f"2025-{m:02d}" for m in range(1, 7)]

# Canonical Categories (BBMP Municipal Grievances)
CATEGORIES = [
    "animals_health",
    "electrical",
    "other",
    "roads",
    "solid_waste",
    "trees",
]

# Empirical Window Category Shares (2024-01 to 2025-06 Window Ground Truth, %)
WINDOW_CATEGORY_SHARES = {
    "electrical": 34.86,
    "solid_waste": 27.99,
    "roads": 14.34,
    "animals_health": 9.37,
    "other": 7.84,
    "trees": 5.60,
}

# Physical Asset Exposure Map (Strictly using validated columns)
CATEGORY_EXPOSURE_MAP = {
    "electrical": "streetlights",
    "roads": "road_length_kms",
    "solid_waste": "households_census_2011",
    "trees": "parks",
    "animals_health": "population_2011",
    "other": "population_2011",
}

# Spatial Kernel Bandwidths (km) for grid search
KERNEL_BANDWIDTHS = [1.5, 2.0, 2.5, 3.5, 5.0]

# CPI Multi-Criteria Policy Weights (Percentile Rank Form)
CPI_WEIGHT_PREDICTED_DEMAND = 0.40
CPI_WEIGHT_POPULATION_DENSITY = 0.30
CPI_WEIGHT_GROWTH_RATE = 0.15
CPI_WEIGHT_INFRA_DEFICIT = 0.15

# Reproducibility
RANDOM_SEED = 42
SPATIAL_CV_SPLITS = 5

