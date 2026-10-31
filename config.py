"""
Ward Pulse: Central Configuration
Manages directory paths, schema definitions, model parameters, and CPI weights.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

# Check for raw files in data/raw or root
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
WARD_SUMMARY_FILE = PREDICTIONS_DIR / "ward_summary_latest.csv"
MODEL_METRICS_FILE = MODELS_DIR / "model_metrics.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
CHAMPION_MODEL_FILE = MODELS_DIR / "champion_regressor.joblib"

# Categories
CATEGORIES = [
    "animals_health",
    "electrical",
    "other",
    "roads",
    "solid_waste",
    "trees",
]

# Civic Priority Index (CPI) Multi-Criteria Weights
CPI_WEIGHT_PREDICTED_DEMAND = 0.50
CPI_WEIGHT_POPULATION_DENSITY = 0.25
CPI_WEIGHT_GROWTH_RATE = 0.15
CPI_WEIGHT_INFRA_DEFICIT = 0.10

# Reproducibility
RANDOM_SEED = 42
SPATIAL_CV_SPLITS = 5
