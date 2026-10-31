"""
Ward Pulse: Backend Inference Layer
Generates non-negative grievance forecasts for the 40 unobserved target wards.
"""

from typing import List, Dict, Any
import csv
import config

def export_predictions_csv(predictions: List[Dict[str, Any]], output_path = config.WARD_PREDICTIONS_FILE):
    """
    Serializes prediction records to match the sample_submission contract:
    ward_id,month,category_group,predicted_count
    """
    fieldnames = ["ward_id", "month", "category_group", "predicted_count"]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            writer.writerow({
                "ward_id": int(p["ward_id"]),
                "month": str(p["month"]),
                "category_group": str(p["category_group"]),
                "predicted_count": round(max(0.0, float(p["predicted_count"])), 2)
            })
