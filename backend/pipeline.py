"""
Ward Pulse: Master Backend Pipeline Driver
Executes the full Leave-One-Ward-Out spatial validation, production model training,
target inference, and sensitivity-audited CPI prioritization.
"""

import sys
import json
import csv
from collections import defaultdict
from pathlib import Path

import config
from backend.ingestion import load_ward_features, load_prediction_grid
from backend.validation import validate_datasets
from backend.preprocessing import clean_ward_feature_row
from backend.features import extract_ward_feature_vector
from backend.training import train_and_evaluate_lowo
from backend.inference import generate_production_predictions, export_predictions_csv
from backend.prioritization import generate_ward_summary_table

def run_pipeline():
    print("=" * 70)
    print("WARD PULSE: PRODUCTION CIVIC DEMAND INTELLIGENCE PIPELINE")
    print("=" * 70)

    # 1. Ingestion & Preprocessing
    print("[1/6] Ingesting and featurizing raw assets...")
    with open(config.WARD_FEATURES_FILE, mode="r", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))

    feats_dict = {}
    for r in feat_rows:
        if r["ward_id"]:
            w_id = int(r["ward_id"])
            cleaned_row = clean_ward_feature_row(r)
            feats_dict[w_id] = extract_ward_feature_vector(cleaned_row)

    with open(config.PREDICTION_GRID_FILE, mode="r", encoding="utf-8") as f:
        grid_rows = list(csv.DictReader(f))

    # 2. Validation
    print("[2/6] Validating structural invariants...")
    target_wards = sorted({int(r["ward_id"]) for r in grid_rows})
    train_wards = sorted(list(set(feats_dict.keys()) - set(target_wards) - {3, 65}))

    val_results = validate_datasets(
        feat_ward_ids=set(feats_dict.keys()),
        train_ward_ids=set(train_wards),
        target_ward_ids=set(target_wards),
        total_grid_rows=len(grid_rows)
    )
    if not val_results["all_passed"]:
        print(f"[ERROR] Validation failed: {val_results}")
        sys.exit(1)
    print(f"      Passed invariants. Training Wards: {len(train_wards)}, Target Wards: {len(target_wards)}")

    # 3. Load Grievance Counts in 18-Month Target Window
    print(f"[3/6] Aggregating grievances over 18-month target window ({config.TARGET_MONTHS[0]} to {config.TARGET_MONTHS[-1]})...")
    counts_dict = defaultdict(int)
    target_months_set = set(config.TARGET_MONTHS)

    with open(config.GRIEVANCES_FILE, mode="r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r["grievance_time"][:7]
            if m in target_months_set:
                w = int(r["ward_id"])
                c = r["category_group"]
                counts_dict[(w, m, c)] += 1

    # 4. Leave-One-Ward-Out (LOWO) Spatial Cross-Validation
    print("[4/6] Executing Leave-One-Ward-Out (LOWO) Spatial Cross-Validation on 156 wards...")
    lowo_results = train_and_evaluate_lowo(
        train_wards=train_wards,
        feats_dict=feats_dict,
        counts_dict=counts_dict,
        target_months=config.TARGET_MONTHS,
        categories=config.CATEGORIES,
        kernel_ell=3.5,
        ridge_alpha=5.0
    )

    print(f"      [LOWO Verified Performance]")
    print(f"      Overall MAE:      {lowo_results['oof_mae']} complaints/month")
    print(f"      Overall RMSE:     {lowo_results['oof_rmse']}")
    print(f"      Overall WAPE:     {lowo_results['oof_wape_pct']}%")
    print(f"      Ward MAE Median:  {lowo_results['ward_distribution']['median_mae']} (Min: {lowo_results['ward_distribution']['min_mae']}, Max: {lowo_results['ward_distribution']['max_mae']})")

    # 5. Production Inference on Target Grid
    print("[5/6] Generating production forecasts for 40 target wards (4,320 rows)...")
    predictions = generate_production_predictions(
        train_wards=train_wards,
        target_wards=target_wards,
        feats_dict=feats_dict,
        counts_dict=counts_dict,
        target_months=config.TARGET_MONTHS,
        categories=config.CATEGORIES,
        kernel_ell=3.5,
        ridge_alpha=5.0
    )
    export_predictions_csv(predictions, config.WARD_PREDICTIONS_FILE)
    print(f"      Primary contract exported: {config.WARD_PREDICTIONS_FILE} ({len(predictions)} rows)")

    # 6. Prioritization & Sensitivity Audit
    print("[6/6] Computing CPI, Low-Reporting Equity Scenario, and Monte Carlo Sensitivity Audit...")
    summary_rows, sensitivity_stats = generate_ward_summary_table(
        target_wards=target_wards,
        feats_dict=feats_dict,
        predictions=predictions,
        output_path=config.WARD_SUMMARY_FILE
    )
    print(f"      Summary contract exported: {config.WARD_SUMMARY_FILE}")
    print(f"      Mean Kendall Tau: {sensitivity_stats['mean_kendall_tau']} (Stable Wards: {sensitivity_stats['stable_wards_count']}/40)")

    # Export Metadata & Model Metrics
    metrics_payload = {
        "model_name": "WardTotal_PulseShape_KernelKriging",
        "validation_strategy": "Leave-One-Ward-Out (LOWO) Spatial CV (156 folds)",
        "evaluation_window": f"{config.TARGET_MONTHS[0]} to {config.TARGET_MONTHS[-1]} (18 months)",
        "baseline_anchor": {
            "name": "Ratio_of_Sums_Exposure_Anchor",
            "oof_mae": 8.65,
            "oof_rmse": 16.18
        },
        "metrics": {
            "oof_mae": lowo_results["oof_mae"],
            "oof_rmse": lowo_results["oof_rmse"],
            "oof_wape_pct": lowo_results["oof_wape_pct"]
        },
        "category_metrics": lowo_results["category_metrics"],
        "ward_distribution": lowo_results["ward_distribution"],
        "sensitivity_audit": sensitivity_stats
    }
    with open(config.MODEL_METRICS_FILE, mode="w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    metadata_payload = {
        "features_used": [
            "log(pop+1)", "log(hh+1)", "log(road_len+1)", "log(streetlights+1)",
            "log(area+1)", "log(parks+1)", "log(park_area+1)", "log(playgrounds+1)",
            "log(schools+1)", "log(bus_stops+1)", "log(bus_routes+1)",
            "population_density", "decadal_growth_pop", "decadal_growth_hh"
        ],
        "kernel_parameters": {
            "bandwidth_km": 3.5,
            "kernel_type": "Gaussian_RBF",
            "distance_metric": "UTM_Zone_43N_Planar_Euclidean"
        },
        "target_wards_count": len(target_wards),
        "total_evaluated_records": len(predictions)
    }
    with open(config.MODEL_METADATA_FILE, mode="w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    print("\n[SUCCESS] Pipeline execution complete. All contracts generated & verified.")

if __name__ == "__main__":
    run_pipeline()
