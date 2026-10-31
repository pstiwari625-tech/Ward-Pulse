"""
Ward Pulse: Master 198-Ward Dataset Generator
Builds a unified, lightweight dataset of all 198 municipal wards in Bangalore:
 - 40 unobserved target wards with production ML forecasts & 90% conformal intervals
 - 158 training wards with 18-month historical ground-truth grievance averages
Saves to data/predictions/all_wards_master.csv and all_wards_master.json for fast UI & API lookups.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def build_master_dataset():
    # 1. Load ward features
    features_file = PROJECT_ROOT / "ward_features.csv"
    with open(features_file, mode="r", encoding="utf-8") as f:
        feat_rows = {int(r["ward_id"]): r for r in csv.DictReader(f) if r["ward_id"]}

    # 2. Load target predictions & intervals
    target_summary_file = PROJECT_ROOT / "data" / "predictions" / "ward_summary_latest.csv"
    with open(target_summary_file, mode="r", encoding="utf-8") as f:
        target_summary = {int(r["ward_id"]): r for r in csv.DictReader(f)}

    target_intervals_file = PROJECT_ROOT / "data" / "predictions" / "ward_summary_intervals.csv"
    with open(target_intervals_file, mode="r", encoding="utf-8") as f:
        target_intervals = {int(r["ward_id"]): r for r in csv.DictReader(f)}

    target_ward_ids = set(target_summary.keys())

    # 3. Load historical counts for training wards
    target_months = set(config.TARGET_MONTHS)
    hist_totals = defaultdict(float)
    hist_cats = defaultdict(lambda: defaultdict(float))

    grievance_file = PROJECT_ROOT / "grievances_public_wards.csv"
    if grievance_file.exists():
        with open(grievance_file, mode="r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = r["grievance_time"][:7]
                if m in target_months:
                    try:
                        w = int(r["ward_id"])
                        c = r["category_group"]
                        hist_totals[w] += 1.0
                        hist_cats[w][c] += 1.0
                    except ValueError:
                        pass

    # 4. Construct unified 198 rows
    master_records = []
    
    # Precompute training demand distribution for fair percentile tiering
    training_monthly_demands = [hist_totals[w] / 18.0 for w in feat_rows if w not in target_ward_ids]
    max_train_demand = max(training_monthly_demands) if training_monthly_demands else 1.0
    p75_train = sorted(training_monthly_demands)[int(len(training_monthly_demands) * 0.75)] if training_monthly_demands else 0
    p40_train = sorted(training_monthly_demands)[int(len(training_monthly_demands) * 0.40)] if training_monthly_demands else 0

    for wid in sorted(feat_rows.keys()):
        feat = feat_rows[wid]
        wname = feat.get("ward_name", f"Ward {wid}")
        zone = feat.get("bbmp_zone_name", "Unknown")
        pop = float(feat.get("population_2011", 0) or 0)
        area = float(feat.get("area_sq_km", 0) or 0)
        roads = float(feat.get("road_length_kms", 0) or 0)
        streetlights = float(feat.get("streetlights", 0) or 0)

        if wid in target_ward_ids:
            # TARGET WARD
            ts = target_summary[wid]
            ti = target_intervals.get(wid, {})

            monthly_demand = round(float(ts["total_predicted_monthly"]), 1)
            total_18m = round(monthly_demand * 18.0, 1)
            low_90 = round(float(ti.get("total_lower_90", monthly_demand * 18 * 0.5)) / 18.0, 1)
            high_90 = round(float(ti.get("total_upper_90", monthly_demand * 18 * 1.5)) / 18.0, 1)
            conf = ti.get("confidence_label", "Moderate")
            ptier = ts.get("priority_tier", "MEDIUM")
            cpi = round(float(ts.get("cpi_score", 0.5)), 3)
            top_c = ti.get("top_category", "solid_waste")

            cat_dict = {
                "electrical": round(float(ts.get("pred_electrical", 0)), 1),
                "solid_waste": round(float(ts.get("pred_solid_waste", 0)), 1),
                "roads": round(float(ts.get("pred_roads", 0)), 1),
                "animals_health": round(float(ts.get("pred_animals_health", 0)), 1),
                "trees": round(float(ts.get("pred_trees", 0)), 1),
                "other": round(float(ts.get("pred_other", 0)), 1),
            }

            rec = {
                "ward_id": wid,
                "ward_name": wname,
                "zone": zone,
                "ward_type": "Target Ward (ML Forecast)",
                "is_target": True,
                "population": int(pop),
                "area_sq_km": round(area, 2),
                "road_length_kms": round(roads, 1),
                "streetlights": int(streetlights),
                "total_monthly_demand": monthly_demand,
                "total_18m_demand": total_18m,
                "lower_90": low_90,
                "upper_90": high_90,
                "interval_90_str": f"[{low_90}, {high_90}]",
                "confidence_tier": conf,
                "priority_tier": ptier,
                "cpi_score": cpi,
                "top_category": top_c,
                "rank_flip_stability": round(float(ts.get("rank_flip_prob", 0.95)), 2),
                **cat_dict
            }
        else:
            # TRAINING WARD
            total_18m = round(hist_totals[wid], 1)
            monthly_demand = round(total_18m / 18.0, 1)
            
            # Derive tier for baseline comparison
            if monthly_demand >= p75_train:
                ptier = "HIGH"
            elif monthly_demand >= p40_train:
                ptier = "MEDIUM"
            else:
                ptier = "LOW"

            cpi = round(min(1.0, monthly_demand / max(1.0, max_train_demand)), 3)

            cat_dict = {}
            for c in ["electrical", "solid_waste", "roads", "animals_health", "trees", "other"]:
                cat_dict[c] = round(hist_cats[wid][c] / 18.0, 1)

            top_c = max(cat_dict, key=cat_dict.get) if any(cat_dict.values()) else "solid_waste"

            rec = {
                "ward_id": wid,
                "ward_name": wname,
                "zone": zone,
                "ward_type": "Historical Observed (Ground Truth)",
                "is_target": False,
                "population": int(pop),
                "area_sq_km": round(area, 2),
                "road_length_kms": round(roads, 1),
                "streetlights": int(streetlights),
                "total_monthly_demand": monthly_demand,
                "total_18m_demand": total_18m,
                "lower_90": monthly_demand,
                "upper_90": monthly_demand,
                "interval_90_str": "Observed Ground Truth",
                "confidence_tier": "Observed Baseline",
                "priority_tier": ptier,
                "cpi_score": cpi,
                "top_category": top_c,
                "rank_flip_stability": 1.00,
                **cat_dict
            }
        master_records.append(rec)

    # 5. Write CSV
    out_csv = PROJECT_ROOT / "data" / "predictions" / "all_wards_master.csv"
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = list(master_records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(master_records)

    # 6. Write JSON
    out_json = PROJECT_ROOT / "data" / "predictions" / "all_wards_master.json"
    with open(out_json, mode="w", encoding="utf-8") as f:
        json.dump(master_records, f, indent=2)

    print(f"[SUCCESS] Generated master dataset with {len(master_records)} wards:")
    print(f" - CSV: {out_csv}")
    print(f" - JSON: {out_json}")

if __name__ == "__main__":
    build_master_dataset()
