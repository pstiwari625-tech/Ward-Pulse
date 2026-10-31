"""
Ward Pulse: Spatial Anti-Leakage Integrity Test
Guarantees zero overlap between training wards and target inference wards.
"""

from pathlib import Path
import csv

def test_spatial_disjoint_sets():
    grievances_path = Path("grievances_public_wards.csv")
    if not grievances_path.exists():
        grievances_path = Path("data/raw/grievances_public_wards.csv")

    grid_path = Path("prediction_grid_template.csv")
    if not grid_path.exists():
        grid_path = Path("data/raw/prediction_grid_template.csv")

    with open(grievances_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        train_wards = {int(r["ward_id"]) for r in reader}

    with open(grid_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        target_wards = {int(r["ward_id"]) for r in reader}

    intersection = train_wards.intersection(target_wards)
    assert len(intersection) == 0, f"Critical Data Leakage: Overlapping wards found: {intersection}"
    assert len(train_wards) == 156, f"Expected 156 training wards, found {len(train_wards)}"
    assert len(target_wards) == 40, f"Expected 40 target wards, found {len(target_wards)}"

if __name__ == "__main__":
    test_spatial_disjoint_sets()
    print("[SUCCESS] Spatial anti-leakage test passed.")
