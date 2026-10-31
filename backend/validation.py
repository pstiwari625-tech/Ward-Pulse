"""
Ward Pulse: Backend Validation Layer
Enforces domain invariants, target row-counts, and prevents data leakage.
"""

from typing import Dict, Any, Set
import config

def validate_datasets(
    feat_ward_ids: Set[int],
    train_ward_ids: Set[int],
    target_ward_ids: Set[int],
    total_grid_rows: int
) -> Dict[str, Any]:
    """
    Validates structural invariants:
    1. Grid row count must be 4320 (40 wards * 18 months * 6 categories)
    2. Overlap between train and target wards must be 0
    3. All target wards must have static features
    """
    results = {
        "grid_rows_valid": total_grid_rows == 4320,
        "target_wards_count": len(target_ward_ids),
        "target_wards_count_valid": len(target_ward_ids) == 40,
        "train_wards_count": len(train_ward_ids),
        "zero_leakage_valid": len(train_ward_ids.intersection(target_ward_ids)) == 0,
        "features_coverage_valid": target_ward_ids.issubset(feat_ward_ids),
    }

    all_passed = all(results.values())
    results["all_passed"] = all_passed
    return results
