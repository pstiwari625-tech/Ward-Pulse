"""
Ward Pulse: Backend ML Training Layer
Implements 5-Fold Spatial GroupKFold Cross-Validation for inductive count models.
"""

from typing import List, Dict, Any, Tuple
import config

def get_spatial_group_kfold_splits(ward_ids: List[int], n_splits: int = config.SPATIAL_CV_SPLITS) -> List[Tuple[List[int], List[int]]]:
    """
    Partitions unique ward_ids into n_splits spatial disjoint sets.
    Guarantees no ward in training fold is present in validation fold.
    """
    sorted_wards = sorted(list(set(ward_ids)))
    folds = []
    chunk_size = len(sorted_wards) // n_splits
    remainder = len(sorted_wards) % n_splits

    start = 0
    for i in range(n_splits):
        end = start + chunk_size + (1 if i < remainder else 0)
        val_wards = sorted_wards[start:end]
        train_wards = [w for w in sorted_wards if w not in val_wards]
        folds.append((train_wards, val_wards))
        start = end

    return folds
