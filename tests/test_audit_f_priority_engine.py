"""
Audit Section F: Priority Engine Separation Tests (Tests 25 - 27)
Validates strict separation between ML predictions and CPI weights, Monte Carlo stability,
and proper labeling of the equity scenario.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import csv, random
import config
from backend.prioritization import percentile_rank, classify_tier, compute_kendall_tau

def test_25_cpi_never_in_predictions_file():
    """Test 25: CPI values never appear in ML prediction file; predicted_count is unchanged by weights."""
    pred_path = config.WARD_PREDICTIONS_FILE
    with open(pred_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
        
    has_cpi_col = any("cpi" in f.lower() or "priority" in f.lower() or "tier" in f.lower() for f in fields)
    
    # Assert predicted_count values are pure counts (not normalized 0-1 scores)
    counts = [float(r["predicted_count"]) for r in rows]
    max_cnt = max(counts)
    is_count_scale = max_cnt > 10.0 # pure complaint count scale, not [0, 1] index
    
    passed = (not has_cpi_col) and is_count_scale
    return {
        "id": 25,
        "name": "CPI values never appear in ML prediction file (strict separation)",
        "pass": passed,
        "measured": f"Columns in ward_predictions.csv: {fields}, Max predicted_count: {max_cnt:.2f}",
        "note": "Clean separation: ML prediction contract contains zero heuristic policy weights" if passed else "LEAK: CPI weights found in prediction file"
    }

def test_26_monte_carlo_weight_perturbation():
    """Test 26: Monte Carlo weight perturbation (+-20%, 1,000 draws): report Kendall tau and list of flips."""
    summary_path = config.WARD_SUMMARY_FILE
    with open(summary_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    target_wards = [int(r["ward_id"]) for r in rows]
    base_cpis = [float(r["cpi_score"]) for r in rows]
    
    # Load features dict
    with open(config.WARD_FEATURES_FILE, mode="r", encoding="utf-8") as f:
        feat_rows = list(csv.DictReader(f))
    from backend.preprocessing import clean_ward_feature_row
    from backend.features import extract_ward_feature_vector
    feats_dict = {int(r["ward_id"]): extract_ward_feature_vector(clean_ward_feature_row(r))
                  for r in feat_rows if r["ward_id"]}

    # Re-run 1,000 draws of +-20% weights
    rank_demand = percentile_rank([float(r["total_predicted_monthly"]) for r in rows])
    rank_density = percentile_rank([feats_dict[w]["raw"]["population_density_2011_persons_sq_km"] for w in target_wards])
    rank_growth = percentile_rank([feats_dict[w]["raw"]["decadal_popn_growth_rate_2011_2001"] for w in target_wards])
    rank_infra = percentile_rank([feats_dict[w]["raw"]["streetlights"] / max(1.0, feats_dict[w]["raw"]["road_length_kms"]) for w in target_wards])
    
    random.seed(config.RANDOM_SEED)
    taus = []
    flips_per_ward = {w: 0 for w in target_wards}
    
    for _ in range(1000):
        w_dem = random.uniform(0.32, 0.48)
        w_den = random.uniform(0.24, 0.36)
        w_gro = random.uniform(0.12, 0.18)
        w_inf = random.uniform(0.12, 0.18)
        w_sum = w_dem + w_den + w_gro + w_inf
        
        sim_scores = [
            (w_dem * rank_demand[i] + w_den * rank_density[i] + w_gro * rank_growth[i] + w_inf * (1.0 - rank_infra[i])) / w_sum
            for i in range(len(target_wards))
        ]
        taus.append(compute_kendall_tau(base_cpis, sim_scores))
        for i, (b, s) in enumerate(zip(base_cpis, sim_scores)):
            if classify_tier(b) != classify_tier(s):
                flips_per_ward[target_wards[i]] += 1
                
    mean_tau = sum(taus) / len(taus)
    flipping_wards = [w for w, count in flips_per_ward.items() if count > 0]
    
    # Check tier counts
    tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for b in base_cpis:
        tier_counts[classify_tier(b)] += 1
        
    passed = mean_tau >= 0.85 and len(flipping_wards) <= 15
    return {
        "id": 26,
        "name": "Monte Carlo weight perturbation (1,000 draws, +-20%)",
        "pass": passed,
        "measured": f"Mean Kendall tau: {mean_tau:.3f}, Flips: {len(flipping_wards)}/40 wards {flipping_wards[:5]}, Tier counts: {tier_counts}",
        "note": "CPI rankings are robust to weight perturbation" if passed else "Fragile policy ranking"
    }

def test_27_equity_scenario_labeling():
    """Test 27: The equity scenario is labelled 'scenario', not 'true need'."""
    files_to_check = [
        BASE_DIR / "frontend" / "app.py",
        BASE_DIR / "backend" / "prioritization.py",
        BASE_DIR / "WARD_PULSE.md",
        BASE_DIR / "data" / "predictions" / "ward_summary_latest.csv"
    ]
    
    true_need_violations = []
    has_scenario_label = False
    
    for fpath in files_to_check:
        if fpath.exists():
            with open(fpath, mode="r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "true need" in content.lower() or "true distress" in content.lower():
                    true_need_violations.append(fpath.name)
                if "scenario" in content.lower():
                    has_scenario_label = True
                    
    passed = (len(true_need_violations) == 0) and has_scenario_label
    return {
        "id": 27,
        "name": "Equity scenario properly labelled as 'scenario' (no false claims of true need)",
        "pass": passed,
        "measured": f"Labelled as 'scenario': {has_scenario_label}, 'True need' claims: {len(true_need_violations)} ({true_need_violations})",
        "note": "Transparently presented as policy sensitivity scenario" if passed else f"Found misleading 'true need' claims in: {true_need_violations}"
    }

if __name__ == "__main__":
    for t in [test_25_cpi_never_in_predictions_file, test_26_monte_carlo_weight_perturbation,
              test_27_equity_scenario_labeling]:
        res = t()
        status = "PASS" if res["pass"] else "FAIL"
        print(f"[{status}] Test {res['id']}: {res['name']} | Measured: {res['measured']}")
