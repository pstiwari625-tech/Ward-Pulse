"""
Master Test Suite Runner (tests/run_full_audit.py)
Executes all 35 tests across Sections A through H.
Computes per-section subtotal counts that add up to 35,
evaluates PASS/FAIL status with measured values, and generates the final audit table.
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import test suites
from tests.test_audit_a_contracts import (
    test_1_ward_predictions_shape_and_columns,
    test_2_key_set_equals_template,
    test_3_no_nulls_nan_inf_non_negative,
    test_4_cardinalities_40_wards_18_months_6_categories,
    test_5_inductive_setting_zero_rows_in_grievances,
    test_6_water_must_not_appear_anywhere
)

from tests.test_audit_b_leakage import (
    test_7_label_perturbation_invariance,
    test_8_neighbour_leak_in_lowo,
    test_9_reproduce_lowo_benchmarks,
    test_10_nested_cv_hyperparameter_check,
    test_11_ward_bootstrap_95ci
)

from tests.test_audit_c_model_quality import (
    test_12_beats_baseline_slices,
    test_13_ward_totals_correlation,
    test_14_extrapolation_safety,
    test_15_seasonality_city_curve_correlation,
    test_16_category_mix_drift
)

from tests.test_audit_d_neighbours_map import (
    test_17_spatial_ablation,
    test_18_bandwidth_sweep,
    test_19_nearest_train_distance_distribution,
    test_20_map_features_grounded
)

from tests.test_audit_e_confidence import (
    test_21_confidence_per_ward,
    test_22_interval_validity,
    test_23_empirical_coverage_under_lowo,
    test_24_interval_width_scales_with_volume
)

from tests.test_audit_f_priority_engine import (
    test_25_cpi_never_in_predictions_file,
    test_26_monte_carlo_weight_perturbation,
    test_27_equity_scenario_labeling
)

from tests.test_audit_g_dashboard import (
    test_28_offline_and_startup,
    test_29_map_tooltips_content,
    test_30_category_selector_changes_map,
    test_31_month_selector_and_reconciliation,
    test_32_governance_tab_matches_test_9,
    test_33_download_predictions_button
)

from tests.test_audit_h_engineering import (
    test_34_pipeline_reproducibility,
    test_35_edge_cases_and_null_tolerance
)

SECTION_MAP = {
    1: "A. Data Contract (R1, R4)",
    2: "A. Data Contract (R1, R4)",
    3: "A. Data Contract (R1, R4)",
    4: "A. Data Contract (R1, R4)",
    5: "A. Data Contract (R1, R4)",
    6: "A. Data Contract (R1, R4)",
    7: "B. Leakage & Validation (R3, R4)",
    8: "B. Leakage & Validation (R3, R4)",
    9: "B. Leakage & Validation (R3, R4)",
    10: "B. Leakage & Validation (R3, R4)",
    11: "B. Leakage & Validation (R3, R4)",
    12: "C. Model Quality Slices (R1, R4)",
    13: "C. Model Quality Slices (R1, R4)",
    14: "C. Model Quality Slices (R1, R4)",
    15: "C. Model Quality Slices (R1, R4)",
    16: "C. Model Quality Slices (R1, R4)",
    17: "D. Neighbours & Map Features (R3)",
    18: "D. Neighbours & Map Features (R3)",
    19: "D. Neighbours & Map Features (R3)",
    20: "D. Neighbours & Map Features (R3)",
    21: "E. Confidence & Uncertainty (R2)",
    22: "E. Confidence & Uncertainty (R2)",
    23: "E. Confidence & Uncertainty (R2)",
    24: "E. Confidence & Uncertainty (R2)",
    25: "F. Priority Engine (R4, R5)",
    26: "F. Priority Engine (R4, R5)",
    27: "F. Priority Engine (R4, R5)",
    28: "G. Dashboard & Presentation (R5)",
    29: "G. Dashboard & Presentation (R5)",
    30: "G. Dashboard & Presentation (R5)",
    31: "G. Dashboard & Presentation (R5)",
    32: "G. Dashboard & Presentation (R5)",
    33: "G. Dashboard & Presentation (R5)",
    34: "H. Engineering & Edge Cases (R4)",
    35: "H. Engineering & Edge Cases (R4)"
}

REQUIREMENT_MAP = {
    1: "R1, R4", 2: "R1, R4", 3: "R1, R4", 4: "R1, R4", 5: "R4", 6: "R1",
    7: "R3, R4", 8: "R3, R4", 9: "R4", 10: "R3, R4", 11: "R4",
    12: "R1, R4", 13: "R4", 14: "R4", 15: "R4", 16: "R1, R4",
    17: "R3", 18: "R3", 19: "R3", 20: "R3",
    21: "R2", 22: "R2", 23: "R2", 24: "R2",
    25: "R4, R5", 26: "R5", 27: "R5",
    28: "R5", 29: "R5", 30: "R1, R5", 31: "R1, R5", 32: "R4, R5", 33: "R5",
    34: "R4", 35: "R4"
}

TEST_FUNCTIONS = [
    test_1_ward_predictions_shape_and_columns,
    test_2_key_set_equals_template,
    test_3_no_nulls_nan_inf_non_negative,
    test_4_cardinalities_40_wards_18_months_6_categories,
    test_5_inductive_setting_zero_rows_in_grievances,
    test_6_water_must_not_appear_anywhere,
    test_7_label_perturbation_invariance,
    test_8_neighbour_leak_in_lowo,
    test_9_reproduce_lowo_benchmarks,
    test_10_nested_cv_hyperparameter_check,
    test_11_ward_bootstrap_95ci,
    test_12_beats_baseline_slices,
    test_13_ward_totals_correlation,
    test_14_extrapolation_safety,
    test_15_seasonality_city_curve_correlation,
    test_16_category_mix_drift,
    test_17_spatial_ablation,
    test_18_bandwidth_sweep,
    test_19_nearest_train_distance_distribution,
    test_20_map_features_grounded,
    test_21_confidence_per_ward,
    test_22_interval_validity,
    test_23_empirical_coverage_under_lowo,
    test_24_interval_width_scales_with_volume,
    test_25_cpi_never_in_predictions_file,
    test_26_monte_carlo_weight_perturbation,
    test_27_equity_scenario_labeling,
    test_28_offline_and_startup,
    test_29_map_tooltips_content,
    test_30_category_selector_changes_map,
    test_31_month_selector_and_reconciliation,
    test_32_governance_tab_matches_test_9,
    test_33_download_predictions_button,
    test_34_pipeline_reproducibility,
    test_35_edge_cases_and_null_tolerance
]

def run_all():
    print(f"=== Starting Full Audit Execution ({len(TEST_FUNCTIONS)} Tests) ===")
    start_time = time.time()
    results = []

    section_stats = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0})

    for fn in TEST_FUNCTIONS:
        t0 = time.time()
        res = fn()
        elapsed = time.time() - t0
        t_id = res["id"]
        sec = SECTION_MAP[t_id]
        res["section"] = sec
        res["requirement"] = REQUIREMENT_MAP.get(t_id, "R4")
        status = "PASS" if res["pass"] else "FAIL"

        section_stats[sec]["total"] += 1
        if res["pass"]:
            section_stats[sec]["pass"] += 1
        else:
            section_stats[sec]["fail"] += 1

        print(f"[{status}] Test {t_id:02d} ({res['requirement']}): {res['name']} ({elapsed:.2f}s)")
        results.append(res)

    total_elapsed = time.time() - start_time
    print(f"\n=== Audit Complete in {total_elapsed:.1f}s ===")

    pass_count = sum(1 for r in results if r["pass"])
    fail_count = sum(1 for r in results if not r["pass"])
    print(f"Total Tests: {len(results)} | Passed: {pass_count} | Failed: {fail_count}\n")

    # Section Summary Table
    print("### Section Summary Breakdown")
    print("| Section | Total | Passed | Failed | Compliance |")
    print("|---|:---:|:---:|:---:|:---:|")
    for sec in sorted(section_stats.keys()):
        st = section_stats[sec]
        pct = (st["pass"] / st["total"]) * 100.0
        print(f"| {sec} | {st['total']} | {st['pass']} | {st['fail']} | {pct:.1f}% |")
    total_pct = (pass_count / len(results)) * 100.0
    print(f"| **TOTAL** | **{len(results)}** | **{pass_count}** | **{fail_count}** | **{total_pct:.1f}%** |\n")

    # Master Table
    print("### Master PASS/FAIL Audit Table")
    print("| Test ID | Requirement | PASS/FAIL | Measured Value | Note |")
    print("|---|---|---|---|---|")
    for r in results:
        status = "**PASS**" if r["pass"] else "**FAIL**"
        print(f"| Test {r['id']} | {r['requirement']} | {status} | {r['measured']} | {r['note']} |")

    return results

if __name__ == "__main__":
    run_all()
