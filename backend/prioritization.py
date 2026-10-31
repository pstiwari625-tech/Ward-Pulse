"""
Ward Pulse: Backend Prioritization Layer
Computes the percentile-rank Civic Priority Index (CPI), Low-Reporting Equity Scenario,
and Monte Carlo rank sensitivity analysis.
"""

from typing import List, Dict, Any, Tuple
import random
import csv
import config

def percentile_rank(values: List[float]) -> List[float]:
    """Returns normalized percentile rank [0.0, 1.0] for a list of values."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    for rank_idx, (orig_idx, _) in enumerate(sorted_pairs):
        ranks[orig_idx] = rank_idx / (n - 1)
    return ranks

def classify_tier(cpi_score: float) -> str:
    """Classifies CPI into operational tiers."""
    if cpi_score >= 0.70:
        return "CRITICAL"
    elif cpi_score >= 0.45:
        return "HIGH"
    elif cpi_score >= 0.25:
        return "MEDIUM"
    return "LOW"

def compute_kendall_tau(rank_a: List[float], rank_b: List[float]) -> float:
    """Computes Kendall's tau rank correlation between two score vectors."""
    n = len(rank_a)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = 1 if rank_a[i] > rank_a[j] else (-1 if rank_a[i] < rank_a[j] else 0)
            sign_b = 1 if rank_b[i] > rank_b[j] else (-1 if rank_b[i] < rank_b[j] else 0)
            prod = sign_a * sign_b
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    return round((concordant - discordant) / max(1, total_pairs), 3)

def generate_ward_summary_table(
    target_wards: List[int],
    feats_dict: Dict[int, Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    output_path = config.WARD_SUMMARY_FILE
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Assembles ward_summary_latest.csv with both standard CPI and Equity Sensitivity Scenario.
    """
    # 1. Aggregate predictions per ward
    ward_cat_totals = {w: {c: 0.0 for c in config.CATEGORIES} for w in target_wards}
    for p in predictions:
        w = p["ward_id"]
        c = p["category_group"]
        ward_cat_totals[w][c] += p["predicted_count"] / 18.0 # monthly average

    ward_monthly_totals = [sum(ward_cat_totals[w].values()) for w in target_wards]
    ward_densities = [feats_dict[w]["raw"]["population_density_2011_persons_sq_km"] for w in target_wards]
    ward_growths = [feats_dict[w]["raw"]["decadal_popn_growth_rate_2011_2001"] for w in target_wards]
    ward_infras = [feats_dict[w]["raw"]["streetlights"] / max(1.0, feats_dict[w]["raw"]["road_length_kms"]) for w in target_wards]

    # Percentile rankings
    rank_demand = percentile_rank(ward_monthly_totals)
    rank_density = percentile_rank(ward_densities)
    rank_growth = percentile_rank(ward_growths)
    rank_infra = percentile_rank(ward_infras)

    # 2. Compute Base CPI
    base_cpis = []
    for i in range(len(target_wards)):
        score = (
            config.CPI_WEIGHT_PREDICTED_DEMAND * rank_demand[i] +
            config.CPI_WEIGHT_POPULATION_DENSITY * rank_density[i] +
            config.CPI_WEIGHT_GROWTH_RATE * rank_growth[i] +
            config.CPI_WEIGHT_INFRA_DEFICIT * (1.0 - rank_infra[i])
        )
        base_cpis.append(round(score, 3))

    # 3. Compute Equity Scenario (Low-reporting penalty adjustment)
    rank_bus = percentile_rank([feats_dict[w]["raw"]["bus_stops"] for w in target_wards])
    equity_cpis = []
    for i in range(len(target_wards)):
        reporting_proxy = 0.5 * rank_bus[i] + 0.5 * (1.0 - rank_density[i])
        equity_demand = rank_demand[i] / (reporting_proxy + 0.2)
        eq_score = (
            0.40 * min(1.0, equity_demand / 2.0) +
            0.30 * rank_density[i] +
            0.15 * rank_growth[i] +
            0.15 * (1.0 - rank_infra[i])
        )
        equity_cpis.append(round(eq_score, 3))

    # 4. Monte Carlo Sensitivity Audit (100 perturbations of weights)
    random.seed(config.RANDOM_SEED)
    taus = []
    flips_count = [0] * len(target_wards)

    for _ in range(100):
        w_dem = random.uniform(0.32, 0.48)
        w_den = random.uniform(0.24, 0.36)
        w_gro = random.uniform(0.12, 0.18)
        w_inf = random.uniform(0.12, 0.18)
        w_sum = w_dem + w_den + w_gro + w_inf

        sim_cpis = [
            (w_dem * rank_demand[i] + w_den * rank_density[i] + w_gro * rank_growth[i] + w_inf * (1.0 - rank_infra[i])) / w_sum
            for i in range(len(target_wards))
        ]
        taus.append(compute_kendall_tau(base_cpis, sim_cpis))

        for i, (b, s) in enumerate(zip(base_cpis, sim_cpis)):
            if classify_tier(b) != classify_tier(s):
                flips_count[i] += 1

    mean_tau = round(sum(taus) / len(taus), 3)

    # 5. Build Output Rows
    summary_rows = []
    for i, w in enumerate(target_wards):
        tier = classify_tier(base_cpis[i])
        eq_tier = classify_tier(equity_cpis[i])
        summary_rows.append({
            "ward_id": w,
            "ward_name": feats_dict[w]["raw"]["ward_name"],
            "zone": feats_dict[w]["raw"]["zone"],
            "total_predicted_monthly": round(ward_monthly_totals[i], 2),
            "cpi_score": base_cpis[i],
            "priority_tier": tier,
            "equity_cpi_score": equity_cpis[i],
            "equity_tier": eq_tier,
            "rank_flip_prob": round(flips_count[i] / 100.0, 2),
            **{f"pred_{c}": round(ward_cat_totals[w][c], 2) for c in config.CATEGORIES}
        })

    # Export to CSV
    if summary_rows:
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    sensitivity_stats = {
        "mean_kendall_tau": mean_tau,
        "stable_wards_count": sum(1 for f in flips_count if f == 0),
        "borderline_wards_count": sum(1 for f in flips_count if f > 0)
    }

    return summary_rows, sensitivity_stats
