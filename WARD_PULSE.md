# WARD PULSE: System Architecture & Technical Implementation Blueprint
**Civic-Demand Intelligence & Ward-Level Prediction Platform**  
*Document Version:* 2.0.0-PROD  
*Target Environment:* Municipal Decision Support / Local Execution  
*Evaluation Protocol:* Leave-One-Ward-Out (LOWO) Spatial Cross-Validation (156 Folds)

---

## 1. Executive Summary & Problem Formulation

**Ward Pulse** is an offline-capable, spatio-temporal municipal demand intelligence system designed to help city administrators proactively anticipate civic grievances across urban wards.

### The True Machine Learning Formulation
1. **Spatial Imputation with Shared Contemporaneous Time:**  
   The dataset contains 198 total wards. Historical complaints are available for **156 training wards** through June 2025. The test set comprises **40 completely unobserved target wards** over an 18-month window (**January 2024 to June 2025**). Because the 156 training wards are observed during the *exact same 18 months*, city-wide macro dynamics (monsoons, portal updates, reporting surges) are known contemporaneously.
2. **Effective Sample Size:**  
   Because static ward attributes are cross-sectional ($N = 156$), deep or unconstrained gradient boosted trees overfit rapidly. The architecture solves this by modeling **18-month ward totals** using regularized log-linear models + in-fold Gaussian spatial residual kriging, then allocating volume temporally via the observed category pulse shape.
3. **Decoupled Governance:**  
   Supervised ML forecasts ($\widehat{y}_{w, t, c}$) are strictly decoupled from policy decision heuristics (the Civic Priority Index, CPI).

---

## 2. Empirical Data Audit & Target Geometry

### 2.1 Spatial Interleaving Audit (Projected UTM Zone 43N)
Coordinate distances between the 40 target wards and the 156 training wards were computed using planar UTM 43N projection ($13^\circ\text{N}, 77.6^\circ\text{E}$):
* **Minimum Distance:** $0.352\text{ km}$ ($352$ meters)
* **25th Percentile:** $0.723\text{ km}$
* **Median Distance:** $1.029\text{ km}$
* **75th Percentile:** $1.844\text{ km}$
* **Maximum Distance:** $2.878\text{ km}$

The 40 target wards are **stratified across all 8 zones** (South: $27.5\%$ target vs. $26.9\%$ train; East: $22.5\%$ vs. $22.4\%$; West: $17.5\%$ vs. $17.3\%$, etc.). Target wards are interleaved within $\sim 1\text{ km}$ of training wards, making **spatial residual kriging** exceptionally effective.

### 2.2 Structural Break & Covariate Shift
1. **The May 2024 Surge:**  
   City-wide grievance volume jumps from $\sim 7.5\text{k}-8.6\text{k}/\text{month}$ (Jan–Apr 2024) to $\mathbf{12.4\text{k}-19.3\text{k}/\text{month}}$ (May 2024 onwards). Ward-specific surge variance ($sd = 0.25$ on log-ratio) has low spatial correlation ($r \approx 0.13$), establishing a hard empirical noise ceiling.
2. **Covariate Shift:**  
   Target wards are systematically $\sim 30\%$ larger than training wards (mean road length $84\text{ km}$ vs. $64\text{ km}$; streetlights $2,395$ vs. $2,050$). Tree-based models flatline on upper-tail feature extrapolation; regularized log-linear Poisson formulations extrapolate proportionally without saturation.

---

## 3. Verified Performance & Benchmark Comparison

All models are evaluated under **Leave-One-Ward-Out (LOWO) Spatial Cross-Validation** (156 folds) strictly on the **18-month target window (`2024-01` to `2025-06`)**:

```
========================================================================================
MODEL ARCHITECTURE COMPARISON (LOWO 156 FOLDS, 18-MONTH TARGET WINDOW)
========================================================================================
Model Architecture                                   OOF MAE       OOF RMSE     WAPE (%)
----------------------------------------------------------------------------------------
Mean Count per Month x Category                       10.53         19.24        74.0%
Stage 1: In-Fold Ratio-of-Sums Anchor                  8.65         16.18        60.8%
Stage 2: Kernel-Smoothed Rates (ℓ = 2.5 km)            7.74         15.10        54.4%
Production: Ward-Total + Kriging (ℓ = 3.5 km) + Pulse  6.64         13.88        46.6%
----------------------------------------------------------------------------------------
Net Improvement over Baseline Anchor:                 -23.3%        -14.2%       -14.2%
========================================================================================
```

### Category-by-Category Accuracy Profile
```
Category        | OOF MAE  | OOF RMSE | OOF WAPE  | Mean Monthly Volume
-------------------------------------------------------------------------
trees           |    2.856 |    5.474 |    59.72% |    4.78
other           |    3.148 |    5.476 |    47.06% |    6.69
animals_health  |    3.601 |    7.123 |    45.02% |    8.00
roads           |    5.686 |   13.429 |    46.45% |   12.24
electrical      |   11.770 |   18.601 |    39.55% |   29.76
solid_waste     |   12.756 |   22.773 |    53.39% |   23.89
```

### Ward-Level Absolute Error Distribution
* **Minimum Ward MAE:** $1.35$ complaints/month (compact core wards)
* **25th Percentile:** $3.30$ complaints/month
* **Median Ward MAE:** **$4.83$ complaints/month**
* **75th Percentile:** $7.49$ complaints/month
* **Maximum Ward MAE:** $29.98$ complaints/month (outer mega-wards like Ward 198 Hemmigepura, actual volume $\approx 58.8$/month)

---

## 4. Production Model Architecture

```
                         [Raw 18-Month Target Window Data]
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
  [156 Ward Cross-Sectional Totals]             [City-Wide Temporal Pulse Shape]
  Y_total(w, c) = Σ_{t=1}^{18} y_{w,t,c}        Shape(t, c) = Σ_w y_{w,t,c} / Σ_w Y_total(w,c)
                 │                                             │
                 ▼                                             │
  [Ridge Regression on 14 Log Covariates]                      │
  log(Ŷ_static) = β_0 + Σ β_k log(x_k + 1)                     │
                 │                                             │
                 ▼                                             │
  [In-Fold Gaussian Residual Kriging]                          │
  r_j = log(Y_act + 1) - log(Ŷ_static + 1)                     │
  W_uj = exp(- d(u, j)^2 / (2ℓ^2)), ℓ = 3.5 km                 │
  Ŷ_kriged(u, c) = Ŷ_static(u, c) · exp(Σ W̃_uj r_j)           │
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        │
                                        ▼
                  [Monthly Output Prediction: 4,320 Rows]
                  ŷ_{u,t,c} = Ŷ_kriged(u, c) × Shape(t, c)
```

1. **Step 1: 14 Log-Transformed Predictors:**  
   $$\mathbf{z}_w = [\log(\text{pop}+1), \log(\text{hh}+1), \log(\text{road\_len}+1), \log(\text{streetlights}+1), \dots]$$
   Extracted strictly from validated numeric columns in `ward_features.csv`.
2. **Step 2: Ridge Regression on 18-Month Totals:**  
   Solves $(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})\boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}$ with $\ell_2$ regularization ($\lambda = 5.0$).
3. **Step 3: Gaussian Residual Kriging ($\ell = 3.5\text{ km}$):**  
   Smooths localized micro-contagion across adjacent ward neighbors using planar UTM 43N Euclidean distances.
4. **Step 4: Temporal Allocation via Pulse Shape:**  
   Multiplies the static spatial total by the observed in-fold monthly curve, completely preserving category seasonality (monsoon road/tree spikes).
5. **Step 5: Median-Loss Alignment:**  
   Applies shrinkage factor $\kappa = 0.95$ to account for Poisson skewness.

---

## 5. Policy Decision Support: CPI & Equity Sensitivity

1. **Percentile-Rank CPI:**
   $$\text{CPI}_w = 0.40 \cdot \text{Rank}(\text{Demand}_w) + 0.30 \cdot \text{Rank}(\text{Density}_w) + 0.15 \cdot \text{Rank}(\text{Growth}_w) + 0.15 \cdot (1 - \text{Rank}(\text{Infra}_w))$$
2. **Low-Reporting Equity Scenario:**
   Corrects for digital reporting bias where affluent wards file disproportionately more complaints than underserved wards:
   $$\text{ReportingProxy}_w = 0.5 \cdot \text{Rank}(\text{Transit}_w) + 0.5 \cdot (1 - \text{Rank}(\text{Density}_w))$$
   $$\text{EquityDemand}_w = \frac{\text{Rank}(\text{Demand}_w)}{\text{ReportingProxy}_w + 0.2}$$
3. **Monte Carlo Sensitivity Audit:**
   Under $100$ random weight perturbations ($\pm 20\%$), the ranking maintains **Kendall's $\tau = 0.944$**. Exactly $30$ of $40$ wards maintain rock-solid priority classifications, while $10$ borderline wards are flagged for administrative discretion.

---

## 6. Execution & Verification Commands

```bash
# 1. Run the entire backend training & inference pipeline
python3 -m backend.pipeline

# 2. Run automated validation test suite
python3 tests/test_contracts.py && python3 tests/test_spatial_integrity.py

# 3. Launch interactive decision dashboard
streamlit run app.py
```
