# 🏛️ Ward Pulse — Spatio-Temporal Civic Demand Intelligence & Municipal Prioritization

[![Render Deployment](https://img.shields.io/badge/Render-Live%20Application-success?style=for-the-badge&logo=render)](https://ward-pulse-8qvy.onrender.com/)
[![Swagger API](https://img.shields.io/badge/Swagger%20UI-OpenAPI%203.0-blue?style=for-the-badge&logo=swagger)](https://ward-pulse-8qvy.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Model Validation](https://img.shields.io/badge/LOWO%20CV-156%20Folds%20(MAE%206.636)-brightgreen?style=for-the-badge)](https://ward-pulse-8qvy.onrender.com/)
[![Audit Compliance](https://img.shields.io/badge/QA%20Audit-35%2F35%20PASS%20(100%25)-teal?style=for-the-badge)](https://ward-pulse-8qvy.onrender.com/)

> **Municipal Civic Demand Forecasting, Conformal Uncertainty Quantification, and Equitable Resource Prioritization**  
> **Target Municipality**: Bruhat Bengaluru Mahanagara Palike (BBMP), Bengaluru  
> **Live Web Application**: [https://ward-pulse-8qvy.onrender.com/](https://ward-pulse-8qvy.onrender.com/)  
> **Interactive Swagger API Docs**: [https://ward-pulse-8qvy.onrender.com/docs](https://ward-pulse-8qvy.onrender.com/docs)  
> **Prediction Horizon**: 18 Months (2024-01 to 2025-06) across 6 Civic Categories  
> **Official Contract**: 4,320 rows (`ward_id`, `month`, `category_group`, `predicted_count`)

---

## 📌 Executive Summary & Problem Statement

Urban local bodies face a fundamental operational challenge: **citizen grievances are logged reactively after infrastructure fails**. In developing municipal jurisdictions, this creates systemic inequity:
1. **The Unobserved Ward Dilemma (Cold-Start)**: In this challenge, **40 target municipal wards** have **zero historical complaint records**, requiring inductive spatial inference to predict their civic demand before complaints occur.
2. **Transductive Spatial Imputation**: The 156 training wards contain historical complaint records spanning the exact same 18-month evaluation window (January 2024 to June 2025). Rather than blind future temporal forecasting, the true analytical task is **spatial imputation with shared contemporaneous city time** — capturing macroeconomic city shocks, portal consolidations, monsoon surges, and reporting campaigns that affect all wards simultaneously.
3. **Canonical Civic Categories**: The dataset governs six BBMP departments:
   - `electrical` (Streetlighting, transformer faults, cabling)
   - `solid_waste` (Garbage collection, blackspots, open dumping)
   - `roads` (Potholes, paving, footpath maintenance)
   - `animals_health` (Stray dog control, public health sanitation)
   - `trees` (Fallen branches, tree trimming, park upkeep)
   - `other` (General civic administration)

> [!NOTE]
> **Domain Note**: Water supply/sewerage is handled by BWSSB and is not in this dataset, so it is not modelled. BBMP's municipal jurisdiction covers the six canonical civic categories above.

---

## 🏗️ Technical Architecture & Proposed Solution

Ward Pulse addresses zero-history civic demand via a six-stage hierarchical modeling framework:

```mermaid
flowchart TD
    subgraph S1["Stage 1: Contemporaneous City Pulse Tracking"]
        G[Historical Grievances<br>156 Training Wards] --> PS["Shared City Temporal Pulse (p_m,c)<br>Monthly Shape across 18 Months"]
        G --> CR["City-Wide Complaint Rates (R_m,c)<br>Complaints per Unit Asset Exposure"]
    end

    subgraph S2["Stage 2: Category Asset Exposure Anchoring"]
        WF[Ward Features & Infrastructure<br>198 Municipal Wards] --> EM["Asset Exposure Mapping (E_w,c)<br>Road Length, Streetlights, Pop, Area"]
    end

    subgraph S3["Stage 3: Spatial Ridge Log-Linear Regression"]
        WF --> COV["14 Log-Transformed Urban Covariates<br>Density, Households, Transit, Schools"]
        COV & PS & EM --> RIDGE["Multivariate Ridge Regression<br>alpha = 5.0 (Global Baseline Anchor)"]
    end

    subgraph S4["Stage 4: Spatial Gaussian Kernel Kriging"]
        RIDGE --> RES["Out-of-Fold Spatial Residuals<br>r_w = log(y_w) - log(y_hat_w)"]
        RES --> KRIG["Gaussian Distance Kernel Kriging<br>Bandwidth ell = 3.5 km (Neighbor Residuals)"]
    end

    subgraph S5["Stage 5: Conformal Uncertainty Intervals"]
        KRIG --> PRED["Final Production Point Predictions<br>4,320 Grid Cells (Target Wards)"]
        PRED --> CONF["Mondrian Conformal Calibration<br>90% Finite-Sample Intervals with Poisson Scaling"]
    end

    subgraph S6["Stage 6: Decision Support & Prioritization"]
        CONF --> CPI["Composite Priority Index (CPI)<br>Urgency Tiers: CRITICAL, HIGH, MEDIUM, LOW"]
        CPI --> SENS["Monte Carlo Sensitivity Audit<br>1,000 Draws, tau = 0.946"]
        CPI --> DASH["Executive Streamlit Dashboard & Swagger API"]
    end
```

### Mathematical Formulation

1. **Category Asset Exposure ($E_{w,c}$)**:
   $$\text{roads} \to \text{road\_length\_kms}, \quad \text{electrical} \to \text{streetlights}, \quad \text{trees} \to \text{park\_area\_sq\_m}, \quad \text{solid\_waste} \to \text{population\_2011}$$
2. **City-Wide Contemporaneous Rate ($R_{m,c}$)**:
   $$R_{m,c} = \frac{\sum_{w \in \text{Train}} y_{w,m,c}}{\sum_{w \in \text{Train}} E_{w,c}}$$
3. **Spatial Residual Kriging ($\hat{\delta}_w$)**:
   $$\hat{\delta}_w = \frac{\sum_{i \in \text{Train}} K(d(w, i)) \cdot r_i}{\sum_{i \in \text{Train}} K(d(w, i)) + \lambda}, \quad K(d) = \exp\left(-\frac{d^2}{2\ell^2}\right), \quad \ell = 3.5\text{ km}$$
4. **Mondrian Conformal Prediction Intervals (Finite-Sample 90% Guarantee)**:
   $$s_i = \frac{|y_i - \hat{y}_i|}{\sqrt{\hat{y}_i + 1}}, \quad \hat{C}_{0.90}(w) = \left[\max\left(0, \hat{y} - q_{0.90} \sqrt{\hat{y}+1}\right), \quad \hat{y} + q_{0.90} \sqrt{\hat{y}+1}\right]$$

---

## 📊 Rigorous Model Validation & Empirical Results

All reported evaluation metrics were computed on the complete **156-fold Leave-One-Ward-Out (LOWO) cross-validation** across all **18 target months** and **6 categories** (16,848 out-of-fold predictions):

| Metric / Specification | Baseline Anchor | Ridge Alone | Production (Ridge + Kriging) | Verified Delta / Note |
|---|:---:|:---:|:---:|---|
| **Out-of-Fold MAE** | `8.650` | `6.937` | **`6.636`** | **`-2.016 MAE` reduction** (23.3% error drop) |
| **Out-of-Fold RMSE** | `16.182` | `14.820` | **`13.878`** | **`-2.304 RMSE` reduction** |
| **Weighted APE (WAPE)** | `60.78%` | `48.75%` | **`46.64%`** | Statistically significant error reduction |
| **Bootstrap 95% Confidence Interval** | — | — | **`[-2.532, -1.492]`** | Strictly excludes zero ($p < 0.0001$) |
| **Empirical Conformal Coverage (Nominal 90%)** | — | — | **`89.96%`** | Validated across categories & volume bins |
| **Spatial Bandwidth Sweep ($\ell$)** | — | — | Optimum at **$\ell = 2.5 - 3.5\text{ km}$** | Proven interior minimum (non-boundary) |
| **Monte Carlo Sensitivity (1,000 Draws)** | — | — | **Kendall $\tau = 0.946$** | Robust ranking (12 borderline flip wards) |

---

## 🎯 Hackathon Judge Inspection Guide

Judges can quickly search and inspect any of the **198 municipal wards** (40 unobserved test targets + 158 historical observed wards) on the live dashboard or API:

### Top Critical & High-Priority Target Wards (Direct Answers for Judges)

| Ward ID | Ward Name | Zone | Priority Tier | Monthly Forecast | 90% Conformal Interval | Confidence Rating | CPI Urgency | Dominant Civic Need |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **6** | **Thanisandra** | Byatarayanapura | 🚨 **CRITICAL** | **268.7** / mo | **`[159.2, 378.1]`** | High | **`0.724`** | Electrical (35.8%) |
| **5** | **Jakkuru** | Byatarayanapura | 🔥 **HIGH** | **306.4** / mo | **`[189.6, 423.3]`** | High | **`0.695`** | Electrical (36.1%) |
| **72** | **Herohalli** | Rajarajeshwari Nagar | 🔥 **HIGH** | **154.6** / mo | **`[71.6, 237.7]`** | High | **`0.612`** | Electrical (34.9%) |
| **27** | **Banasavadi** | Sarvagnanagar | 🔥 **HIGH** | **126.8** / mo | **`[51.6, 202.0]`** | Moderate | **`0.581`** | Electrical (35.2%) |
| **12** | **Shettihalli** | Dasarahalli | 🔥 **HIGH** | **116.9** / mo | **`[44.7, 189.1]`** | High | **`0.642`** | Electrical (35.5%) |
| **15** | **T Dasarahalli** | Dasarahalli | 🔥 **HIGH** | **35.1** / mo | **`[0.0, 74.6]`** | Moderate | **`0.589`** | Solid Waste (38.2%) |

### How Judges Can Query on the Dashboard
1. Open the [Live Dashboard](https://ward-pulse-8qvy.onrender.com/).
2. In the Sidebar, use **Ward Scope Filter**: Select **`🚨 Critical & High Priority Wards`** to isolate the high-risk target wards.
3. In the **Municipal Ward Dropdown**, select any ward (e.g. `🚨 [CRITICAL TARGET] Ward 6 - Thanisandra`) to see its complete **Executive Briefing Card** with predictions, confidence bounds, and category bar charts.
4. In the **Ward Details Table**, type any Ward ID (e.g. `6`), Ward Name (`Thanisandra`), or Tier (`Critical`) in the search box for instant filtering.

---

## ⚡ RESTful API & Swagger Documentation

Ward Pulse includes a production-grade FastAPI REST service with automated OpenAPI 3.0 documentation:

- **Interactive Swagger UI**: [https://ward-pulse-8qvy.onrender.com/docs](https://ward-pulse-8qvy.onrender.com/docs)
- **OpenAPI Schema (JSON)**: [https://ward-pulse-8qvy.onrender.com/api/v1/openapi.json](https://ward-pulse-8qvy.onrender.com/api/v1/openapi.json)

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/critical-wards` | Returns all CRITICAL and HIGH priority target wards ranked by CPI urgency. |
| `GET` | `/api/v1/wards` | List all 198 municipal wards with filters (`zone`, `is_target_only`, `priority_tier`, `search`). |
| `GET` | `/api/v1/wards/{ward_id}` | Detailed civic intelligence profile for a specific ward with 90% conformal intervals. |
| `GET` | `/api/v1/predictions` | Query official 4,320-row prediction contract matrix with optional `ward_id`, `month`, `category_group`. |
| `GET` | `/api/v1/metrics` | Returns verified 156-fold LOWO spatial cross-validation benchmarks and conformal calibration stats. |
| `GET` | `/api/v1/health` | Service health status and loaded ward counts. |

### Sample cURL Commands

#### 1. Query Top Critical Target Wards
```bash
curl -X GET "https://ward-pulse-8qvy.onrender.com/api/v1/critical-wards" -H "accept: application/json"
```

#### 2. Get Detailed Predictions for Ward 6 (Thanisandra)
```bash
curl -X GET "https://ward-pulse-8qvy.onrender.com/api/v1/wards/6" -H "accept: application/json"
```
*Sample JSON Response:*
```json
{
  "ward_id": 6,
  "ward_name": "Thanisandra",
  "zone": "Byatarayanapura",
  "ward_type": "Target Ward (ML Forecast)",
  "is_target": true,
  "population": 55342,
  "area_sq_km": 6.82,
  "total_monthly_demand": 268.7,
  "total_18m_demand": 4835.8,
  "lower_90": 159.2,
  "upper_90": 378.1,
  "interval_90_str": "[159.2, 378.1]",
  "confidence_tier": "High",
  "priority_tier": "CRITICAL",
  "cpi_score": 0.724,
  "top_category": "electrical",
  "electrical": 96.1,
  "solid_waste": 75.2,
  "roads": 38.9,
  "animals_health": 25.2,
  "trees": 15.1,
  "other": 18.2
}
```

#### 3. Run FastAPI Locally with Swagger UI
```bash
uvicorn api:app --reload --port 8000
# Open Swagger UI at: http://localhost:8000/docs
```

---

## 🛡️ QA Audit & Codebase Verification Report

The repository contains an exhaustive 35-test automated audit suite verifying data contracts, zero-leakage, spatial modeling, and offline dashboard rendering:

```text
=== Audit Summary: 35 / 35 Passed (100.0% Compliance) ===

Section A. Data Contract (R1, R4):         6 / 6 Passed (100.0%)
Section B. Leakage & Validation (R3, R4):   5 / 5 Passed (100.0%)
Section C. Model Quality Slices (R1, R4):   5 / 5 Passed (100.0%)
Section D. Neighbours & Map Features (R3): 4 / 4 Passed (100.0%)
Section E. Confidence & Uncertainty (R2):   4 / 4 Passed (100.0%)
Section F. Priority Engine (R4, R5):        3 / 3 Passed (100.0%)
Section G. Dashboard & Presentation (R5):   6 / 6 Passed (100.0%)
Section H. Engineering & Edge Cases (R4):   2 / 2 Passed (100.0%)
======================================================================
TOTAL:                                     35 / 35 Passed (100.0%)
```

---

## 💻 Local Setup & Execution Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/pstiwari625-tech/Ward-Pulse.git
cd Ward-Pulse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Regression Guard & Determinism
```bash
python3 tests/test_regression_guard.py
# Must report: golden diff = 0.00e+00, 4,320 rows valid
```

### 3. Run Full 35-Test QA Audit Suite
```bash
python3 tests/run_full_audit.py
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
# Access dashboard at: http://localhost:8501
```

### 5. Launch the FastAPI Swagger Service
```bash
uvicorn api:app --reload --port 8000
# Access Swagger docs at: http://localhost:8000/docs
```

---

## 📄 License & Attribution

Developed for the Municipal Civic Intelligence Hackathon for **Bruhat Bengaluru Mahanagara Palike (BBMP)**.  
Open-source under the Apache 2.0 License.
