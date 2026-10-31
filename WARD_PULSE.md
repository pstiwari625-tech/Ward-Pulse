# WARD PULSE: System Architecture & Technical Implementation Blueprint
**Civic-Demand Intelligence & Ward-Level Prediction Platform**  
*Document Version:* 1.0.0-PROD  
*Target Environment:* Municipal Hackathon (Lucknow, India) / Local Execution  
*Evaluation Scope:* Technical Jury & Production-Readiness Assessment

---

## 1. Executive Architecture Overview

**Ward Pulse** is an offline-capable, spatio-temporal civic demand intelligence and decision-support system. It addresses municipal resource misallocation across city wards by quantifying and predicting localized civic grievances.

### Architectural Reality Check
Analysis of the raw data reveals that **Ward Pulse is not a standard auto-regressive time-series forecasting problem**, but an **inductive spatio-temporal transfer problem**:
1. **The Disjoint Partition:** The dataset provides historical complaint records for **156 wards** (Feb 2020 – Jun 2025), while the target inference set consists of **40 completely unobserved target wards** over an 18-month horizon (Jan 2024 – Jun 2025). Target wards have **zero historical complaint records**.
2. **The Core ML Task:** The model must learn an inductive mapping from **static ward-level urban infrastructure, demographic features, spatial contiguity, and macro seasonal cycles** to ward-level monthly complaint volume across 6 civic categories (`electrical`, `solid_waste`, `roads`, `animals_health`, `trees`, `other`).
3. **Decoupled Scoring vs. Prediction:** The system strictly separates:
   * **Trained ML Predictions:** Statistically learned expected complaint counts per ward, month, and category.
   * **Civic Priority Index (CPI):** A transparent, deterministic, multi-criteria heuristic score combining predicted grievance pressure with infrastructural density and socio-demographic vulnerability. Heuristic scores are **never** obfuscated as ML predictions.

The architecture enforces a strict **unidirectional, file-contract-driven pipeline**: Backend ML outputs static, schema-validated artifacts (`ward_predictions.csv`, `model_metrics.json`, `model_metadata.json`), which a decoupled Streamlit frontend ingests asynchronously.

---

## 2. Dataset Feasibility Assessment

An exhaustive audit of the project workspace (`p3-ward-civic-demand`) yields the following empirical findings:

```
Total Files Audited: 6
├── README.md                      (1,016 B)
├── grievances_public_wards.csv    (51.15 MB, 556,561 records)
├── ward_features.csv              (42.11 KB, 198 ward records)
├── wards.geojson                  (995.38 KB, 198 polygon features)
├── prediction_grid_template.csv   (90.25 KB, 4,320 target rows)
└── sample_submission.csv          (191 B, schema template)
```

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 198 TOTAL WARDS                                        │
│ ┌───────────────────────────────────────┐  ┌───────────────────────┐  ┌──────────────┐ │
│ │          156 TRAINING WARDS           │  │    40 TARGET WARDS     │  │   2 EXCLUDED │ │
│ │  (556,561 Historical Complaints,     │  │ (Zero History in Train,│  │  (Wards 3 &   │ │
│ │   Feb 2020 – Jun 2025)                │  │  18-Month Pred Grid)   │  │   65)        │ │
│ └───────────────────────────────────────┘  └───────────────────────┘  └──────────────┘ │
│                                                                                        │
│ Polygon Geometries in wards.geojson: 198 / 198 (100% Matched)                          │
│ Static Attributes in ward_features.csv: 198 / 198 (100% Matched)                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 File Inventory & Integrity Audit
1. **`grievances_public_wards.csv`:**
   * **Row Count:** 556,561 rows.
   * **Columns (7):** `complaint_id` (int64), `category_group` (object), `category_raw` (object), `sub_category` (object), `grievance_time` (string: `YYYY-MM-DD HH:MM:SS`), `ward_id` (int64), `status` (object).
   * **Temporal Span:** `2020-02-08 01:01:00` to `2025-06-19 12:59:00` (65 contiguous calendar months).
   * **Ward Distribution:** Exactly 156 unique `ward_id`s.

2. **`ward_features.csv`:**
   * **Row Count:** 198 rows (1 per ward).
   * **Columns (30):** Structural census and infrastructure indicators: `ward_id`, `ward_name`, `bbmp_zone_name`, `assembly_constituency_name`, `mp_constituency`, `population_2011`, `area_sq_km`, `households_census_2011`, `road_length_kms`, `lakes`, `lake_area_sq_m`, `parks`, `park_area_sq_m`, `playgrounds`, `playground_area_sq_m`, `govt_schools`, `police_stations`, `fire_stations`, `bus_stops`, `bmtc_ttmc`, `streetlights`, `police_stn`, `bbmp_division`, `bbmp_sub_division`, `population_density_2011_persons_sq_km`, `decadal_popn_growth_rate_2011_2001`, `decadal_households_growth_rate_2011_2001`, `bus_routes`, `centroid_lon`, `centroid_lat`.

3. **`wards.geojson`:**
   * **Features:** 198 Polygon/MultiPolygon records matching `properties.ward_id` in [1, 198]. WGS84 CRS (`EPSG:4326`). Coordinate bounds align with Bengaluru municipal coordinates ($77.45^\circ\text{E} - 77.78^\circ\text{E}, 12.83^\circ\text{N} - 13.14^\circ\text{N}$).

4. **`prediction_grid_template.csv`:**
   * **Row Count:** Exactly 4,320 rows ($40\text{ wards} \times 18\text{ months } [2024\text{-}01 \text{ to } 2025\text{-}06] \times 6\text{ categories}$).

### 2.2 Category & Target Variable Analysis
Grievances map to 6 distinct categories. Notice that **there is no standalone `water` category** in the ground-truth data (water complaints are subsumed under `other` or `engineering/roads`).

| Category Group | Total Complaints | Percentage | Primary Sub-Categories |
| :--- | :--- | :--- | :--- |
| `electrical` | 225,303 | 40.48% | Street light not working, wire snapping, pole repair |
| `solid_waste` | 137,975 | 24.79% | Garbage dump, sweeping not done, burning garbage |
| `roads` | 92,458 | 16.61% | Potholes, road cutting, debris removal, footpath maintenance |
| `animals_health` | 40,355 | 7.25% | Animal birth control (ABC), stray dog neutering, dead animals |
| `other` | 34,689 | 6.23% | Encroachments, illegal banners, health nuisances |
| `trees` | 25,781 | 4.63% | Fallen trees, branch trimming, hazardous tree clearance |

### 2.3 Data Quality & Consistency Audit
* **Missing Values in `ward_features.csv`:**
  * Free-text field `police_stn` has 112 nulls ($56.5\%$). Drop `police_stn` (unstructured text); retain numeric `police_stations` (0 nulls).
  * Formatted numeric columns contain string formatting artifacts:
    * `population_density_2011_persons_sq_km`: Contains commas (e.g. `"3,182"`).
    * `decadal_popn_growth_rate_2011_2001` & `decadal_households_growth_rate_2011_2001`: Contain percentage symbols (e.g. `"59.10%"`).
    * Coerce via robust regex cleansing to `float64`.
* **Missing Wards Anomaly:**
  * Total wards in city geometry = 198.
  * Train wards = 156. Target wards = 40.
  * $156 + 40 = 196$.
  * Two wards—**Ward 3 (Atturu, Yelahanka)** and **Ward 65 (Kadu Malleshwara Ward, West)**—are present in `ward_features.csv` and `wards.geojson` but are **absent from both train complaints and the target prediction grid**.
  * The ingestion pipeline flags Wards 3 and 65 as `UNOBSERVED_NON_EVALUATED`, enabling spatial feature computation (e.g., polygon neighbors) while excluding them from the supervised evaluation split.

---

## 3. Architecture Design Principles

1. **Contract-First Decoupling:** Backend ML and Frontend UI communicate exclusively through deterministic, versioned flat files (`.csv`, `.json`, `.geojson`). No live IPC, shared memory, or dynamic API dependencies.
2. **Strict Spatial Integrity (Anti-Leakage):** To evaluate performance reliably on 40 unobserved wards, cross-validation must be strictly **Spatial Group K-Fold** partitioned on `ward_id`. Standard K-Fold or random temporal splits leak spatial ward identity and yield invalid, optimistic evaluation metrics.
3. **Transparent Separation of Science and Heuristics:**
   The dashboard explicitly color-codes and separates model predictions ($\widehat{Y}$) from the heuristic Civic Priority Index ($\text{CPI}$).
4. **Reproducibility & Zero External Infrastructure:** The entire pipeline executes locally with a clean Python 3.10+ virtual environment (`pandas`, `numpy`, `scikit-learn`, `geopandas`, `joblib`, `streamlit`, `plotly`, `folium`). No Docker, Kafka, or cloud services required.

---

## 4. High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND PROCESSING PIPELINE                               │
│                                                                                        │
│  [Raw Data Store]                                                                      │
│  ├── grievances_public_wards.csv                                                       │
│  ├── ward_features.csv                                                                 │
│  ├── wards.geojson                                                                     │
│  └── prediction_grid_template.csv                                                      │
│           │                                                                            │
│           ▼                                                                            │
│  ┌───────────────────────────────┐     ┌───────────────────────────────┐               │
│  │ A. Ingestion & Validation     │ ──▶ │ B. Preprocessing & Cleanse    │               │
│  │ (Schema check, Ward audit)    │     │ (Regex numeric, Zero-fill)    │               │
│  └───────────────────────────────┘     └───────────────────────────────┘               │
│                                                        │                               │
│                                                        ▼                               │
│  ┌───────────────────────────────┐     ┌───────────────────────────────┐               │
│  │ D. ML Training & Spatial CV   │ ◀── │ C. Feature Engineering Layer  │               │
│  │ (Poisson GLM / Random Forest) │     │ (Density, Spatial Lag, Season)│               │
│  └───────────────────────────────┘     └───────────────────────────────┘               │
│           │                                                                            │
│           ▼                                                                            │
│  ┌───────────────────────────────┐     ┌───────────────────────────────┐               │
│  │ E. Model Evaluation & Export  │ ──▶ │ F. Inference & CPI Scoring    │               │
│  │ (Spatial Out-of-Fold Metrics) │     │ (Generate Target Predictions) │               │
│  └───────────────────────────────┘     └───────────────────────────────┘               │
└────────────────────────────────────────────────┬───────────────────────────────────────┘
                                                 │ Output Contracts
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                FILE CONTRACT INTERFACE                                 │
│  ├── data/predictions/ward_predictions.csv     (Ward x Month x Category Predictions)   │
│  ├── data/predictions/ward_summary_latest.csv  (Ward-aggregated Priority & Totals)     │
│  ├── models/model_metrics.json                 (OOF MAE, RMSE, R2, Poisson Deviance)   │
│  └── models/model_metadata.json                (Feature importance, Run timestamp)     │
└────────────────────────────────────────────────┬───────────────────────────────────────┘
                                                 │ File Ingestion
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             FRONTEND PRESENTATION LAYER                                │
│                                (Streamlit Dashboard)                                   │
│                                                                                        │
│  ┌────────────────────────────┐ ┌────────────────────────────┐ ┌─────────────────────┐ │
│  │ Choropleth Map (Folium)    │ │ Spatio-Temporal Explorer   │ │ Model Governance &  │ │
│  │ Predicted Demand vs. CPI   │ │ Category Trends (Plotly)   │ │ Explainability View │ │
│  └────────────────────────────┘ └────────────────────────────┘ └─────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Detailed Module Architecture

### Layer A: Data Ingestion Layer (`backend/ingestion.py`)
Loads raw CSV and GeoJSON files with strict type enforcement.

### Layer B: Data Validation Layer (`backend/validation.py`)
Enforces integrity invariants prior to transformations (target grid row counts, date bounds, zero ward leakage).

### Layer C: Data Preprocessing Layer (`backend/preprocessing.py`)
Cleans numeric strings, aggregates complaints into ward-month-category counts, and performs zero-inflation panel Cartesian expansion.

### Layer D: Feature Engineering Layer (`backend/features.py`)
Extracts per-capita infrastructure ratios, topological neighbor lags from `wards.geojson`, and cyclical harmonic seasonal features ($\sin, \cos$).

### Layer E: ML Training Layer (`backend/training.py`)
Trains inductive count regressors across 5-Fold Spatial GroupKFold splitting strictly on `ward_id`.

### Layer F: Model Evaluation Layer (`backend/evaluation.py`)
Computes Spatial Out-of-Fold (OOF) metrics (MAE, RMSE, $R^2$, Poisson deviance).

### Layer G: Inference Layer (`backend/inference.py`)
Generates 4,320 predictions for the 40 unseen wards over the 18-month evaluation horizon.

### Layer H: Priority Classification Layer (`backend/prioritization.py`)
Computes the deterministic Civic Priority Index (CPI):
$$\text{CPI}_w = 0.50 \cdot \widetilde{Y}_w + 0.25 \cdot \widetilde{\text{Density}}_w + 0.15 \cdot \widetilde{\text{Growth}}_w + 0.10 \cdot (1 - \widetilde{\text{Infra}}_w)$$

### Layer I: Data Export Layer
Serializes standardized artifacts to `data/predictions/` and `models/`.

### Layer J: Frontend Presentation Layer (`frontend/app.py`)
Streamlit interactive executive dashboard consuming exported contracts.

---

## 6. Directory Structure

```
ward-pulse/
├── README.md                           # System overview & judge briefing
├── WARD_PULSE.md                       # Full architecture specification (this file)
├── requirements.txt                    # Pinned production dependencies
├── config.py                           # Central configuration
│
├── data/
│   ├── raw/                            # Immutable raw competition assets
│   ├── processed/                      # Intermediary clean feature tables
│   └── predictions/                    # Output contracts ingested by frontend
│       ├── ward_predictions.csv        # Detailed (Ward x Month x Category)
│       └── ward_summary_latest.csv     # Ward aggregate & CPI priority rank
│
├── backend/
│   ├── __init__.py
│   ├── pipeline.py                     # Master CLI driver
│   ├── ingestion.py                    # Schema enforcement & file loading
│   ├── validation.py                   # Data integrity & invariant checks
│   ├── preprocessing.py                # Regex cleansing, panel generation
│   ├── features.py                     # Spatial contiguity & temporal encoding
│   ├── training.py                     # Spatial GroupKFold ML training
│   ├── evaluation.py                   # OOF metrics & model scoring
│   ├── inference.py                    # Generation of target predictions
│   └── prioritization.py               # Deterministic CPI scoring formula
│
├── models/
│   ├── champion_regressor.joblib       # Persisted scikit-learn model artifact
│   ├── model_metrics.json              # Evaluated spatial metrics (MAE, RMSE, R2)
│   └── model_metadata.json             # Feature importances & training metadata
│
├── frontend/
│   ├── __init__.py
│   ├── app.py                          # Streamlit application entrypoint
│   ├── components/
│   │   ├── __init__.py
│   │   ├── map_view.py                 # Folium choropleth component
│   │   ├── charts.py                   # Plotly charts
│   │   └── metrics_cards.py            # KPI metric cards
│   └── utils.py                        # GeoJSON loader & data caching helpers
│
└── tests/
    ├── __init__.py
    ├── test_contracts.py               # Schema validation of backend exports
    └── test_spatial_integrity.py       # Verification of zero train-test leakage
```

---

## 7. Execution Quickstart

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Full Backend ML Pipeline
python3 -m backend.pipeline

# 3. Verify Contracts & Spatial Integrity
pytest tests/

# 4. Launch Dashboard
streamlit run frontend/app.py
```
