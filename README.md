# 🏛️ Ward Pulse — Civic Intelligence Dashboard

> **Predicting ward-wise civic complaint demand for proactive municipal governance.**  
> Target Deployment: **Bruhat Bengaluru Mahanagara Palike (BBMP), Bengaluru**  
> Categories Modelled: **electrical, solid_waste, roads, animals_health, trees, other**

---

## 📌 Project Overview

**Ward Pulse** is a civic intelligence platform for forecasting municipal grievance demand across Bengaluru's municipal wards across 6 civic categories: electrical, solid_waste, roads, animals_health, trees, and other.

> **Note on Water Supply & Sewerage**: Water supply/sewerage is handled by BWSSB and is not in this dataset, so it is not modelled.

In particular, Ward Pulse specializes in **zero-history complaint prediction**, inferring grievance demand in underserved or newly incorporated wards that lack historical digital logs.

---

## 🏗️ Phase 1 Scope & Architecture

Phase 1 establishes the complete frontend design system, modular layout, routing, and backend integration contracts without hardcoding fake ML outputs or blocking concurrent backend work.

### Directory Structure

```
ward-pulse/
│
├── app.py                     # Main application entry point & routing
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & run guide
│
├── components/                # Modular UI components
│   ├── __init__.py
│   ├── sidebar.py             # Sidebar navigation, filters preview, branding
│   ├── dashboard.py           # Header, welcome hero, KPI grid, view compositions
│   ├── charts.py              # Analytics placeholders & Plotly stubs
│   └── ward_map.py            # Ward map placeholder & Folium stubs
│
├── data/
│   └── sample_predictions.csv # Schema preview dataset (Clearly labelled)
│
├── assets/
│   └── style.css              # Custom civic-tech CSS design system
│
└── utils/
    ├── __init__.py
    └── data_loader.py         # Pipeline status detector & CSV contract loader
```

---

## 🎨 UI/UX Design System

The interface adheres to a modern civic-tech design philosophy:

| Role | Color | Hex Code |
| :--- | :--- | :--- |
| **Primary** | Deep Navy | `#142B45` |
| **Secondary** | Teal | `#168C83` |
| **Accent** | Warm Orange | `#F59E0B` |
| **Background** | Light Neutral | `#F5F7FA` |
| **Cards** | Clean White | `#FFFFFF` |
| **Text** | Dark Slate | `#263445` |
| **Muted** | Slate Gray | `#64748B` |

- **Subtle borders & shadows**: Elevated card hierarchy with rounded corners (`12px`).
- **Live status indicators**: Explicit pills and banners showing whether the official ML pipeline or sample preview is active.

---

## 🚀 Setup & Execution

### 1. Requirements

- Python 3.10+ (tested on Python 3.14)
- Streamlit, Pandas, Plotly, Folium, streamlit-folium

### 2. Install Dependencies

```bash
cd ward-pulse
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
cd ward-pulse
streamlit run app.py
```

Or from the root directory:

```bash
streamlit run ward-pulse/app.py
```

---

## 🤝 Backend & ML Integration Contract (For Teammate)

To connect the live machine learning prediction output:

1. Create directory `ward-pulse/data/predictions/` (if not present).
2. Export your predictions to:
   ```
   ward-pulse/data/predictions/ward_predictions.csv
   ```
3. Ensure the CSV contains at least the agreed columns:
   ```csv
   ward_id,predicted_complaints
   101,42
   102,38
   103,29
   ...
   ```
   *(Optional additional columns like `ward_name`, `zone`, `demand_tier` will be automatically rendered).*

4. The frontend dynamically detects this file! Once `ward_predictions.csv` exists, the UI status automatically toggles from **"Official Dataset Pending"** to **"Official ML Predictions Connected"** with zero code changes.

---

## 📋 Hackathon Roadmap

- [x] **Phase 1: Frontend Foundation & UI Setup** *(Complete)*
- [x] **Phase 2: Data Ingestion & Ward Analytics Visualizations** *(Integrated)*
- [x] **Phase 3: Interactive Geo-Spatial Ward Map & Choropleth** *(Integrated)*
- [x] **Phase 4: Comparative Ward Intelligence & Model Governance** *(Integrated)*
