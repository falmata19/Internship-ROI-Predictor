# Internship ROI Predictor

A full-stack salary trajectory predictor trained on real US Department of Labor H1B disclosure data (~600k records, FY2023).

## Stack
- **ML**: XGBoost regression model trained on real H1B data
- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JS + Chart.js

## Setup

### 1. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the model (one-time, ~5 min)
Downloads the DOL H1B FY2023 dataset (~150MB) and trains the XGBoost model.
```bash
cd backend
python train_model.py
```
This saves `model_artifacts.pkl` in the backend folder.

### 3. Start the API server
```bash
cd backend
uvicorn main:app --reload
```
API runs at `http://localhost:8000`
Docs at `http://localhost:8000/docs`

### 4. Open the frontend
```bash
open frontend/index.html
```
Or just drag `frontend/index.html` into your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check if model is loaded |
| POST | `/predict` | Get salary projection |
| GET | `/industries` | List industries + avg salaries |

### Example predict request
```json
POST /predict
{
  "hourly_pay": 50,
  "role": "swe",
  "industry": "bigtech",
  "location": "sf",
  "what_if": "negotiate"
}
```

### Example response
```json
{
  "base_year1": 142000,
  "projection": [89000, 142000, 167000, 193000, 220000, 246000],
  "what_if_projection": [98000, 156000, 184000, 212000, 242000, 271000],
  "industry_avg_projection": [130000, 175000, ...],
  "model_source": "XGBoost trained on DOL H1B FY2023 (~600k records)",
  "confidence_interval": { "low": 216000, "high": 275000 }
}
```

## What-if Scenarios
- `negotiate` — +10% across all years
- `relocate_sf` — +18% (SF cost-of-living premium)
- `mba` — +30% applied from year 3 onward
- `switch_bigtech` — industry multiplier applied if not already in big tech

## Project Structure
```
internship-roi/
├── backend/
│   ├── train_model.py     # Downloads DOL data, trains XGBoost, saves pkl
│   ├── main.py            # FastAPI server
│   └── requirements.txt
└── frontend/
    └── index.html         # Full UI, hits localhost:8000
```

## Resume bullets (once you build it)
- Trained an XGBoost regression model on 600k+ DOL H1B salary disclosure records to predict 5-year compensation trajectories
- Built a FastAPI backend with a `/predict` endpoint serving real-time inference with what-if scenario modeling
- Developed a full-stack web application enabling students to compare internship offers against industry benchmarks using live model predictions
