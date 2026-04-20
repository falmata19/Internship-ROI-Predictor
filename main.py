"""
main.py — Internship ROI Predictor API
Run: uvicorn main:app --reload
"""

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional

app = FastAPI(title="Internship ROI Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
try:
    with open("model_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    MODEL = artifacts["model"]
    LE_ROLE = artifacts["le_role"]
    LE_INDUSTRY = artifacts["le_industry"]
    LE_LOCATION = artifacts["le_location"]
    print("Model loaded successfully.")
except FileNotFoundError:
    MODEL = None
    print("WARNING: model_artifacts.pkl not found. Run train_model.py first.")


# ---------------------------------------------------------------------------
# Growth rates by industry (annualized, years 1-5)
# ---------------------------------------------------------------------------
GROWTH_RATES = {
    "bigtech":     [0.35, 0.18, 0.15, 0.14, 0.12],
    "startup":     [0.30, 0.25, 0.20, 0.18, 0.15],
    "finance":     [0.32, 0.20, 0.18, 0.15, 0.12],
    "consulting":  [0.28, 0.18, 0.15, 0.13, 0.12],
    "healthcare":  [0.22, 0.15, 0.12, 0.10, 0.10],
    "retail":      [0.18, 0.12, 0.10, 0.08, 0.08],
    "gov":         [0.15, 0.10, 0.08, 0.07, 0.07],
}

WHATIF_MULTIPLIERS = {
    "negotiate": lambda salaries: [round(s * 1.10) for s in salaries],
    "relocate_sf": lambda salaries: [round(s * 1.18) for s in salaries],
    "mba": lambda salaries: [
        round(s * (1.30 if i >= 3 else 1.0)) for i, s in enumerate(salaries)
    ],
    "switch_bigtech": lambda salaries: [
        round(s * (1.15 + i * 0.02)) for i, s in enumerate(salaries)
    ],
}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
RoleType = Literal["swe", "ds", "pm", "finance", "consulting", "design", "other"]
IndustryType = Literal["bigtech", "startup", "finance", "consulting", "healthcare", "retail", "gov"]
LocationType = Literal["sf", "nyc", "seattle", "austin", "chicago", "boston", "remote", "other"]
WhatIfType = Literal["negotiate", "relocate_sf", "mba", "switch_bigtech"]

class PredictRequest(BaseModel):
    hourly_pay: float = Field(..., gt=0, lt=500, description="Internship hourly pay rate")
    role: RoleType
    industry: IndustryType
    location: LocationType
    what_if: Optional[WhatIfType] = None

class PredictResponse(BaseModel):
    base_year1: int
    projection: list[int]          # 6 values: year 0 (base) through year 5
    what_if_projection: list[int] | None
    industry_avg_projection: list[int]
    model_source: str
    confidence_interval: dict      # low / high bounds for year 5


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def encode_safe(le, value: str) -> int:
    """Encode a label, falling back to 0 if unseen."""
    try:
        return int(le.transform([value])[0])
    except ValueError:
        return 0


def predict_base_salary(hourly_pay: float, role: str, industry: str, location: str) -> int:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    role_enc = encode_safe(LE_ROLE, role)
    industry_enc = encode_safe(LE_INDUSTRY, industry)
    location_enc = encode_safe(LE_LOCATION, location)

    X = np.array([[role_enc, industry_enc, location_enc]])
    model_pred = float(MODEL.predict(X)[0])

    # Blend model prediction with internship pay signal (annualized * conversion factor)
    pay_signal = hourly_pay * 2080 * 1.55
    blended = model_pred * 0.65 + pay_signal * 0.35

    return max(40000, round(blended))


def build_projection(base: int, industry: str) -> list[int]:
    rates = GROWTH_RATES.get(industry, GROWTH_RATES["bigtech"])
    proj = [base]
    for rate in rates:
        proj.append(round(proj[-1] * (1 + rate)))
    return proj


INDUSTRY_BASELINES = {
    "bigtech": 130000, "startup": 105000, "finance": 120000,
    "consulting": 95000, "healthcare": 85000, "retail": 72000, "gov": 65000,
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    base = predict_base_salary(req.hourly_pay, req.role, req.industry, req.location)
    projection = build_projection(base, req.industry)

    what_if_proj = None
    if req.what_if and req.what_if in WHATIF_MULTIPLIERS:
        what_if_proj = WHATIF_MULTIPLIERS[req.what_if](projection)

    industry_base = INDUSTRY_BASELINES.get(req.industry, 90000)
    avg_proj = build_projection(industry_base, req.industry)

    # Simple confidence interval: ±12% at year 5
    y5 = projection[-1]
    ci = {"low": round(y5 * 0.88), "high": round(y5 * 1.12)}

    return PredictResponse(
        base_year1=projection[1],
        projection=projection,
        what_if_projection=what_if_proj,
        industry_avg_projection=avg_proj,
        model_source="XGBoost trained on DOL H1B FY2023 (~600k records)",
        confidence_interval=ci,
    )


@app.get("/industries")
def industries():
    return {
        "bigtech": {"label": "Big Tech", "avg_base": 130000},
        "startup": {"label": "Startup", "avg_base": 105000},
        "finance": {"label": "Finance / Banking", "avg_base": 120000},
        "consulting": {"label": "Consulting", "avg_base": 95000},
        "healthcare": {"label": "Healthcare / Biotech", "avg_base": 85000},
        "retail": {"label": "Retail / Consumer", "avg_base": 72000},
        "gov": {"label": "Government / Nonprofit", "avg_base": 65000},
    }
