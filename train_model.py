"""
train_model.py
--------------
Downloads real H1B salary disclosure data from the US Department of Labor,
trains an XGBoost regression model, and saves it for the API to use.

Run once before starting the server:
    python train_model.py
"""

import pandas as pd
import numpy as np
import requests
import os
import pickle
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# ---------------------------------------------------------------------------
# 1. Download real H1B disclosure data (FY2023, DOL public dataset)
# ---------------------------------------------------------------------------
H1B_URL = (
    "https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/"
    "LCA_Disclosure_Data_FY2023_Q4.xlsx"
)

CACHE_PATH = "h1b_fy2023.xlsx"

def load_data():
    if not os.path.exists(CACHE_PATH):
        print("Downloading H1B disclosure data from DOL (~150MB, one-time)...")
        r = requests.get(H1B_URL, timeout=120)
        r.raise_for_status()
        with open(CACHE_PATH, "wb") as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print("Using cached H1B data.")

    print("Loading Excel file (this may take ~30s)...")
    df = pd.read_excel(CACHE_PATH, engine="openpyxl")
    return df


# ---------------------------------------------------------------------------
# 2. Clean and feature-engineer
# ---------------------------------------------------------------------------
ROLE_MAP = {
    "SOFTWARE": "swe",
    "ENGINEER": "swe",
    "DEVELOPER": "swe",
    "DATA SCIENTIST": "ds",
    "MACHINE LEARNING": "ds",
    "ANALYST": "ds",
    "PRODUCT MANAGER": "pm",
    "FINANCIAL": "finance",
    "QUANTITATIVE": "finance",
    "CONSULTANT": "consulting",
    "DESIGNER": "design",
    "UX": "design",
}

INDUSTRY_MAP = {
    "INFORMATION": "bigtech",
    "TECHNOLOGY": "bigtech",
    "FINANCE": "finance",
    "BANKING": "finance",
    "INSURANCE": "finance",
    "HEALTH": "healthcare",
    "MEDICAL": "healthcare",
    "RETAIL": "retail",
    "CONSULTING": "consulting",
    "MANAGEMENT": "consulting",
    "GOVERNMENT": "gov",
    "EDUCATION": "gov",
}

COL_MAPPING = {
    "WAGE_RATE_OF_PAY_FROM": "wage",
    "WAGE_UNIT_OF_PAY": "wage_unit",
    "SOC_TITLE": "job_title",
    "EMPLOYER_STATE": "state",
    "CASE_STATUS": "status",
    "NAICS_CODE": "naics",
}

STATE_COL_MULTIPLIER = {
    "CA": 1.25, "NY": 1.18, "WA": 1.12, "MA": 1.05,
    "TX": 1.0,  "IL": 0.97, "CO": 1.02, "GA": 0.93,
    "FL": 0.90, "VA": 1.05,
}

def map_role(title):
    t = str(title).upper()
    for kw, role in ROLE_MAP.items():
        if kw in t:
            return role
    return "other"

def map_location(state):
    s = str(state).strip().upper()
    if s in ("CA",):
        return "sf"
    if s in ("NY", "NJ"):
        return "nyc"
    if s in ("WA",):
        return "seattle"
    if s in ("TX",):
        return "austin"
    if s in ("IL",):
        return "chicago"
    if s in ("MA",):
        return "boston"
    return "other"

def clean(df):
    df = df.rename(columns={k: v for k, v in COL_MAPPING.items() if k in df.columns})

    # Keep only certified cases with annual wages
    if "status" in df.columns:
        df = df[df["status"].str.upper().str.contains("CERTIFIED", na=False)]

    # Normalize wage to annual
    if "wage_unit" in df.columns:
        df = df[df["wage_unit"].isin(["Year", "Hour", "Month", "Week", "Bi-Weekly"])]
        unit_mult = {"Year": 1, "Hour": 2080, "Month": 12, "Week": 52, "Bi-Weekly": 26}
        df["wage"] = pd.to_numeric(df["wage"], errors="coerce")
        df["wage"] = df.apply(
            lambda r: r["wage"] * unit_mult.get(r["wage_unit"], 1), axis=1
        )

    # Sanity filter
    df = df[(df["wage"] >= 30000) & (df["wage"] <= 500000)]

    # Feature engineering
    if "job_title" in df.columns:
        df["role"] = df["job_title"].apply(map_role)
    else:
        df["role"] = "other"

    if "state" in df.columns:
        df["location"] = df["state"].apply(map_location)
    else:
        df["location"] = "other"

    df["industry"] = "bigtech"  # default; real NAICS mapping below

    if "naics" in df.columns:
        naics = df["naics"].astype(str).str[:2]
        naics_industry = {
            "51": "bigtech", "52": "finance", "54": "consulting",
            "62": "healthcare", "44": "retail", "45": "retail",
            "92": "gov", "61": "gov",
        }
        df["industry"] = naics.map(naics_industry).fillna("bigtech")

    df = df[["wage", "role", "industry", "location"]].dropna()
    return df


# ---------------------------------------------------------------------------
# 3. Encode + train
# ---------------------------------------------------------------------------
def train(df):
    le_role = LabelEncoder()
    le_industry = LabelEncoder()
    le_location = LabelEncoder()

    df["role_enc"] = le_role.fit_transform(df["role"])
    df["industry_enc"] = le_industry.fit_transform(df["industry"])
    df["location_enc"] = le_location.fit_transform(df["location"])

    X = df[["role_enc", "industry_enc", "location_enc"]]
    y = df["wage"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\nModel performance — MAE: ${mae:,.0f} | R²: {r2:.3f}")

    return model, le_role, le_industry, le_location


# ---------------------------------------------------------------------------
# 4. Save artifacts
# ---------------------------------------------------------------------------
def save(model, le_role, le_industry, le_location):
    artifacts = {
        "model": model,
        "le_role": le_role,
        "le_industry": le_industry,
        "le_location": le_location,
    }
    with open("model_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("Saved model_artifacts.pkl")


if __name__ == "__main__":
    df_raw = load_data()
    df_clean = clean(df_raw)
    print(f"Training on {len(df_clean):,} H1B records...")
    model, le_role, le_industry, le_location = train(df_clean)
    save(model, le_role, le_industry, le_location)
    print("\nDone! Run: uvicorn main:app --reload")
