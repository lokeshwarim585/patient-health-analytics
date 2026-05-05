"""
preprocess.py
-------------
Cleans and preprocesses the raw patient dataset:
  - Handles missing values
  - Encodes categorical features
  - Adds derived health indicators
  - Outputs a clean CSV ready for analysis
"""

import pandas as pd
import numpy as np
import os


# ── Risk score thresholds ────────────────────────────────────────────────────
BMI_OBESE       = 30.0
GLUCOSE_HIGH    = 126.0
BP_SYSTOLIC_HI  = 140
CHOLESTEROL_HI  = 240


def load_raw(path: str = "data/patients_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["admission_date"])
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric cols with median; categorical with mode."""
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"Missing values after imputation: {df.isna().sum().sum()}")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer clinically meaningful features."""

    # Age group
    bins   = [0, 18, 35, 50, 65, 100]
    labels = ["<18", "18-35", "36-50", "51-65", "65+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # BMI category
    bmi_bins   = [0, 18.5, 25, 30, 35, 100]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese I", "Obese II+"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bmi_bins, labels=bmi_labels, right=False)

    # Blood pressure stage
    def bp_stage(row):
        s, d = row["systolic_bp"], row["diastolic_bp"]
        if s < 120 and d < 80:   return "Normal"
        if s < 130 and d < 80:   return "Elevated"
        if s < 140 or d < 90:    return "Stage 1 HT"
        return "Stage 2 HT"

    df["bp_stage"] = df.apply(bp_stage, axis=1)

    # Diabetes risk flag
    df["high_glucose_flag"] = (df["blood_glucose_mg_dl"] >= GLUCOSE_HIGH).astype(int)

    # High cholesterol flag
    df["high_cholesterol_flag"] = (df["cholesterol_mg_dl"] >= CHOLESTEROL_HI).astype(int)

    # Composite risk score (0–5)
    df["risk_score"] = (
        (df["bmi"]                >= BMI_OBESE).astype(int) +
        (df["blood_glucose_mg_dl"] >= GLUCOSE_HIGH).astype(int) +
        (df["systolic_bp"]         >= BP_SYSTOLIC_HI).astype(int) +
        (df["cholesterol_mg_dl"]   >= CHOLESTEROL_HI).astype(int) +
        df["family_history_of_disease"] +
        (df["smoking_status"] == "Current").astype(int)
    )

    # Season of admission
    df["admission_month"]  = df["admission_date"].dt.month
    df["admission_year"]   = df["admission_date"].dt.year
    season_map = {12:"Winter",1:"Winter",2:"Winter",
                  3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",
                  9:"Autumn",10:"Autumn",11:"Autumn"}
    df["admission_season"] = df["admission_month"].map(season_map)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode for ML-ready export."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    encode_cols = ["gender", "blood_group", "region", "smoking_status",
                   "physical_activity_level", "primary_diagnosis",
                   "age_group", "bmi_category", "bp_stage", "admission_season"]
    for col in encode_cols:
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    return df


def clean_pipeline(raw_path: str = "data/patients_raw.csv",
                   out_path:  str = "data/patients_clean.csv") -> pd.DataFrame:
    df = load_raw(raw_path)
    df = handle_missing(df)
    df = add_derived_features(df)
    df = encode_categoricals(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Clean dataset saved → {out_path}  ({len(df):,} rows × {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    clean_pipeline()
