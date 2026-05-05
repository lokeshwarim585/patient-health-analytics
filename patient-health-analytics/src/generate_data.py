"""
generate_data.py
----------------
Generates a realistic synthetic patient dataset for health analytics.
No real patient data is used — all records are fabricated using Faker + NumPy.
"""

import pandas as pd
import numpy as np
from faker import Faker
import os

fake = Faker()
np.random.seed(42)

N = 2000  # number of patients

DISEASES = [
    "Diabetes Type 2", "Hypertension", "Coronary Artery Disease",
    "Asthma", "COPD", "Obesity", "Depression", "Anxiety Disorder",
    "Chronic Kidney Disease", "None"
]

DISEASE_WEIGHTS = [0.12, 0.18, 0.08, 0.10, 0.06, 0.15, 0.09, 0.07, 0.05, 0.10]

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
GENDERS = ["Male", "Female", "Other"]
REGIONS = ["North", "South", "East", "West", "Central"]
SMOKING_STATUS = ["Never", "Former", "Current"]
ACTIVITY_LEVELS = ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]


def generate_patient_data(n: int = N) -> pd.DataFrame:
    ages = np.random.normal(loc=45, scale=18, size=n).clip(1, 95).astype(int)
    genders = np.random.choice(GENDERS, size=n, p=[0.49, 0.49, 0.02])
    bmis = np.random.normal(loc=27, scale=6, size=n).clip(14, 55).round(1)
    systolic_bp = np.random.normal(loc=120, scale=18, size=n).clip(80, 200).astype(int)
    diastolic_bp = np.random.normal(loc=78, scale=12, size=n).clip(50, 130).astype(int)
    blood_glucose = np.random.normal(loc=100, scale=30, size=n).clip(60, 400).round(1)
    cholesterol = np.random.normal(loc=195, scale=40, size=n).clip(100, 350).astype(int)
    heart_rate = np.random.normal(loc=75, scale=12, size=n).clip(45, 130).astype(int)
    smoking = np.random.choice(SMOKING_STATUS, size=n, p=[0.50, 0.25, 0.25])
    activity = np.random.choice(ACTIVITY_LEVELS, size=n, p=[0.30, 0.35, 0.25, 0.10])
    blood_group = np.random.choice(BLOOD_GROUPS, size=n)
    region = np.random.choice(REGIONS, size=n)
    family_history = np.random.choice([0, 1], size=n, p=[0.55, 0.45])
    hospital_visits = np.random.poisson(lam=2.5, size=n).clip(0, 20)

    # Primary disease influenced by age/BMI
    disease_probs = np.tile(DISEASE_WEIGHTS, (n, 1)).astype(float)
    disease_probs[ages > 55, 0] += 0.05   # more diabetes in older
    disease_probs[ages > 55, 1] += 0.08   # more hypertension in older
    disease_probs[bmis > 30, 5] += 0.10   # more obesity diagnosis
    disease_probs /= disease_probs.sum(axis=1, keepdims=True)

    primary_disease = np.array([
        np.random.choice(DISEASES, p=disease_probs[i]) for i in range(n)
    ])

    admission_dates = [fake.date_between(start_date="-3y", end_date="today") for _ in range(n)]

    df = pd.DataFrame({
        "patient_id": [f"P{str(i+1).zfill(5)}" for i in range(n)],
        "age": ages,
        "gender": genders,
        "blood_group": blood_group,
        "region": region,
        "bmi": bmis,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "blood_glucose_mg_dl": blood_glucose,
        "cholesterol_mg_dl": cholesterol,
        "heart_rate_bpm": heart_rate,
        "smoking_status": smoking,
        "physical_activity_level": activity,
        "family_history_of_disease": family_history,
        "hospital_visits_per_year": hospital_visits,
        "primary_diagnosis": primary_disease,
        "admission_date": admission_dates,
    })

    # Inject ~3% missing values to simulate real-world messiness
    for col in ["bmi", "blood_glucose_mg_dl", "cholesterol_mg_dl", "smoking_status"]:
        mask = np.random.choice([True, False], size=n, p=[0.03, 0.97])
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_patient_data()
    df.to_csv("data/patients_raw.csv", index=False)
    print(f"✅ Generated {len(df)} patient records → data/patients_raw.csv")
    print(df.head())
