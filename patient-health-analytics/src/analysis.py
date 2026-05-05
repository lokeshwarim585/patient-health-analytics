"""
analysis.py
-----------
Statistical analysis of the cleaned patient dataset:
  - Disease prevalence & demographics
  - Correlation analysis
  - Risk factor identification
  - Regional and temporal trends
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def load_clean(path: str = "data/patients_clean.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["admission_date"])


# ── 1. Disease Prevalence ────────────────────────────────────────────────────
def disease_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """Returns disease counts + prevalence rate (%)."""
    counts = df["primary_diagnosis"].value_counts().reset_index()
    counts.columns = ["disease", "count"]
    counts["prevalence_pct"] = (counts["count"] / len(df) * 100).round(2)
    return counts


# ── 2. Demographic Breakdown ─────────────────────────────────────────────────
def demographic_breakdown(df: pd.DataFrame) -> dict:
    return {
        "age_group_distribution": df["age_group"].value_counts().to_dict(),
        "gender_distribution":    df["gender"].value_counts().to_dict(),
        "region_distribution":    df["region"].value_counts().to_dict(),
        "disease_by_gender": (
            df.groupby(["primary_diagnosis", "gender"])
              .size().unstack(fill_value=0)
        ),
        "disease_by_age_group": (
            df.groupby(["primary_diagnosis", "age_group"])
              .size().unstack(fill_value=0)
        ),
    }


# ── 3. Numeric Health Metrics Summary ───────────────────────────────────────
def health_metrics_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["age", "bmi", "systolic_bp", "diastolic_bp",
            "blood_glucose_mg_dl", "cholesterol_mg_dl",
            "heart_rate_bpm", "risk_score"]
    return df[cols].describe().round(2)


# ── 4. Correlation Matrix ────────────────────────────────────────────────────
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["age", "bmi", "systolic_bp", "diastolic_bp",
                "blood_glucose_mg_dl", "cholesterol_mg_dl",
                "heart_rate_bpm", "risk_score", "hospital_visits_per_year",
                "family_history_of_disease"]
    return df[num_cols].corr().round(3)


# ── 5. Risk Factor Analysis ──────────────────────────────────────────────────
def risk_factor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Mean risk score by lifestyle factors."""
    rows = []
    for factor in ["smoking_status", "physical_activity_level",
                   "bmi_category", "bp_stage", "age_group"]:
        grp = df.groupby(factor)["risk_score"].agg(["mean", "median", "count"]).reset_index()
        grp.columns = ["category", "mean_risk", "median_risk", "count"]
        grp.insert(0, "factor", factor)
        rows.append(grp)
    return pd.concat(rows, ignore_index=True).round(3)


# ── 6. Chi-Square Tests: Disease vs Categorical Factors ─────────────────────
def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Tests association between primary_diagnosis and lifestyle factors."""
    factors = ["smoking_status", "physical_activity_level",
               "region", "gender", "age_group"]
    results = []
    for factor in factors:
        contingency = pd.crosstab(df["primary_diagnosis"], df[factor])
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        results.append({
            "factor":    factor,
            "chi2_stat": round(chi2, 3),
            "p_value":   round(p, 5),
            "dof":       dof,
            "significant (p<0.05)": p < 0.05,
        })
    return pd.DataFrame(results)


# ── 7. Temporal Trends ───────────────────────────────────────────────────────
def temporal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly admission counts and avg risk score over time."""
    df["ym"] = df["admission_date"].dt.to_period("M")
    trend = (
        df.groupby("ym")
          .agg(admissions=("patient_id","count"),
               avg_risk=("risk_score","mean"))
          .reset_index()
    )
    trend["ym"] = trend["ym"].astype(str)
    return trend.round(3)


# ── 8. Regional Disease Heatmap Data ────────────────────────────────────────
def regional_disease_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["region"], df["primary_diagnosis"])


# ── 9. High-Risk Patient Cohort ──────────────────────────────────────────────
def high_risk_patients(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Returns patients with risk_score >= threshold."""
    hr = df[df["risk_score"] >= threshold].copy()
    print(f"High-risk patients (score ≥ {threshold}): {len(hr):,} ({len(hr)/len(df)*100:.1f}%)")
    return hr[[
        "patient_id","age","gender","region","primary_diagnosis",
        "bmi","systolic_bp","blood_glucose_mg_dl","cholesterol_mg_dl",
        "smoking_status","risk_score"
    ]]


# ── Run All ──────────────────────────────────────────────────────────────────
def run_all(path: str = "data/patients_clean.csv") -> dict:
    df = load_clean(path)
    print(f"\n{'='*55}")
    print("  PATIENT HEALTH ANALYTICS — SUMMARY")
    print(f"{'='*55}")
    print(f"  Total patients : {len(df):,}")
    print(f"  Features       : {df.shape[1]}")

    results = {
        "prevalence":        disease_prevalence(df),
        "demographics":      demographic_breakdown(df),
        "health_summary":    health_metrics_summary(df),
        "correlations":      correlation_matrix(df),
        "risk_factors":      risk_factor_analysis(df),
        "chi_square":        chi_square_tests(df),
        "temporal_trends":   temporal_trends(df),
        "regional_heatmap":  regional_disease_pivot(df),
        "high_risk_cohort":  high_risk_patients(df),
    }

    print("\n── Disease Prevalence ──")
    print(results["prevalence"].to_string(index=False))

    print("\n── Chi-Square Significance Tests ──")
    print(results["chi_square"].to_string(index=False))

    print("\n── Risk Factor Mean Scores ──")
    print(results["risk_factors"].to_string(index=False))

    return results


if __name__ == "__main__":
    run_all()
