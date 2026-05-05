"""
tests/test_pipeline.py
-----------------------
Basic unit tests for data generation, preprocessing, and analysis modules.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest

from src.generate_data import generate_patient_data
from src.preprocess    import handle_missing, add_derived_features
from src.analysis      import (disease_prevalence, correlation_matrix,
                               risk_factor_analysis, chi_square_tests)


@pytest.fixture(scope="module")
def raw_df():
    return generate_patient_data(n=200)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    df = raw_df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df = handle_missing(df)
    df = add_derived_features(df)
    return df


# ── Data Generation ──────────────────────────────────────────────────────────
class TestDataGeneration:
    def test_row_count(self, raw_df):
        assert len(raw_df) == 200

    def test_required_columns(self, raw_df):
        required = ["patient_id", "age", "gender", "bmi",
                    "systolic_bp", "blood_glucose_mg_dl", "primary_diagnosis"]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_age_range(self, raw_df):
        assert raw_df["age"].between(1, 95).all()

    def test_bmi_range(self, raw_df):
        assert raw_df["bmi"].dropna().between(14, 55).all()

    def test_unique_patient_ids(self, raw_df):
        assert raw_df["patient_id"].nunique() == len(raw_df)


# ── Preprocessing ────────────────────────────────────────────────────────────
class TestPreprocessing:
    def test_no_missing_after_impute(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_risk_score_bounds(self, clean_df):
        assert clean_df["risk_score"].between(0, 6).all()

    def test_bmi_category_exists(self, clean_df):
        assert "bmi_category" in clean_df.columns
        expected = {"Underweight", "Normal", "Overweight", "Obese I", "Obese II+"}
        actual   = set(clean_df["bmi_category"].astype(str).unique())
        assert actual.issubset(expected)

    def test_age_group_labels(self, clean_df):
        expected = {"<18", "18-35", "36-50", "51-65", "65+"}
        actual   = set(clean_df["age_group"].astype(str).unique())
        assert actual.issubset(expected)

    def test_bp_stage_values(self, clean_df):
        valid = {"Normal", "Elevated", "Stage 1 HT", "Stage 2 HT"}
        assert set(clean_df["bp_stage"].unique()).issubset(valid)


# ── Analysis ─────────────────────────────────────────────────────────────────
class TestAnalysis:
    def test_prevalence_sums_to_100(self, clean_df):
        prev = disease_prevalence(clean_df)
        assert abs(prev["prevalence_pct"].sum() - 100.0) < 0.5

    def test_correlation_matrix_shape(self, clean_df):
        corr = correlation_matrix(clean_df)
        assert corr.shape[0] == corr.shape[1]

    def test_correlation_diagonal_ones(self, clean_df):
        corr = correlation_matrix(clean_df)
        assert np.allclose(np.diag(corr.values), 1.0)

    def test_risk_factor_output(self, clean_df):
        rf = risk_factor_analysis(clean_df)
        assert "factor" in rf.columns
        assert "mean_risk" in rf.columns
        assert len(rf) > 0

    def test_chi_square_output(self, clean_df):
        cs = chi_square_tests(clean_df)
        assert "p_value" in cs.columns
        assert len(cs) == 5         # 5 factors tested
        assert cs["p_value"].between(0, 1).all()
