"""
main.py
-------
Entry point — runs the full pipeline:
  1. Generate synthetic data
  2. Clean & preprocess
  3. Statistical analysis
  4. Visualizations
  5. ML risk prediction model
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from src.generate_data import generate_patient_data
from src.preprocess    import clean_pipeline
from src.analysis      import run_all
from src.visualize     import generate_all
from src.model         import train_and_evaluate


def main():
    print("=" * 60)
    print("  PATIENT HEALTH ANALYTICS — FULL PIPELINE")
    print("=" * 60)

    # Step 1 — Generate data
    print("\n[1/5] Generating synthetic patient dataset...")
    os.makedirs("data", exist_ok=True)
    df_raw = generate_patient_data(n=2000)
    df_raw.to_csv("data/patients_raw.csv", index=False)
    print(f"      {len(df_raw):,} patient records created.")

    # Step 2 — Preprocess
    print("\n[2/5] Cleaning & preprocessing...")
    df_clean = clean_pipeline("data/patients_raw.csv", "data/patients_clean.csv")

    # Step 3 — Analysis
    print("\n[3/5] Running statistical analysis...")
    results = run_all("data/patients_clean.csv")

    # Save key tables
    os.makedirs("reports", exist_ok=True)
    results["prevalence"].to_csv("reports/disease_prevalence.csv", index=False)
    results["correlations"].to_csv("reports/correlation_matrix.csv")
    results["risk_factors"].to_csv("reports/risk_factors.csv", index=False)
    results["chi_square"].to_csv("reports/chi_square_tests.csv", index=False)
    results["high_risk_cohort"].to_csv("reports/high_risk_patients.csv", index=False)
    print("      Tables saved → reports/")

    # Step 4 — Visualizations
    print("\n[4/5] Generating charts...")
    generate_all("data/patients_clean.csv")

    # Step 5 — ML Model
    print("\n[5/5] Training risk prediction model...")
    model_results = train_and_evaluate("data/patients_clean.csv")

    print("\n" + "=" * 60)
    print("  ✅  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  📁  data/            — raw + clean CSVs")
    print(f"  📁  reports/         — analysis tables + figures")
    print(f"  📊  reports/figures/ — {len(os.listdir('reports/figures'))} charts generated")
    print(f"  🤖  Model Test AUC   — {model_results['test_auc']:.4f}")
    print()


if __name__ == "__main__":
    main()
