# 🏥 Patient Health Analytics

> Analyzing patient data to uncover disease trends, health patterns, and risk factors using Python, Pandas, and Machine Learning.

---

## 📌 Project Overview

This project performs end-to-end analysis of patient health records to:

- Identify the **prevalence** of diseases across populations
- Discover **demographic patterns** (age, gender, region)
- Quantify **risk factors** (BMI, blood glucose, smoking, activity level)
- Detect **correlations** between health metrics
- Track **temporal admissions trends** over time
- Predict **high-risk patients** using a Random Forest classifier

---

## 🗂️ Project Structure

```
patient-health-analytics/
│
├── data/
│   ├── patients_raw.csv          # Generated synthetic dataset
│   └── patients_clean.csv        # Cleaned & feature-engineered dataset
│
├── src/
│   ├── generate_data.py          # Synthetic patient data generator
│   ├── preprocess.py             # Data cleaning & feature engineering
│   ├── analysis.py               # Statistical analysis functions
│   ├── visualize.py              # Chart generation (9 plots)
│   └── model.py                  # ML risk prediction (Random Forest)
│
├── reports/
│   ├── disease_prevalence.csv
│   ├── correlation_matrix.csv
│   ├── risk_factors.csv
│   ├── chi_square_tests.csv
│   ├── high_risk_patients.csv
│   └── figures/                  # All generated PNG charts
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── tests/
│   └── test_pipeline.py
│
├── main.py                       # ▶ Run the full pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.9+** | Core language |
| **Pandas** | Data loading, cleaning, aggregation |
| **NumPy** | Numerical operations |
| **Matplotlib / Seaborn** | Statistical visualizations |
| **SciPy** | Chi-square significance tests |
| **Scikit-learn** | Random Forest, preprocessing, evaluation |
| **Faker** | Realistic synthetic data generation |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/patient-health-analytics.git
cd patient-health-analytics
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python main.py
```

This will:
- Generate 2,000 synthetic patient records
- Clean and preprocess the data
- Run all statistical analyses
- Save 12 charts to `reports/figures/`
- Train and evaluate the ML risk classifier

---

## 📊 Key Analyses

### Disease Prevalence
Calculates the percentage of patients diagnosed with each condition — Hypertension, Diabetes Type 2, Obesity, and others.

### Risk Factor Analysis
A composite **risk score (0–6)** is computed per patient based on:
- BMI ≥ 30
- Blood glucose ≥ 126 mg/dL (diabetes threshold)
- Systolic BP ≥ 140 mmHg
- Cholesterol ≥ 240 mg/dL
- Family history of disease
- Current smoker

### Statistical Tests
Chi-square tests evaluate whether smoking status, physical activity, region, gender, and age group are significantly associated with disease diagnosis.

### ML Risk Prediction
A **Random Forest** model predicts whether a patient is high-risk (score ≥ 4), evaluated with:
- ROC-AUC score
- 5-fold cross-validation
- Confusion matrix
- Feature importance ranking

---

## 📈 Sample Visualizations

| Chart | Description |
|---|---|
| `01_disease_prevalence.png` | Horizontal bar — disease frequency |
| `02_age_by_disease.png` | Box plot — age distribution per diagnosis |
| `03_correlation_heatmap.png` | Heatmap of feature correlations |
| `04_bmi_vs_glucose.png` | Scatter plot with diagnosis coloring |
| `05_risk_score_distribution.png` | Histogram + boxplot by smoking |
| `06_regional_heatmap.png` | Disease counts across regions |
| `07_temporal_trend.png` | Monthly admissions over 3 years |
| `08_activity_vs_disease.png` | Stacked bar — activity vs diagnosis |
| `09_bp_stage_distribution.png` | Pie chart — BP stage breakdown |
| `10_roc_curve.png` | ROC curve for risk classifier |
| `11_confusion_matrix.png` | Prediction confusion matrix |
| `12_feature_importance.png` | Top predictors in the RF model |

---

## ⚠️ Disclaimer

> All patient data in this project is **100% synthetic** and generated programmatically using the Faker library. No real patient records are used. This project is for **educational and research demonstration purposes only**.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss changes.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes
4. Open a Pull Request
