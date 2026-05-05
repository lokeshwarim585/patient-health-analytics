"""
visualize.py
------------
Generates all charts and saves them to reports/figures/.
Each function saves a PNG and returns the filepath.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

from src.analysis import (
    load_clean, disease_prevalence, correlation_matrix,
    risk_factor_analysis, temporal_trends, regional_disease_pivot,
    health_metrics_summary
)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE   = "Set2"
ACCENT    = "#2E86AB"
BG_COLOR  = "#F8F9FA"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})


def _save(fig, name: str) -> str:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ── 1. Disease Prevalence Bar Chart ─────────────────────────────────────────
def plot_disease_prevalence(df: pd.DataFrame) -> str:
    data = disease_prevalence(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette(PALETTE, len(data))
    bars = ax.barh(data["disease"], data["prevalence_pct"], color=colors, edgecolor="white")
    for bar, val in zip(bars, data["prevalence_pct"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val}%", va="center", fontsize=9)
    ax.set_xlabel("Prevalence (%)")
    ax.set_title("Disease Prevalence Among Patient Population", fontweight="bold", pad=12)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, "01_disease_prevalence.png")


# ── 2. Age Distribution by Disease ──────────────────────────────────────────
def plot_age_distribution(df: pd.DataFrame) -> str:
    top_diseases = df["primary_diagnosis"].value_counts().head(6).index
    subset = df[df["primary_diagnosis"].isin(top_diseases)]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=subset, x="primary_diagnosis", y="age",
                palette=PALETTE, ax=ax, width=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title("Age Distribution by Primary Diagnosis", fontweight="bold", pad=12)
    ax.set_xlabel("Primary Diagnosis")
    ax.set_ylabel("Age (years)")
    fig.tight_layout()
    return _save(fig, "02_age_by_disease.png")


# ── 3. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
    fig.tight_layout()
    return _save(fig, "03_correlation_heatmap.png")


# ── 4. BMI vs Blood Glucose Scatter ─────────────────────────────────────────
def plot_bmi_glucose_scatter(df: pd.DataFrame) -> str:
    sample = df.sample(min(600, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 6))
    diseases = sample["primary_diagnosis"].unique()
    palette  = dict(zip(diseases, sns.color_palette(PALETTE, len(diseases))))
    for disease, grp in sample.groupby("primary_diagnosis"):
        ax.scatter(grp["bmi"], grp["blood_glucose_mg_dl"],
                   label=disease, alpha=0.55, s=28,
                   color=palette[disease], edgecolors="none")
    ax.axhline(126, color="red", linestyle="--", linewidth=1, label="Diabetes threshold (126)")
    ax.axvline(30,  color="orange", linestyle="--", linewidth=1, label="Obese BMI (30)")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Blood Glucose (mg/dL)")
    ax.set_title("BMI vs Blood Glucose by Diagnosis", fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, "04_bmi_vs_glucose.png")


# ── 5. Risk Score Distribution ───────────────────────────────────────────────
def plot_risk_score_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall histogram
    axes[0].hist(df["risk_score"], bins=range(0, 8), color=ACCENT,
                 edgecolor="white", rwidth=0.8, align="left")
    axes[0].set_title("Overall Risk Score Distribution", fontweight="bold")
    axes[0].set_xlabel("Composite Risk Score")
    axes[0].set_ylabel("Number of Patients")

    # By smoking status
    sns.boxplot(data=df, x="smoking_status", y="risk_score",
                palette="pastel", ax=axes[1])
    axes[1].set_title("Risk Score by Smoking Status", fontweight="bold")
    axes[1].set_xlabel("Smoking Status")
    axes[1].set_ylabel("Composite Risk Score")

    fig.suptitle("Patient Risk Score Analysis", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "05_risk_score_distribution.png")


# ── 6. Regional Disease Heatmap ──────────────────────────────────────────────
def plot_regional_heatmap(df: pd.DataFrame) -> str:
    pivot = regional_disease_pivot(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.3, ax=ax, annot_kws={"size": 8})
    ax.set_title("Disease Count by Region", fontweight="bold", pad=12)
    ax.set_ylabel("Region")
    ax.set_xlabel("Primary Diagnosis")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    return _save(fig, "06_regional_heatmap.png")


# ── 7. Temporal Admissions Trend ─────────────────────────────────────────────
def plot_temporal_trend(df: pd.DataFrame) -> str:
    trend = temporal_trends(df)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    x = range(len(trend))
    ax1.bar(x, trend["admissions"], color=ACCENT, alpha=0.6, label="Monthly Admissions")
    ax2.plot(x, trend["avg_risk"], color="crimson", linewidth=2,
             marker="o", markersize=3, label="Avg Risk Score")
    ax1.set_xticks(list(x)[::3])
    ax1.set_xticklabels(trend["ym"].iloc[::3], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Monthly Admissions")
    ax2.set_ylabel("Avg Risk Score", color="crimson")
    ax1.set_title("Monthly Admissions & Average Risk Score Over Time", fontweight="bold", pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
    fig.tight_layout()
    return _save(fig, "07_temporal_trend.png")


# ── 8. Gender & Activity vs Disease (Stacked) ────────────────────────────────
def plot_activity_disease(df: pd.DataFrame) -> str:
    pivot = pd.crosstab(df["physical_activity_level"],
                        df["primary_diagnosis"], normalize="index") * 100
    order = ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=True, colormap=PALETTE, ax=ax, width=0.65)
    ax.set_title("Disease Distribution by Physical Activity Level", fontweight="bold", pad=12)
    ax.set_xlabel("Physical Activity Level")
    ax.set_ylabel("% of Patients")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, "08_activity_vs_disease.png")


# ── 9. BP Stage Pie Chart ────────────────────────────────────────────────────
def plot_bp_stage_pie(df: pd.DataFrame) -> str:
    bp_counts = df["bp_stage"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = sns.color_palette("Set2", len(bp_counts))
    wedges, texts, autotexts = ax.pie(
        bp_counts, labels=bp_counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.80,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("Blood Pressure Stage Distribution", fontweight="bold", pad=20)
    fig.tight_layout()
    return _save(fig, "09_bp_stage_distribution.png")


# ── Generate All ─────────────────────────────────────────────────────────────
def generate_all(path: str = "data/patients_clean.csv") -> list:
    df = load_clean(path)
    print(f"\nGenerating visualizations for {len(df):,} patients...")
    saved = []
    saved.append(plot_disease_prevalence(df))
    saved.append(plot_age_distribution(df))
    saved.append(plot_correlation_heatmap(df))
    saved.append(plot_bmi_glucose_scatter(df))
    saved.append(plot_risk_score_distribution(df))
    saved.append(plot_regional_heatmap(df))
    saved.append(plot_temporal_trend(df))
    saved.append(plot_activity_disease(df))
    saved.append(plot_bp_stage_pie(df))
    print(f"\n✅ {len(saved)} charts saved to {FIGURES_DIR}/")
    return saved


if __name__ == "__main__":
    generate_all()
