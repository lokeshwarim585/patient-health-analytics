"""
model.py
--------
Trains a Random Forest classifier to predict high-risk patients
(risk_score >= 4) and evaluates performance with standard ML metrics.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline

FIGURES_DIR = "reports/figures"
MODELS_DIR  = "reports/models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

FEATURE_COLS = [
    "age", "bmi", "systolic_bp", "diastolic_bp",
    "blood_glucose_mg_dl", "cholesterol_mg_dl", "heart_rate_bpm",
    "family_history_of_disease", "hospital_visits_per_year",
    "gender_enc", "smoking_status_enc", "physical_activity_level_enc",
    "region_enc", "high_glucose_flag", "high_cholesterol_flag",
]
THRESHOLD = 4   # risk_score >= 4 → high-risk


def load_data(path: str = "data/patients_clean.csv"):
    df = pd.read_csv(path)
    df["high_risk"] = (df["risk_score"] >= THRESHOLD).astype(int)
    X = df[FEATURE_COLS]
    y = df["high_risk"]
    return X, y, df


def train_and_evaluate(path: str = "data/patients_clean.csv") -> dict:
    X, y, df = load_data(path)
    print(f"\nClass distribution: {dict(y.value_counts())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline with scaler + RF
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, max_depth=8,
                                          class_weight="balanced", random_state=42))
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nTest ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = pipe.named_steps["clf"].feature_importances_
    feat_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # Plots
    _plot_roc(y_test, y_proba, roc_auc)
    _plot_confusion(y_test, y_pred)
    _plot_feature_importance(feat_df)

    return {
        "cv_auc":      cv_scores,
        "test_auc":    roc_auc,
        "report":      report,
        "feat_importance": feat_df,
        "model":       pipe,
    }


def _plot_roc(y_test, y_proba, auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2E86AB", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2E86AB")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — High-Risk Patient Classifier", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "10_roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _plot_confusion(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax,
        display_labels=["Low-Risk", "High-Risk"],
        colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "11_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _plot_feature_importance(feat_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_df)))
    ax.barh(feat_df["feature"], feat_df["importance"], color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
    ax.set_title("Feature Importance — Random Forest", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "12_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


if __name__ == "__main__":
    train_and_evaluate()
