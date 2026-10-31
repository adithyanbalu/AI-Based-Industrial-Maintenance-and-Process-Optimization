"""
AI-Based Industrial Maintenance and Process Optimization System
Predictive Machine Failure Classification Model
Module Owner: Adithyan B (Team Lead)
Infosys Springboard Virtual Internship 7.0
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "Dataset.csv"
OUT = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import feature engineer
try:
    from src.features import IndustrialFeatureEngineer
except ImportError:
    from features import IndustrialFeatureEngineer

# Standard feature definitions
MODEL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"

# Leakage columns to strictly exclude
EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]



def get_feature_lists():
    """Returns numerical and categorical feature column names."""
    categorical_features = ["Type"]
    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    engineered_numerical = numerical_features + [
        "Temp_diff",
        "Power_W",
        "Strain",
    ]
    return categorical_features, numerical_features, engineered_numerical


def build_pipeline():
    """
    Constructs the end-to-end scikit-learn pipeline:
    IndustrialFeatureEngineer -> Preprocessor -> Balanced Tuned RandomForestClassifier
    """
    categorical_features, _, engineered_numerical = get_feature_lists()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, engineered_numerical),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model_pipeline = Pipeline([
        ("engineer", IndustrialFeatureEngineer()),
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    return model_pipeline


def train_improved_model(data_path=DATA_PATH, save_artifacts=True):
    """
    Trains the improved predictive maintenance model and generates artifacts.
    """
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()

    # Stratified split to preserve 3.39% failure class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records: {len(X_train)} (Failures: {int(y_train.sum())})")
    print(f"Testing records:  {len(X_test)} (Failures: {int(y_test.sum())})")

    # Train improved model
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_fail = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_fail = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_fail = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Failure", "Failure"],
        zero_division=0,
    )

    results_text = f"""
==============================================================
AI-BASED INDUSTRIAL MAINTENANCE - IMPROVED MODEL RESULTS
==============================================================
Total Dataset Rows:       {len(df)}
Training Samples:         {len(X_train)} (Class 0: {len(X_train)-int(y_train.sum())}, Class 1: {int(y_train.sum())})
Testing Samples:          {len(X_test)} (Class 0: {len(X_test)-int(y_test.sum())}, Class 1: {int(y_test.sum())})
Input Features Used:      {MODEL_FEATURES}
Domain Derived Features:  ['Temp_diff', 'Power_W', 'Strain']
Target Column:            {TARGET}

Model Architecture:
-------------------
Algorithm:                Random Forest Classifier
Number of Estimators:     250
Max Depth:                12
Min Samples Split:        4
Class Weighting:          balanced
Class Imbalance Handling: Cost-sensitive balanced weights + physics domain features

EVALUATION METRICS (TEST SET, N={len(X_test)}):
----------------------------------------------
Overall Accuracy:         {acc * 100:.2f}%
Failure Class Precision:  {prec_fail * 100:.2f}%
Failure Class Recall:     {rec_fail * 100:.2f}%  (Catches {cm[1, 1]} of {cm[1, 0] + cm[1, 1]} failures!)
Failure Class F1-Score:   {f1_fail:.4f}
ROC-AUC Score:            {roc_auc:.4f}

CONFUSION MATRIX:
-----------------
                   Predicted No Failure    Predicted Failure
Actual No Failure: {cm[0, 0]:>20} {cm[0, 1]:>20}
Actual Failure:    {cm[1, 0]:>20} {cm[1, 1]:>20}

False Negatives (Missed Failures): {cm[1, 0]} (Reduced from 24 in baseline to {cm[1, 0]})
False Positives (False Alarms):    {cm[0, 1]}

CLASSIFICATION REPORT:
----------------------
{report}
==============================================================
"""
    print(results_text)

    # Feature importances
    clf = pipeline.named_steps["classifier"]
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_[
        "categorical"
    ].named_steps["encoder"]
    cat_names = list(cat_encoder.get_feature_names_out(["Type"]))
    _, _, eng_num = get_feature_lists()
    all_feature_names = eng_num + cat_names
    importances = clf.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    metadata = {
        "model_name": "Improved Random Forest Classifier",
        "features": MODEL_FEATURES,
        "derived_features": ["Temp_diff", "Power_W", "Strain"],
        "hyperparameters": {
            "n_estimators": 250,
            "max_depth": 12,
            "min_samples_split": 4,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "metrics": {
            "accuracy": float(acc),
            "precision_failure": float(prec_fail),
            "recall_failure": float(rec_fail),
            "f1_failure": float(f1_fail),
            "roc_auc": float(roc_auc),
            "confusion_matrix": {
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1]),
            },
        },
        "baseline_comparison": {
            "baseline_accuracy": 0.9795,
            "baseline_precision": 0.7213,
            "baseline_recall": 0.6471,
            "baseline_f1": 0.6822,
            "baseline_roc_auc": 0.9612,
            "baseline_fn": 24,
            "improved_fn": int(cm[1, 0]),
        },
        "feature_importances": fi_df.to_dict(orient="records"),
    }

    if save_artifacts:
        # Save model
        model_save_path = MODELS_DIR / "best_model.pkl"
        joblib.dump(pipeline, model_save_path)
        print(f"Model saved to: {model_save_path}")

        # Save metadata
        metadata_path = MODELS_DIR / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Metadata saved to: {metadata_path}")

        # Save textual results
        (OUT / "model_results.txt").write_text(results_text, encoding="utf-8")
        (OUT / "improved_model_results.txt").write_text(
            results_text, encoding="utf-8"
        )

        # Plot 1: Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0B132B")
        ax.set_facecolor("#1C2541")
        im = ax.imshow(cm, cmap="Blues")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        ax.set_title(
            "Predictive Maintenance - Confusion Matrix\n(Improved Random Forest)",
            fontsize=12,
            color="white",
            pad=14,
        )
        ax.set_xlabel("Predicted Label", fontsize=11, color="white", labelpad=8)
        ax.set_ylabel("Actual Label", fontsize=11, color="white", labelpad=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Failure (0)", "Failure (1)"], color="white")
        ax.set_yticklabels(["No Failure (0)", "Failure (1)"], color="white")

        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                color = "white" if cm[r, c] > cm.max() / 2 else "cyan"
                ax.text(
                    c,
                    r,
                    f"{cm[r, c]}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=14,
                    weight="bold",
                )

        fig.tight_layout()
        cm_path = OUT / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Confusion matrix plot saved: {cm_path}")

        # Plot 2: Feature Importance
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0B132B")
        ax.set_facecolor("#1C2541")
        colors = [
            "#38BDF8" if "Power" in f or "Strain" in f or "Temp_diff" in f else "#3A86FF"
            for f in fi_df["Feature"][::-1]
        ]
        bars = ax.barh(
            fi_df["Feature"][::-1],
            fi_df["Importance"][::-1],
            color=colors,
            edgecolor="#60A5FA",
            height=0.65,
        )
        ax.set_title(
            "Feature Importances - Improved Model\n(Cyan = Physics-Informed Derived Features)",
            fontsize=12,
            color="white",
            pad=14,
        )
        ax.set_xlabel("Relative Importance Score", fontsize=11, color="white", labelpad=8)
        ax.tick_params(colors="white")
        ax.grid(axis="x", linestyle="--", alpha=0.3, color="#64748B")
        for spine in ax.spines.values():
            spine.set_color("#334155")

        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                va="center",
                color="white",
                fontsize=9,
            )

        fig.tight_layout()
        fi_path = OUT / "feature_importance.png"
        fig.savefig(fi_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Feature importance plot saved: {fi_path}")

        # Plot 3: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0B132B")
        ax.set_facecolor("#1C2541")
        ax.plot(
            fpr,
            tpr,
            color="#38BDF8",
            lw=2.5,
            label=f"Improved RF (AUC = {roc_auc:.4f})",
        )
        ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.2, label="Random Guess (0.50)")
        ax.set_title(
            "ROC Curve - Predictive Maintenance Model",
            fontsize=12,
            color="white",
            pad=14,
        )
        ax.set_xlabel("False Positive Rate", fontsize=11, color="white", labelpad=8)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, color="white", labelpad=8)
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748B")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        legend = ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155")
        plt.setp(legend.get_texts(), color="white")

        fig.tight_layout()
        roc_path = OUT / "roc_curve.png"
        fig.savefig(roc_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"ROC curve plot saved: {roc_path}")

    return pipeline, metadata


def load_saved_model(model_path=MODELS_DIR / "best_model.pkl"):
    """Loads and returns the saved Joblib model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Saved model not found at '{model_path}'. Please run training first."
        )
    return joblib.load(model_path)


if __name__ == "__main__":
    train_improved_model()
