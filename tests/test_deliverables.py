"""
Unit and Integration Tests for Adithyan's Deliverables
AI-Based Industrial Maintenance and Process Optimization System
Infosys Springboard Virtual Internship 7.0
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pytest

DATA_PATH = ROOT / "data" / "Dataset.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
COMPARISON_CSV = ROOT / "outputs" / "model_comparison.csv"

from src.features import IndustrialFeatureEngineer, MODEL_FEATURES

from src.prediction import (
    InvalidInputError,
    load_prediction_model,
    predict_machines,
    predict_single,
    validate_input_dataframe,
)
from dashboard.components import fallback_generate_maintenance_guidance


def test_dataset_integrity():
    """Verify AI4I benchmark dataset structure and quality."""
    assert DATA_PATH.exists(), f"Dataset file missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (10000, 14), f"Expected shape (10000, 14), got {df.shape}"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected missing values"
    assert "Machine failure" in df.columns, "Target column 'Machine failure' not found"
    assert df["Machine failure"].value_counts().to_dict() == {0: 9661, 1: 339}


def test_feature_engineering_transformer():
    """Verify physics-informed domain feature derivation."""
    sample = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }])

    engineer = IndustrialFeatureEngineer()
    res = engineer.transform(sample)

    assert "Temp_diff" in res.columns
    assert "Power_W" in res.columns
    assert "Strain" in res.columns
    assert np.isclose(res["Temp_diff"].iloc[0], 10.0)
    expected_power = 1500 * 40.0 * (2.0 * np.pi / 60.0)
    assert np.isclose(res["Power_W"].iloc[0], expected_power)
    assert np.isclose(res["Strain"].iloc[0], 4000.0)


def test_saved_model_and_metadata():
    """Verify model file exists, unpickles cleanly, and achieves target recall."""
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    model = load_prediction_model(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded model does not have 'predict' method"
    assert hasattr(model, "predict_proba"), "Loaded model does not have 'predict_proba' method"

    assert METADATA_PATH.exists(), f"Metadata missing at {METADATA_PATH}"
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = meta["metrics"]

    assert metrics["accuracy"] >= 0.98, f"Accuracy below 98%: {metrics['accuracy']}"
    assert metrics["recall_failure"] >= 0.80, f"Failure recall below 80%: {metrics['recall_failure']}"
    assert metrics["roc_auc"] >= 0.97, f"ROC-AUC below 0.97: {metrics['roc_auc']}"


def test_model_comparison_artifacts():
    """Verify model comparison output CSV contains all evaluated models."""
    assert COMPARISON_CSV.exists(), f"Comparison CSV missing at {COMPARISON_CSV}"
    comp_df = pd.read_csv(COMPARISON_CSV)
    models_present = comp_df["Model"].tolist()

    assert any("Logistic Regression" in m for m in models_present)
    assert any("Decision Tree" in m for m in models_present)
    assert any("Random Forest" in m for m in models_present)
    assert any("Improved Random Forest" in m for m in models_present)


def test_prediction_single_normal():
    """Verify single prediction on nominal machine parameters."""
    res = predict_single(
        type_="M",
        air_temp=298.1,
        proc_temp=308.6,
        rot_speed=1551.0,
        torque=42.8,
        tool_wear=15.0,
    )

    assert res["Prediction"] == 0
    assert res["Prediction_Label"] == "No Failure"
    assert res["Failure_Probability"] < 0.20
    assert res["Risk_Level"] == "Low Risk"


def test_prediction_single_hazard():
    """Verify single prediction on severe overstrain & wear parameters."""
    res = predict_single(
        type_="L",
        air_temp=302.5,
        proc_temp=310.2,
        rot_speed=1350.0,
        torque=64.0,
        tool_wear=220.0,
    )

    assert res["Prediction"] == 1
    assert res["Prediction_Label"] == "Failure"
    assert res["Failure_Probability"] > 0.80
    assert res["Risk_Level"] == "High Risk / Critical"
    assert "Tool wear" in res["Diagnostic_Notes"] or "TWF" in res["Diagnostic_Notes"]


def test_prediction_batch_dataframe():
    """Verify batch prediction on multiple machine readings."""
    df = pd.read_csv(DATA_PATH).head(20)
    res_df = predict_machines(df)

    assert len(res_df) == 20
    for col in ["Prediction", "Prediction_Label", "Failure_Probability", "Risk_Level", "Diagnostic_Notes"]:
        assert col in res_df.columns
    assert set(res_df["Prediction"].unique()).issubset({0, 1})
    assert set(res_df["Prediction_Label"].unique()).issubset({"No Failure", "Failure"})


def test_prediction_validation_rejections():
    """Verify that malformed or out-of-bounds inputs raise InvalidInputError."""
    # Missing columns
    with pytest.raises(InvalidInputError):
        predict_machines(pd.DataFrame([{"Type": "L", "Torque [Nm]": 40.0}]))

    # Invalid Type category
    with pytest.raises(InvalidInputError):
        predict_single(type_="Z", air_temp=300, proc_temp=310, rot_speed=1500, torque=40, tool_wear=10)

    # Physically impossible air temp (negative Kelvin)
    with pytest.raises(InvalidInputError):
        predict_single(type_="M", air_temp=-50, proc_temp=310, rot_speed=1500, torque=40, tool_wear=10)



def test_maintenance_advisor_rules():
    """Verify deterministic rule generation across physical failure modes."""

    # Tool wear hazard
    tw_guidance = fallback_generate_maintenance_guidance({
        "Type": "L", "Air temperature [K]": 300, "Process temperature [K]": 310,
        "Rotational speed [rpm]": 1500, "Torque [Nm]": 40, "Tool wear [min]": 215
    })
    assert any("Tooling System" in g["category"] for g in tw_guidance)

    # Heat dissipation hazard
    hdf_guidance = fallback_generate_maintenance_guidance({
        "Type": "L", "Air temperature [K]": 303.5, "Process temperature [K]": 311.0,
        "Rotational speed [rpm]": 1300, "Torque [Nm]": 40, "Tool wear [min]": 50
    })
    assert any("Thermal & Cooling" in g["category"] for g in hdf_guidance)

    # Power failure hazard
    pwf_guidance = fallback_generate_maintenance_guidance({
        "Type": "H", "Air temperature [K]": 300, "Process temperature [K]": 310,
        "Rotational speed [rpm]": 2400, "Torque [Nm]": 50, "Tool wear [min]": 50
    })
    assert any("Drive & Electrical" in g["category"] for g in pwf_guidance)

    # Nominal healthy machine
    nom_guidance = fallback_generate_maintenance_guidance({
        "Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 15
    })
    assert any("Nominal" in g["category"] for g in nom_guidance)
