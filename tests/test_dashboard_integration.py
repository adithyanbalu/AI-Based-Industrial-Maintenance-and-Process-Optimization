"""
Integration Tests for Dashboard Views and End-to-End Batch Telemetry Workflow
Adithyan B – Milestone 4 Verification
Infosys Springboard Virtual Internship 7.0
"""

import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation import validate_machine_data, MODEL_FEATURES
from src.prediction import predict_machines, predict_single, InvalidInputError
from dashboard.components import (
    load_benchmark_dataset,
    load_comparison_data,
    get_maintenance_advisor,
    dataset_summary,
)


@pytest.mark.parametrize("view_file", [
    "dashboard/views/overview.py",
    "dashboard/views/prediction.py",
    "dashboard/views/batch.py",
    "dashboard/views/analytics.py",
    "dashboard/views/performance.py",
    "dashboard/views/maintenance.py",
])
def test_dashboard_view_executes_without_exceptions(view_file):
    """Verify that each of the six dashboard view pages compiles and runs cleanly."""
    at = AppTest.from_file(view_file)
    at.run(timeout=30)
    assert len(at.exception) == 0, f"{view_file} raised exceptions: {[e.value for e in at.exception]}"


def test_batch_validation_distinguishes_missing_from_rejected():
    """Verify that missing values are imputable while out-of-range values are strictly rejected."""
    # 1. Missing values: strict rejects, imputation accepts
    df_missing = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": None,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 0.0,
    }])
    strict_valid, strict_errs, _ = validate_machine_data(df_missing, allow_imputation=False)
    imputed_valid, fatal_errs, _ = validate_machine_data(df_missing, allow_imputation=True)
    assert not strict_valid, "Strict validation must reject missing values"
    assert imputed_valid, "Imputation-enabled validation must accept missing values"
    assert len(fatal_errs) == 0, "No fatal errors should exist for missing values when imputation is enabled"

    # 2. Out-of-range values: both strict and imputation reject
    df_oor = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 500.0,  # Range is 270 - 330
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 0.0,
    }])
    strict_valid, strict_errs, _ = validate_machine_data(df_oor, allow_imputation=False)
    imputed_valid, fatal_errs, _ = validate_machine_data(df_oor, allow_imputation=True)
    assert not strict_valid, "Strict validation must reject out-of-range value"
    assert not imputed_valid, "Imputation must never accept or clip out-of-range values"
    assert any("outside the allowed range" in e for e in fatal_errs)

    # 3. Invalid product type: both reject
    df_bad_type = pd.DataFrame([{
        "Type": "Z",  # Allowed: L, M, H
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 0.0,
    }])
    imputed_valid, fatal_errs, _ = validate_machine_data(df_bad_type, allow_imputation=True)
    assert not imputed_valid, "Invalid product type must be rejected even with imputation"


def test_batch_prediction_preserves_extra_columns():
    """Verify that batch prediction retains customer-provided extra columns (e.g. UDI, Plant, Notes)."""
    batch_input = pd.DataFrame([
        {
            "UDI": 101,
            "Product ID": "M_TEST_01",
            "Plant": "Detroit_Spindle_1",
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 15.0,
        },
        {
            "UDI": 102,
            "Product ID": "L_TEST_02",
            "Plant": "Stuttgart_Spindle_3",
            "Type": "L",
            "Air temperature [K]": 302.5,
            "Process temperature [K]": 310.2,
            "Rotational speed [rpm]": 1350.0,
            "Torque [Nm]": 64.0,
            "Tool wear [min]": 220.0,
        },
    ])

    results = predict_machines(batch_input, allow_imputation=True)

    assert len(results) == 2
    # Extra columns must be preserved
    assert "UDI" in results.columns
    assert "Product ID" in results.columns
    assert "Plant" in results.columns
    assert results.loc[0, "Plant"] == "Detroit_Spindle_1"
    assert results.loc[1, "Plant"] == "Stuttgart_Spindle_3"

    # Enriched prediction columns must exist
    for col in ["Prediction", "Prediction_Label", "Status", "Failure_Probability", "Risk_Level", "Diagnostic_Notes"]:
        assert col in results.columns


def test_maintenance_guidance_integration():
    """Verify that the maintenance advisor generates structured and actionable guidance."""
    advisor = get_maintenance_advisor()

    # Normal machine
    normal_readings = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 15.0,
    }
    guidance = advisor(normal_readings)
    assert len(guidance) > 0
    assert any("Nominal" in g.get("category", "") or "Healthy" in g.get("mechanism", "") for g in guidance)

    # Tool wear hazard
    hazard_readings = {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 225.0,  # Severe wear >= 200 min
    }
    hazard_guidance = advisor(hazard_readings)
    assert any("Tool" in g.get("category", "") and "CRITICAL" in g.get("priority", "") for g in hazard_guidance)
