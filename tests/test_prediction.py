
import pandas as pd
import pytest

from src.prediction import (
    InvalidInputError,
    assign_risk_level,
    validate_input_dataframe,
    predict_machines,
    predict_single,
)

SAMPLE_INPUT = {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 15,
}


def test_valid_machine_input():
    valid, errors, cleaned = validate_input_dataframe(
        pd.DataFrame([SAMPLE_INPUT])
    )
    assert valid is True
    assert errors == []
    assert len(cleaned) == 1


def test_missing_required_column():
    data = pd.DataFrame([{"Type": "M"}])
    valid, errors, cleaned = validate_input_dataframe(data)
    assert valid is False
    assert any("Missing required feature columns" in e for e in errors)


def test_invalid_machine_type():
    data = SAMPLE_INPUT.copy()
    data["Type"] = "X"
    valid, errors, cleaned = validate_input_dataframe(pd.DataFrame([data]))
    assert valid is False


def test_empty_dataframe():
    valid, errors, cleaned = validate_input_dataframe(pd.DataFrame())
    assert valid is False


def test_out_of_range_temperature():
    data = SAMPLE_INPUT.copy()
    data["Air temperature [K]"] = 500
    valid, errors, cleaned = validate_input_dataframe(pd.DataFrame([data]))
    assert valid is False


def test_risk_level_classification():
    assert assign_risk_level(0.10) == "Low Risk"
    assert assign_risk_level(0.50) == "Medium / Warning"
    assert assign_risk_level(0.80) == "High Risk / Critical"


def test_single_machine_prediction():
    result = predict_single(**SAMPLE_INPUT)
    assert result["Prediction"] in [0, 1]
    assert 0 <= result["Failure_Probability"] <= 1
    assert result["Prediction_Label"] in ["Failure", "No Failure"]


def test_multiple_machine_predictions():
    results = predict_machines([SAMPLE_INPUT, SAMPLE_INPUT])
    assert len(results) == 2
    assert "Prediction" in results.columns
    assert "Risk_Level" in results.columns


def test_missing_input_rejected():
    incomplete = {"Type": "M"}
    with pytest.raises(InvalidInputError):
        predict_machines(incomplete)
