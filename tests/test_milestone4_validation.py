"""Milestone 4 regression tests: validation must happen before model inference."""
import numpy as np
import pandas as pd
import pytest

from src.validation import MODEL_FEATURES, validate_machine_data
from src.prediction import InvalidInputError, predict_machines, predict_single

VALID = {
    "Type": "M", "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8, "Tool wear [min]": 15,
}

@pytest.mark.parametrize("bad", [None, "", "abc", True, float("inf"), float("nan")])
def test_invalid_numeric_values_rejected(bad):
    row = dict(VALID, **{"Torque [Nm]": bad})
    ok, errors, cleaned = validate_machine_data(pd.DataFrame([row]))
    assert not ok and errors and cleaned.empty

@pytest.mark.parametrize("column,value", [
    ("Air temperature [K]", 269), ("Process temperature [K]", 341),
    ("Rotational speed [rpm]", 4001), ("Torque [Nm]", 0),
    ("Tool wear [min]", 401),
])
def test_out_of_range_values_rejected_even_when_imputation_enabled(column, value):
    row = dict(VALID); row[column] = value
    ok, errors, _ = validate_machine_data(pd.DataFrame([row]), allow_imputation=True)
    assert not ok and any(column in message for message in errors)

@pytest.mark.parametrize("value", ["X", "", None, 4])
def test_invalid_product_type_rejected(value):
    row = dict(VALID, Type=value)
    ok, errors, _ = validate_machine_data(pd.DataFrame([row]))
    assert not ok and any("Product Type" in message for message in errors)


def test_numeric_strings_are_converted():
    row = dict(VALID, **{"Torque [Nm]": "42.8", "Rotational speed [rpm]": "1551"})
    ok, errors, cleaned = validate_machine_data(pd.DataFrame([row]))
    assert ok and not errors
    assert cleaned["Torque [Nm]"].dtype.kind in "fi"


def test_missing_values_rejected_by_default():
    row = dict(VALID, **{"Torque [Nm]": None})
    ok, errors, _ = validate_machine_data(pd.DataFrame([row]))
    assert not ok and any("missing" in message.lower() for message in errors)


def test_missing_values_can_be_imputed_only_when_explicitly_enabled():
    row = dict(VALID, **{"Torque [Nm]": None})
    ok, errors, cleaned = validate_machine_data(pd.DataFrame([row]), allow_imputation=True)
    assert ok and not errors and cleaned.loc[0, "Torque [Nm]"] == 40.0


def test_missing_column_rejected():
    ok, errors, _ = validate_machine_data(pd.DataFrame([{ "Type": "M" }]))
    assert not ok and "Missing required feature columns" in errors[0]


def test_invalid_batch_row_blocks_all_predictions(monkeypatch):
    def should_not_load(*args, **kwargs):
        pytest.fail("Model must not load when input validation fails")
    monkeypatch.setattr("src.prediction.load_prediction_model", should_not_load)
    rows = [VALID, dict(VALID, **{"Torque [Nm]": "not-a-number"})]
    with pytest.raises(InvalidInputError, match="validation failed"):
        predict_machines(rows)


def test_single_invalid_input_raises_domain_error(monkeypatch):
    monkeypatch.setattr("src.prediction.load_prediction_model", lambda *a, **k: pytest.fail("Model must not load"))
    row = dict(VALID, **{"Torque [Nm]": "bad"})
    with pytest.raises(InvalidInputError, match="must be numeric"):
        predict_single(**row)


def test_empty_batch_rejected():
    with pytest.raises(InvalidInputError):
        predict_machines([])


def test_threshold_validation_precedes_model_load(monkeypatch):
    monkeypatch.setattr("src.prediction.load_prediction_model", lambda *a, **k: pytest.fail("Model must not load"))
    with pytest.raises(InvalidInputError, match="threshold"):
        predict_machines(VALID, threshold=1.5)


def test_csv_validation_wrapper_uses_shared_rules():
    from src.data_validation import validate_records
    good, errors = validate_records(pd.DataFrame([VALID]))
    assert not errors and len(good) == 1
    bad_row = dict(VALID, **{"Torque [Nm]": "bad"})
    rejected, errors = validate_records(pd.DataFrame([bad_row]))
    assert rejected.empty and errors


def test_feature_engineer_rejects_invalid_input():
    from src.features import IndustrialFeatureEngineer
    bad_row = dict(VALID, **{"Torque [Nm]": -5})
    with pytest.raises(ValueError, match="rejected invalid machine input"):
        IndustrialFeatureEngineer().transform(pd.DataFrame([bad_row]))
