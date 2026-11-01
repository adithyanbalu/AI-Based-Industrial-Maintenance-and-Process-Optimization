
import pandas as pd
import pytest

from src.prediction import validate_input_dataframe, InvalidInputError, predict_machines


@pytest.fixture
def valid_machine_data():
    return pd.DataFrame({
        "Type": ["M"],
        "Air temperature [K]": [298.1],
        "Process temperature [K]": [308.6],
        "Rotational speed [rpm]": [1551],
        "Torque [Nm]": [42.8],
        "Tool wear [min]": [15],
    })


def test_valid_machine_input(valid_machine_data):
    is_valid, errors, validated = validate_input_dataframe(valid_machine_data)

    assert is_valid is True
    assert errors == []
    assert len(validated) == 1


def test_missing_required_column(valid_machine_data):
    data = valid_machine_data.drop(columns=["Torque [Nm]"])

    is_valid, errors, validated = validate_input_dataframe(data)

    assert is_valid is False
    assert any("Missing required feature columns" in e for e in errors)


def test_invalid_machine_type(valid_machine_data):
    data = valid_machine_data.copy()
    data.loc[0, "Type"] = "X"

    is_valid, errors, validated = validate_input_dataframe(data)

    assert is_valid is False
    assert any("Invalid Machine Type" in e for e in errors)


def test_missing_values_rejected(valid_machine_data):
    data = valid_machine_data.copy()
    data.loc[0, "Torque [Nm]"] = None

    is_valid, errors, validated = validate_input_dataframe(data)

    assert is_valid is False
    assert any("missing/null values" in e for e in errors)


def test_invalid_numerical_value_rejected(valid_machine_data):
    data = valid_machine_data.copy().astype({
        "Torque [Nm]": "object"
    })
    data.loc[0, "Torque [Nm]"] = "invalid"

    is_valid, errors, validated = validate_input_dataframe(data)

    assert is_valid is False
    assert any("non-numeric" in e for e in errors)


def test_out_of_range_value_rejected(valid_machine_data):
    data = valid_machine_data.copy()
    data.loc[0, "Torque [Nm]"] = 200

    is_valid, errors, validated = validate_input_dataframe(data)

    assert is_valid is False
    assert any("outside plausible physical range" in e for e in errors)


def test_empty_dataframe_rejected():
    data = pd.DataFrame()

    is_valid, errors, validated = validate_input_dataframe(data)

    assert is_valid is False
    assert any("zero records" in e for e in errors)


def test_imputation_enabled_for_missing_values(valid_machine_data):
    data = valid_machine_data.copy()
    data.loc[0, "Torque [Nm]"] = None

    is_valid, errors, validated = validate_input_dataframe(
        data, allow_imputation=True
    )

    assert is_valid is True
    assert errors == []
    assert validated.loc[0, "Torque [Nm]"] == 40.0


def test_imputation_clips_out_of_range_values(valid_machine_data):
    data = valid_machine_data.copy()
    data.loc[0, "Torque [Nm]"] = 200

    is_valid, errors, validated = validate_input_dataframe(
        data, allow_imputation=True
    )

    assert is_valid is True
    assert errors == []
    assert validated.loc[0, "Torque [Nm]"] == 150.0
