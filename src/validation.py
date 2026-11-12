"""Centralised validation for single and batch machine-failure predictions.

All prediction entry points must call this module before model inference.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import numbers

import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
NUMERIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Broad sanity limits for the AI4I-style industrial telemetry used here.
    "Air temperature [K]": (270.0, 330.0),
    "Process temperature [K]": (280.0, 340.0),
    "Rotational speed [rpm]": (500.0, 4000.0),
    "Torque [Nm]": (1.0, 150.0),
    "Tool wear [min]": (0.0, 400.0),
}
VALID_TYPES = {"L", "M", "H"}
DEFAULTS: Dict[str, Any] = {
    "Type": "M", "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1500.0,
    "Torque [Nm]": 40.0, "Tool wear [min]": 0.0,
}


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except (TypeError, ValueError):
        return False


def validate_machine_data(
    data: Any, *, allow_imputation: bool = False
) -> Tuple[bool, List[str], pd.DataFrame]:
    """Return ``(valid, errors, cleaned_df)``; never sends bad rows onward.

    Numeric strings are accepted and converted. Missing values are rejected by
    default; optional batch imputation fills only missing cells. Invalid strings,
    infinities, invalid categories and out-of-range values are always rejected.
    Extra CSV columns are ignored for inference and retained by the caller.
    """
    if isinstance(data, dict):
        frame = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        frame = pd.DataFrame([data.to_dict()])
    elif isinstance(data, list):
        if not data or not all(isinstance(row, dict) for row in data):
            return False, ["Batch input must be a non-empty list of dictionaries."], pd.DataFrame()
        frame = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        return False, ["Input must be a pandas DataFrame, dictionary, Series, or list of dictionaries."], pd.DataFrame()

    if frame.empty:
        return False, ["Input contains zero machine records."], pd.DataFrame()

    missing_columns = [name for name in MODEL_FEATURES if name not in frame.columns]
    if missing_columns:
        return False, ["Missing required feature columns: " + ", ".join(missing_columns)], pd.DataFrame()

    cleaned = frame[MODEL_FEATURES].copy().reset_index(drop=True)
    errors: List[str] = []

    # Product type: never silently replace invalid category values.
    type_values = []
    for idx, value in cleaned["Type"].items():
        if _is_missing(value) or (isinstance(value, str) and not value.strip()):
            if allow_imputation:
                type_values.append(DEFAULTS["Type"])
            else:
                errors.append(f"Row {idx + 1}: Product Type is missing; enter L, M, or H.")
                type_values.append(value)
            continue
        if not isinstance(value, str):
            errors.append(f"Row {idx + 1}: Product Type must be text (L, M, or H).")
            type_values.append(value)
            continue
        value = value.strip().upper()
        if value not in VALID_TYPES:
            errors.append(f"Row {idx + 1}: invalid Product Type '{value}'; allowed values are L, M, H.")
        type_values.append(value)
    cleaned["Type"] = type_values

    for column, (lower, upper) in NUMERIC_BOUNDS.items():
        converted = []
        for idx, value in cleaned[column].items():
            if _is_missing(value) or (isinstance(value, str) and not value.strip()):
                if allow_imputation:
                    converted.append(float(DEFAULTS[column]))
                else:
                    errors.append(f"Row {idx + 1}: {column} is missing.")
                    converted.append(np.nan)
                continue
            if isinstance(value, (bool, np.bool_)):
                errors.append(f"Row {idx + 1}: {column} must be a numeric value, not a boolean.")
                converted.append(np.nan)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                errors.append(f"Row {idx + 1}: {column} must be numeric; received {value!r}.")
                converted.append(np.nan)
                continue
            if not math.isfinite(number):
                errors.append(f"Row {idx + 1}: {column} must be a finite number (not NaN or infinity).")
                converted.append(np.nan)
                continue
            if number < lower or number > upper:
                errors.append(f"Row {idx + 1}: {column}={number:g} is outside the allowed range {lower:g} to {upper:g}.")
            converted.append(number)
        cleaned[column] = converted

    # Ensure no partially valid dataset is returned as inference-ready.
    valid = not errors
    return valid, errors, cleaned if valid else pd.DataFrame()


def validate_single_machine(values: Dict[str, Any]) -> Tuple[bool, List[str], pd.DataFrame]:
    """Validate one machine record using the same rules as batch inference."""
    return validate_machine_data(values, allow_imputation=False)
