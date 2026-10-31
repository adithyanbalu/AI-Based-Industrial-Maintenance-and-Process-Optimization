"""
AI-Based Industrial Maintenance and Process Optimization System
Reusable Machine Failure Prediction Module
Module Owner: Adithyan B (Team Lead)
Infosys Springboard Virtual Internship 7.0
"""

from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd


# Import feature engineer
try:
    from src.features import IndustrialFeatureEngineer, MODEL_FEATURES
except ImportError:
    from features import IndustrialFeatureEngineer, MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.pkl"
DEFAULT_METADATA_PATH = ROOT / "models" / "model_metadata.json"

VALID_TYPES = {"L", "M", "H"}

# Plausible physical operating boundaries for industrial machinery in AI4I benchmark
FEATURE_BOUNDS = {
    "Air temperature [K]": (270.0, 330.0),
    "Process temperature [K]": (280.0, 340.0),
    "Rotational speed [rpm]": (500.0, 4000.0),
    "Torque [Nm]": (1.0, 150.0),
    "Tool wear [min]": (0.0, 400.0),
}


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class InvalidInputError(ValueError):
    """Raised when machine parameters fail validation."""
    pass


@lru_cache(maxsize=1)
def load_prediction_model(model_path: Optional[Union[str, Path]] = None):
    """
    Loads and caches the trained machine learning pipeline.
    Ensures optimal performance in dashboard and batch inference without re-reading disk.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise ModelLoadError(
            f"Trained model not found at '{path}'. "
            "Please train the model first by executing: python src/predictive_model.py"
        )
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise ModelLoadError(f"Failed to load trained model from '{path}': {str(e)}")


def load_model_metadata(metadata_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads model performance metadata and training summary."""
    path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_input_dataframe(
    df: pd.DataFrame,
    allow_imputation: bool = False,
) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validates machine parameters before passing to the ML model:
    - Checks required columns presence
    - Validates data types and casts appropriately
    - Checks for missing/null values (can impute if allow_imputation=True)
    - Validates categorical machine Type (L, M, H)
    - Validates physical numerical boundaries
    """
    errors = []
    if not isinstance(df, pd.DataFrame):
        return False, ["Input must be a pandas DataFrame."], pd.DataFrame()

    if df.empty:
        return False, ["Input DataFrame contains zero records."], pd.DataFrame()

    # Column presence check
    missing_cols = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required feature columns: {missing_cols}")
        return False, errors, pd.DataFrame()

    validated_df = df[MODEL_FEATURES].copy()

    # Check for nulls
    null_counts = validated_df.isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if allow_imputation:
            # Impute standard nominal baselines
            defaults = {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 0.0,
            }
            for col, val in defaults.items():
                if col in validated_df.columns:
                    validated_df[col] = validated_df[col].fillna(val)
        else:
            errors.append(f"Input contains missing/null values: {cols_with_nulls}")

    # Validate Type
    validated_df["Type"] = validated_df["Type"].fillna("M").astype(str).str.strip().str.upper()
    invalid_types = set(validated_df["Type"]) - VALID_TYPES
    if invalid_types:
        if allow_imputation:
            # Default invalid types to most frequent 'L' or 'M'
            validated_df["Type"] = validated_df["Type"].replace(list(invalid_types), "M")
        else:
            errors.append(
                f"Invalid Machine Type(s) detected: {invalid_types}. Must be one of: {VALID_TYPES}"
            )

    # Numerical coercion and bound checks
    for col, (min_v, max_v) in FEATURE_BOUNDS.items():
        try:
            validated_df[col] = pd.to_numeric(validated_df[col], errors="coerce")
        except Exception:
            errors.append(f"Could not parse column '{col}' as numeric.")
            continue

        if validated_df[col].isnull().any():
            if allow_imputation:
                defaults = {
                    "Air temperature [K]": 300.0,
                    "Process temperature [K]": 310.0,
                    "Rotational speed [rpm]": 1500.0,
                    "Torque [Nm]": 40.0,
                    "Tool wear [min]": 0.0,
                }
                validated_df[col] = validated_df[col].fillna(defaults[col])
            else:
                errors.append(f"Column '{col}' contains non-numeric or unparsable values.")

        out_of_bounds = validated_df[
            (validated_df[col] < min_v) | (validated_df[col] > max_v)
        ]
        if not out_of_bounds.empty:
            if allow_imputation:
                validated_df[col] = validated_df[col].clip(min_v, max_v)
            else:
                errors.append(
                    f"Column '{col}' has {len(out_of_bounds)} values outside plausible physical range [{min_v}, {max_v}]."
                )

    is_valid = len(errors) == 0
    return is_valid, errors, validated_df



def generate_diagnostic_explanation(row: pd.Series, prob: float) -> str:
    """
    Generates rule-based diagnostic notes explaining the machine's risk factors.
    Relates physical operating points to potential failure mechanisms.
    """
    notes = []
    air_t = row.get("Air temperature [K]", 300.0)
    proc_t = row.get("Process temperature [K]", 310.0)
    rot_speed = row.get("Rotational speed [rpm]", 1500.0)
    torque = row.get("Torque [Nm]", 40.0)
    tool_wear = row.get("Tool wear [min]", 100.0)

    delta_t = proc_t - air_t
    power_w = rot_speed * torque * (2.0 * np.pi / 60.0)
    strain = torque * tool_wear

    # Tool wear
    if tool_wear >= 200:
        notes.append(f"Severe tool wear ({tool_wear:.0f} min >= 200 min; TWF hazard)")
    elif tool_wear >= 160:
        notes.append(f"Elevated tool wear ({tool_wear:.0f} min; nearing end-of-life)")

    # Thermal dissipation
    if delta_t < 8.6 and rot_speed < 1380:
        notes.append(
            f"Low thermal delta ({delta_t:.1f} K < 8.6 K) at low speed ({rot_speed:.0f} rpm; HDF heat hazard)"
        )
    elif proc_t > 313.0:
        notes.append(f"Elevated process temperature ({proc_t:.1f} K)")

    # Power overload / underload
    if power_w > 9000:
        notes.append(f"High power output ({power_w/1000:.2f} kW > 9.0 kW; PWF overload hazard)")
    elif power_w < 3500 and rot_speed < 1250:
        notes.append(f"Abnormally low mechanical power ({power_w/1000:.2f} kW < 3.5 kW; stall risk)")

    # Overstrain
    if strain > 11000:
        notes.append(f"High torque-wear overstrain ({strain:.0f} > 11000; OSF fracture hazard)")

    # Torque extremes
    if torque > 60:
        notes.append(f"Excessive torque load ({torque:.1f} Nm > 60 Nm)")
    elif torque < 15:
        notes.append(f"Unusually low torque load ({torque:.1f} Nm < 15 Nm)")

    if not notes:
        if prob < 0.30:
            return "Nominal operating envelope: all sensor readings within normal parameters."
        else:
            return "Multi-parameter statistical anomaly detected by machine learning model."

    return " | ".join(notes)


def assign_risk_level(prob: float) -> str:
    """Categorizes failure probability into intuitive operational risk tiers."""
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.65:
        return "Medium / Warning"
    else:
        return "High Risk / Critical"


def predict_machines(
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], pd.Series],
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.50,
    allow_imputation: bool = False,
) -> pd.DataFrame:
    """
    Main prediction interface as specified in the project plan.
    Accepts validated machine input (single or multiple records).
    Returns failure predictions, probabilities, and diagnostic notes.

    Parameters:
    -----------
    input_data : pd.DataFrame, dict, list of dicts, or pd.Series
        Machine parameter records containing MODEL_FEATURES.
    model_path : Path or str, optional
        Path to saved joblib model.
    threshold : float, default 0.50
        Decision threshold for binary failure classification.
    allow_imputation : bool, default False
        Whether to gracefully impute missing values for batch telemetry.

    Returns:
    --------
    pd.DataFrame
        Enriched DataFrame with prediction columns:
        - 'Prediction': 0 or 1
        - 'Prediction_Label': 'No Failure' or 'Failure'
        - 'Status': 'Normal' or 'Failure'
        - 'Failure_Probability': float in [0.0, 1.0]
        - 'Risk_Level': 'Low Risk', 'Medium / Warning', 'High Risk / Critical'
        - 'Diagnostic_Notes': string explaining risk factors
    """
    # 1. Normalize input to DataFrame
    if isinstance(input_data, dict):
        df_raw = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_raw = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.Series):
        df_raw = pd.DataFrame([input_data.to_dict()])
    elif isinstance(input_data, pd.DataFrame):
        df_raw = input_data.copy()
    else:
        raise InvalidInputError(
            f"Unsupported input type '{type(input_data)}'. "
            "Expected pandas DataFrame, dict, or list of dicts."
        )

    # 2. Validate input
    is_valid, error_list, validated_df = validate_input_dataframe(df_raw, allow_imputation=allow_imputation)
    if not is_valid:
        raise InvalidInputError(
            "Machine parameter validation failed:\n" + "\n".join(f"- {e}" for e in error_list)
        )

    # 3. Load model
    model = load_prediction_model(model_path)

    # 4. Generate predictions
    probs = model.predict_proba(validated_df)[:, 1]
    preds = (probs >= threshold).astype(int)

    # 5. Build results DataFrame
    results_df = df_raw.copy()
    results_df["Prediction"] = preds
    results_df["Prediction_Label"] = np.where(preds == 1, "Failure", "No Failure")
    results_df["Status"] = np.where(preds == 1, "Failure", "Normal")
    results_df["Failure_Probability"] = np.round(probs, 4)
    results_df["Risk_Level"] = [assign_risk_level(p) for p in probs]
    results_df["Diagnostic_Notes"] = [
        generate_diagnostic_explanation(row, p)
        for (_, row), p in zip(validated_df.iterrows(), probs)
    ]

    return results_df



def predict_single(
    type_: Optional[str] = None,
    air_temp: Optional[float] = None,
    proc_temp: Optional[float] = None,
    rot_speed: Optional[float] = None,
    torque: Optional[float] = None,
    tool_wear: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to predict a single machine reading.
    Accepts explicit parameters or kwargs with standard column names.
    Returns a dictionary of prediction results.
    """
    data = {
        "Type": kwargs.get("Type", type_),
        "Air temperature [K]": kwargs.get("Air temperature [K]", air_temp),
        "Process temperature [K]": kwargs.get("Process temperature [K]", proc_temp),
        "Rotational speed [rpm]": kwargs.get("Rotational speed [rpm]", rot_speed),
        "Torque [Nm]": kwargs.get("Torque [Nm]", torque),
        "Tool wear [min]": kwargs.get("Tool wear [min]", tool_wear),
    }

    # Ensure all values are present
    missing = [k for k, v in data.items() if v is None]
    if missing:
        raise InvalidInputError(f"Missing required parameters for single prediction: {missing}")

    data["Type"] = str(data["Type"]).strip().upper()
    for col in ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]:
        data[col] = float(data[col])

    res_df = predict_machines(data, model_path=model_path)
    return res_df.iloc[0].to_dict()


if __name__ == "__main__":
    print("Testing src/prediction.py with sample machine inputs...")

    # Test 1: Normal Machine
    normal_case = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 15,
    }
    res_normal = predict_single(**normal_case)
    print("\n--- Test 1: Normal Machine ---")
    print(f"Status: {res_normal['Prediction_Label']} (Prob: {res_normal['Failure_Probability']*100:.1f}%)")
    print(f"Risk:   {res_normal['Risk_Level']}")
    print(f"Notes:  {res_normal['Diagnostic_Notes']}")

    # Test 2: Failure Hazard (High Tool Wear + High Torque)
    failure_case = {
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 64.0,
        "Tool wear [min]": 220,
    }
    res_fail = predict_single(**failure_case)
    print("\n--- Test 2: High Tool Wear & Overstrain Machine ---")
    print(f"Status: {res_fail['Prediction_Label']} (Prob: {res_fail['Failure_Probability']*100:.1f}%)")
    print(f"Risk:   {res_fail['Risk_Level']}")
    print(f"Notes:  {res_fail['Diagnostic_Notes']}")

