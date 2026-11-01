
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction import predict_machines

DATA_PATH = ROOT / "data" / "Dataset.csv"


def test_csv_to_prediction_integration():
    # Read sample records from the actual dataset
    assert DATA_PATH.exists(), "Dataset.csv is missing"

    input_df = pd.read_csv(DATA_PATH).head(10)

    # Run the prediction pipeline
    result_df = predict_machines(input_df)

    # Verify the prediction output
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 10

    required_columns = [
        "Prediction",
        "Prediction_Label",
        "Status",
        "Failure_Probability",
        "Risk_Level",
        "Diagnostic_Notes",
    ]

    for column in required_columns:
        assert column in result_df.columns

    assert result_df["Prediction"].notna().all()
    assert result_df["Failure_Probability"].between(0, 1).all()
    assert result_df["Risk_Level"].notna().all()


def test_multiple_machine_records_integration():
    machine_data = pd.DataFrame([
        {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 15,
        },
        {
            "Type": "L",
            "Air temperature [K]": 302.5,
            "Process temperature [K]": 310.2,
            "Rotational speed [rpm]": 1350,
            "Torque [Nm]": 64,
            "Tool wear [min]": 220,
        },
    ])

    result_df = predict_machines(machine_data)

    assert len(result_df) == 2
    assert result_df["Prediction"].isin([0, 1]).all()
    assert result_df["Prediction_Label"].isin(
        ["No Failure", "Failure"]
    ).all()
    assert result_df["Failure_Probability"].between(0, 1).all()
