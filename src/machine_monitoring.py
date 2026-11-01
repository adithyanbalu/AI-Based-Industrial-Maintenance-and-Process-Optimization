import pandas as pd

from src.data_validation import (
    REQUIRED_COLUMNS,
    validate_csv_columns,
    validate_records,
)

from src.prediction import predict_machines


def monitor_machines(raw_data):
    """
    Validates machine readings and generates failure predictions.

    Returns:
        valid_data: Validated machine readings.
        errors: Validation errors.
        predictions: ML prediction results.
    """

    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if raw_data.empty:
        return pd.DataFrame(), ["No machine records provided."], pd.DataFrame()

    missing = validate_csv_columns(raw_data)

    if missing:
        return (
            pd.DataFrame(),
            ["Missing required columns: " + ", ".join(missing)],
            pd.DataFrame(),
        )

    valid_data, errors = validate_records(raw_data)

    if valid_data.empty:
        return valid_data, errors, pd.DataFrame()

    predictions = predict_machines(valid_data)

    return valid_data, errors, predictions


if __name__ == "__main__":
    sample = pd.DataFrame(
        [[
            "M",
            298.1,
            308.6,
            1551,
            42.8,
            15
        ]],
        columns=REQUIRED_COLUMNS,
    )

    valid, errors, results = monitor_machines(sample)

    print("Validation errors:", errors)
    print("\nValidated machine readings:")
    print(valid)

    print("\nMachine monitoring predictions:")
    print(results)
