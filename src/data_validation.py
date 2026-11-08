"""CSV schema and row validation compatibility wrapper."""
import pandas as pd
from src.validation import MODEL_FEATURES, validate_machine_data

REQUIRED_COLUMNS = MODEL_FEATURES
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
ALLOWED_TYPES = {"L", "M", "H"}


def validate_csv_columns(df):
    if not isinstance(df, pd.DataFrame):
        return REQUIRED_COLUMNS.copy()
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def validate_records(df, allow_imputation=False):
    valid, errors, cleaned = validate_machine_data(df, allow_imputation=allow_imputation)
    if not valid:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), errors
    return cleaned, []
