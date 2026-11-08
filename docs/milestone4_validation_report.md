# Milestone 4 — Data Validation and Error Handling

## Project
AI-Based Industrial Maintenance and Process Optimization System

## Objective
Ensure that incomplete, malformed, or physically implausible machine telemetry is detected before it reaches the Random Forest prediction model. The same validation rules are used by single-record and batch prediction paths.

## Files delivered or updated

| File | Work completed |
|---|---|
| `src/validation.py` | New central validation module, shared schema, allowed product types, numerical limits, optional missing-cell imputation |
| `src/prediction.py` | Prediction API delegates to central validation; validates threshold and inputs before loading the model; single-input errors are converted to `InvalidInputError` |
| `src/data_validation.py` | Backward-compatible CSV validation wrapper using the central validator |
| `src/features.py` | Defensive validation before deriving temperature difference, power, and strain features |
| `dashboard/views/prediction.py` | User-friendly validation/model/unexpected-error handling |
| `dashboard/views/batch.py` | Batch pre-check, optional missing-cell imputation, rejects invalid text/categories/ranges, and reports errors before scoring |
| `dashboard/components.py` | Validator wrapper supports explicit imputation setting |
| `tests/test_validation.py` | Updated assertions to reflect strict rejection of out-of-range values |
| `tests/test_milestone4_validation.py` | New regression tests for missing, malformed, non-finite, out-of-range, single, batch, and pre-inference validation |
| `src/model_comparison.py` | Fixed existing missing opening module docstring delimiter found during whole-project syntax check |

## Required machine parameters

All six features are mandatory:

1. `Type` — categorical value `L`, `M`, or `H` (case-insensitive; whitespace trimmed).
2. `Air temperature [K]` — numeric, 270 to 330 K.
3. `Process temperature [K]` — numeric, 280 to 340 K.
4. `Rotational speed [rpm]` — numeric, 500 to 4,000 rpm.
5. `Torque [Nm]` — numeric, 1 to 150 Nm.
6. `Tool wear [min]` — numeric, 0 to 400 min.

These are broad configured sanity limits for the project's AI4I-style machine telemetry, not universal operating limits for every industrial machine. They can be adjusted in `src/validation.py` to match a different sensor or machine specification.

## Validation behaviour

- Missing required columns: reject the complete request and identify the missing columns.
- Missing cells: reject by default. Batch scoring may explicitly set `allow_imputation=True`; only missing cells are then filled using documented nominal defaults.
- Numeric strings: accepted and converted to numeric values.
- Invalid strings, booleans, NaN, and positive/negative infinity in numeric fields: reject with row and column details.
- Invalid product types: reject; never silently replace an invalid category.
- Out-of-range numeric values: reject, including when missing-value imputation is enabled. Values are never silently clipped.
- Empty batches and unsupported input structures: reject with a readable validation error.
- Batch policy: if any row is invalid, the batch is rejected as a whole; no partial prediction is run.
- Prediction policy: validation is completed before model loading and before `predict_proba` is called.
- Feature engineering: the transformer validates its input before calculating derived features.

## Test scenarios and results

| Scenario | Expected behaviour | Result |
|---|---|---|
| Valid single machine | Accept and predict | Passed |
| Valid batch | Accept all rows and predict | Passed |
| Missing required column | Reject with column name | Passed |
| Missing numeric value | Reject by default | Passed |
| Explicit missing-value imputation | Fill documented default | Passed |
| Invalid numeric text | Reject with row/field details | Passed |
| Boolean / NaN / infinity | Reject | Passed |
| Numeric strings | Convert and accept when in range | Passed |
| Invalid product type | Reject | Passed |
| Below/above numerical bounds | Reject; no clipping | Passed |
| Invalid batch row | Block model loading and inference | Passed |
| Invalid single input | Raise `InvalidInputError` | Passed |
| Invalid decision threshold | Reject before model loading | Passed |
| Empty batch | Reject cleanly | Passed |

## Automated verification performed in this environment

Command:

```bash
python -m pytest -q tests/test_validation.py tests/test_prediction.py tests/test_integration.py tests/test_preprocessing.py tests/test_milestone4_validation.py
```

**Result: 48 passed.** Seven scikit-learn `InconsistentVersionWarning` warnings were emitted because the saved model was serialized with scikit-learn 1.9.0 while the verification environment has scikit-learn 1.8.0. Prediction tests completed successfully, but the project should be run with the pinned training/runtime dependency version to avoid pickle compatibility risk.

A full `python -m pytest -q` run could not collect `tests/test_deliverables.py` in this environment because Streamlit is not installed (`ModuleNotFoundError: streamlit`). This is an environment limitation, not a passing full-suite result. Install `requirements.txt` before running the full suite and dashboard.

Whole-project syntax check performed:

```bash
python -m py_compile src/*.py dashboard/*.py dashboard/views/*.py tests/test_milestone4_validation.py
```

Result: passed after correcting the existing module docstring delimiter in `src/model_comparison.py`.

## Run instructions

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -v
streamlit run dashboard/app.py
```

For batch uploads, blank cells can be imputed only where the dashboard explicitly enables the batch imputation option. Invalid strings, categories, and values outside the configured limits must be corrected in the CSV and uploaded again.

## Completion checklist

- [x] Missing inputs detected
- [x] Invalid numerical values handled
- [x] Incorrect data types handled
- [x] Numerical ranges validated
- [x] Understandable errors returned
- [x] Single prediction uses shared validation
- [x] Batch prediction uses shared validation
- [x] Invalid inputs blocked before model loading/inference
- [x] Feature engineering protected by validation
- [x] Validation scenarios and test results documented
- [ ] Full dashboard test suite rerun after installing all pinned dependencies in the target environment

## Conclusion
Milestone 4 validation is integrated into the prediction path. Invalid machine telemetry is stopped before inference, and batch missing-value handling is explicit rather than silently clipping invalid data. The focused automated regression suite passed 48 tests in the available environment. The full deliverable should receive a final run in the target environment with the project's dependencies installed.
