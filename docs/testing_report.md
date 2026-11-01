# Testing Report

## AI-Based Industrial Maintenance and Process Optimization System

### 1. Introduction

This document describes the automated and manual testing activities performed for the AI-Based Industrial Maintenance and Process Optimization System.

The objective is to verify data preprocessing, input validation, model loading, prediction functionality, CSV processing, dashboard functionality, failure analytics, and maintenance recommendations.

### 2. Testing Environment

- Operating System: Windows
- Programming Language: Python 3.14.7
- Testing Framework: Pytest 9.2.0
- Machine Learning Library: Scikit-learn 1.9.1
- Dashboard Framework: Streamlit 1.66.1

### 3. Automated Testing

The complete automated test suite was executed using:

```bash
python -m pytest -v
```

**Overall Result: 32 tests passed.**

| Test File | Tests Passed |
|---|---:|
| test_preprocessing.py | 3 |
| test_validation.py | 9 |
| test_prediction.py | 9 |
| test_integration.py | 2 |
| test_deliverables.py | 9 |
| **Total** | **32** |

### 4. Testing Coverage

#### 4.1 Preprocessing Tests

- Dataset loading
- Preprocessing report generation
- Output visualization generation

Result: 3 tests passed.

#### 4.2 Input Validation Tests

- Valid machine input
- Missing required columns
- Invalid machine types
- Missing values
- Invalid numerical values
- Out-of-range values
- Empty input data
- Missing value imputation
- Out-of-range value handling

Result: 9 tests passed.

Note: These tests exercise `validate_input_dataframe()` in `src/prediction.py`. They do not verify a separate `src/data_validation.py` module.

#### 4.3 Prediction Tests

- Valid machine prediction
- Missing required columns
- Invalid machine type
- Empty input
- Out-of-range temperature
- Risk classification
- Single-machine prediction
- Multiple-machine prediction
- Missing input rejection

Result: 9 tests passed.

#### 4.4 Integration Tests

- CSV dataset records passed through the prediction pipeline.
- Multiple manual machine records passed through the prediction pipeline.
- Prediction output structure and probabilities were checked.

Result: 2 tests passed.

#### 4.5 Existing Deliverables Tests

The existing project test file verifies:

- Dataset integrity
- Feature engineering
- Saved model artifacts
- Model comparison outputs
- Predictions
- Input validation
- Maintenance advisor fallback behaviour

Result: 9 tests passed.

### 5. Manual Dashboard Testing

The Streamlit dashboard was tested using manual machine inputs, CSV uploads, and the available dashboard pages.

#### 5.1 Normal Machine Input

Observed result:

- Status: NORMAL
- Failure Probability: 0.1%
- Risk Level: Low Risk

Result: Passed.

#### 5.2 High-Risk Machine Input

Observed result:

- Failure predicted
- Failure Probability: 96.6%
- Risk Level: High Risk / Critical

Result: Passed.

#### 5.3 CSV Batch Upload

The project dataset was uploaded through the dashboard.

Observed result:

- Total records processed: 10,000
- Normal predictions: 9,633
- Predicted failures: 367
- Average failure probability: 5.64%

Result: Passed.

#### 5.4 Failure Analytics

Observed dataset information:

- Historical records: 10,000
- Actual failure records: 339
- Actual failure rate: 3.39%
- HDF displayed as the dominant failure mode with 115 events.

Result: Passed.

#### 5.5 Incorrect CSV Format

A deliberately malformed CSV file (`invalid_format.csv`) was uploaded.

Observed result:

- Error message: Could not read the file.
- CSV parsing error was displayed.
- Prediction was stopped.

Result: Passed. Invalid CSV format was handled correctly.

#### 5.6 Missing Required CSV Columns

A CSV upload with missing required columns was checked.

Result: Passed, as confirmed during manual verification.

#### 5.7 Model Loading Error

The model loading function was tested using a deliberately nonexistent model path:

`models/nonexistent_test_model.pkl`

Observed result:

- `ModelLoadError` was raised.
- A clear error message indicated that the trained model could not be found.
- The original trained model file was not modified.

Result: Passed. Model loading errors were handled correctly.

#### 5.8 Maintenance Recommendations

The Maintenance Insights dashboard was tested with the following machine readings:

- Machine Type: L
- Tool Wear: 210 min
- Air Temperature: 303.50 K
- Process Temperature: 311.00 K
- Rotational Speed: 1320 rpm
- Torque: 62 Nm

Observed result:

- Maintenance Status: ACTION REQUIRED
- Failure Probability: 97.0%
- Risk Level: High Risk / Critical
- Critical rules triggered: 3
- Tool Wear: Critical
- Temperature: Critical
- Torque and Load: Critical

The dashboard displayed suggested maintenance actions for the detected risk factors.

Result: Passed.

#### 5.9 Model Performance Dashboard

The Model Performance page was inspected.

Observed results for the selected Improved Random Forest model:

- Accuracy: 98.55%
- Precision: 76.00%
- Recall: 83.82%
- F1-score: 0.7972
- ROC-AUC: 0.9823

The algorithm comparison table, metric comparison, ROC curves, and confusion matrix were displayed.

Result: Passed.

### 6. Warnings and Observations

The automated test execution completed with 32 passed tests and 14 warnings.

Scikit-learn generated version compatibility warnings because the saved model contains estimators created using version 1.9.0, while the current environment uses version 1.9.1.

The model owner should align the training and execution versions or regenerate the saved model using the agreed environment.

The separate modules `src/data_validation.py` and `src/machine_monitoring.py` were not present in the inspected source directory. Their individual functionality could not be tested as separate modules.

### 7. Overall Testing Result

- Automated tests: 32 passed.
- Normal machine input: Passed.
- High-risk machine input: Passed.
- CSV batch processing: Passed.
- Invalid CSV format handling: Passed.
- Missing required CSV columns: Passed.
- Model loading error handling: Passed.
- Failure Analytics: Passed.
- Maintenance recommendations: Passed.
- Model Performance dashboard: Passed.
- Core dashboard workflows: Verified.

### 8. Conclusion

The automated test suite completed successfully with all 32 tests passing.

Manual testing verified machine input processing, prediction functionality, batch CSV processing, input error handling, model loading error handling, failure analytics, maintenance recommendations, and model performance visualization.

The tested application workflows behaved as expected during the verification activities.

Further testing should be performed if new modules or functionality are introduced.