# Testing Report

The current Milestone 4 validation and error-handling test results are documented in [`milestone4_validation_report.md`](milestone4_validation_report.md).

## Latest focused automated verification

Command:

```bash
python -m pytest -q tests/test_validation.py tests/test_prediction.py tests/test_integration.py tests/test_preprocessing.py tests/test_milestone4_validation.py
```

**Result: 48 passed, 7 warnings.** The warnings are scikit-learn model-serialization version warnings: the saved model was created with scikit-learn 1.9.0, while the verification runtime used 1.8.0.

The full test suite was not completed in this environment because Streamlit was unavailable during collection of `tests/test_deliverables.py`. Install all dependencies from `requirements.txt` and rerun `python -m pytest -v` in the target environment.

Refer to the Milestone 4 report for the full scenario matrix, validation policy, limitations, and completion checklist.
