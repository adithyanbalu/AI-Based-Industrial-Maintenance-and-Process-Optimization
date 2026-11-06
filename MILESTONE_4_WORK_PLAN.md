# Milestone 4 – Individual Work Plan
## AI-Based Industrial Maintenance and Process Optimization System

---

## 1. Milestone Overview

**Project:** AI-Based Industrial Maintenance and Process Optimization System  
**Internship:** Infosys Springboard Virtual Internship 7.0  
**Milestone:** Milestone 4 – Final Development, Integration, Testing and Documentation

### Objective

The objective of Milestone 4 is to improve the reliability, accuracy, usability, and completeness of the existing project.

The project already includes data preprocessing, failure analysis, feature engineering, machine learning models, prediction functionality, and a Streamlit dashboard.

Milestone 4 will focus on verifying the existing implementation, improving error handling, testing all functionalities, integrating the components, and preparing the project for final demonstration.

**Important:** The existing dashboard and completed modules should be improved and verified, not unnecessarily rebuilt.

---

## 2. Individual Responsibilities

### 2.1 Adithyan B – Dashboard Verification and Final Integration

**Primary responsibility:** Ensure that all project components work together correctly and that the dashboard is ready for the final demonstration.

**Tasks:**

1. Review the existing Streamlit dashboard.
2. Verify all six dashboard sections:
   - Overview
   - Prediction
   - Batch Prediction
   - Failure Analytics
   - Model Performance
   - Maintenance Insights
3. Test the connection between the dashboard and the prediction modules.
4. Verify that predictions are displayed correctly.
5. Check that charts, metrics, tables, and other dashboard components display the correct information.
6. Identify and fix dashboard errors, broken navigation, layout issues, and inconsistent outputs.
7. Integrate the completed work from the other team members.
8. Verify that the project runs correctly from the project root.
9. Prepare the final project execution instructions.
10. Coordinate the final GitHub integration and ensure the main branch contains the verified implementation.

**Files to review or work on:**

- `dashboard/app.py`
- `dashboard/components.py`
- `dashboard/styles.py`
- `dashboard/views/overview.py`
- `dashboard/views/prediction.py`
- `dashboard/views/batch.py`
- `dashboard/views/analytics.py`
- `dashboard/views/performance.py`
- `dashboard/views/maintenance.py`

**Expected deliverables:**

- Fully verified Streamlit dashboard.
- Correct integration of all project modules.
- Fixed dashboard errors and display issues.
- Final execution instructions.
- Integrated and working project on the main branch.

**Completion checklist:**

- [ ] All six dashboard sections work.
- [ ] Single prediction works correctly.
- [ ] Batch prediction works correctly.
- [ ] Dashboard metrics match the underlying model and dataset results.
- [ ] Charts and tables display correctly.
- [ ] No major navigation or runtime errors.
- [ ] All integrated modules work together.
- [ ] Project starts successfully using the documented command.

---

### 2.2 Hriday Kaustubh Machiraju – Data Validation and Error Handling

**Primary responsibility:** Improve the reliability of the input data and prediction process by implementing and verifying validation and error handling.

**Tasks:**

1. Review the existing validation implementation.
2. Verify that all required machine parameters are accepted correctly.
3. Check that missing values are handled properly.
4. Check that invalid data types are detected.
5. Validate numerical input ranges.
6. Handle incorrect or incomplete user inputs.
7. Verify that invalid inputs do not reach the prediction model.
8. Ensure meaningful error messages are displayed instead of unexpected application crashes.
9. Verify validation for both single and batch predictions.
10. Integrate validation with the existing prediction pipeline.

**Files to review or work on:**

- `src/validation.py`
- `src/prediction.py`
- `src/features.py`
- Related dashboard prediction modules.

**Expected deliverables:**

- Verified input validation module.
- Improved error handling.
- Correct validation for single predictions.
- Correct validation for batch predictions.
- Documented validation scenarios and test results.

**Completion checklist:**

- [ ] Missing inputs are detected.
- [ ] Invalid numerical values are handled.
- [ ] Incorrect data types are handled.
- [ ] Invalid machine parameter values are rejected appropriately.
- [ ] Error messages are understandable.
- [ ] Single prediction validation works.
- [ ] Batch prediction validation works.
- [ ] Invalid inputs do not crash the application.

---

### 2.3 Dheepesh N – Machine Learning Model Verification and Evaluation

**Primary responsibility:** Verify the machine learning models, evaluate their performance, and ensure that the selected model is suitable for machine failure prediction.

**Tasks:**

1. Review the existing Random Forest implementation.
2. Review the model comparison implementation.
3. Verify that the training and testing process works correctly.
4. Evaluate the models using appropriate classification metrics.
5. Compare model performance using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
6. Pay particular attention to the failure class because identifying machine failures is the main objective.
7. Review the confusion matrix.
8. Verify feature importance results.
9. Check for inconsistencies between the saved model and reported evaluation results.
10. Ensure the selected model is saved correctly.
11. Verify that the prediction module loads and uses the intended saved model.
12. Document the final verified model results.

**Files to review or work on:**

- `src/predictive_model.py`
- `src/model_comparison.py`
- `src/features.py`
- `src/prediction.py`
- `models/best_model.pkl`
- `models/model_metadata.json`

**Expected deliverables:**

- Verified machine learning model.
- Model comparison results.
- Updated evaluation metrics.
- Confusion matrix and feature importance results.
- Verified saved model.
- Final model evaluation report.

**Completion checklist:**

- [ ] Model training runs successfully.
- [ ] Model evaluation runs successfully.
- [ ] Accuracy, precision, recall, and F1-score are calculated.
- [ ] ROC-AUC is verified where applicable.
- [ ] Confusion matrix is generated.
- [ ] Feature importance is verified.
- [ ] Failure-class performance is reviewed.
- [ ] Saved model loads successfully.
- [ ] Prediction results are consistent with the selected model.
- [ ] Final metrics are documented using actual evaluation results.

---

### 2.4 Ganga Vanaparthi – Testing, Quality Assurance and Documentation

**Primary responsibility:** Verify the complete project through systematic testing and ensure that the project documentation is complete and understandable.

**Tasks:**

1. Review the existing automated test suite.
2. Execute all existing tests.
3. Identify failing or missing test cases.
4. Add test cases for data preprocessing.
5. Add test cases for feature engineering.
6. Add test cases for prediction functionality.
7. Add test cases for input validation and error handling.
8. Test batch prediction with valid and invalid inputs.
9. Verify that the dashboard does not produce unexpected runtime errors during normal use.
10. Perform end-to-end testing of the complete project.
11. Check that the project documentation matches the actual implementation.
12. Update the README with:
    - Project overview
    - Features
    - Technologies used
    - Project structure
    - Installation instructions
    - Execution instructions
    - Model evaluation results
    - Dashboard information
13. Document known limitations and possible future improvements.

**Files to review or work on:**

- `tests/`
- `README.md`
- Existing project documentation.
- Relevant source modules for test coverage.

**Expected deliverables:**

- Updated automated test suite.
- Test execution report.
- End-to-end testing results.
- Updated README.
- Documented project limitations and future scope.

**Completion checklist:**

- [ ] Existing tests are executed.
- [ ] Failing tests are investigated.
- [ ] Preprocessing tests are verified.
- [ ] Feature engineering tests are verified.
- [ ] Prediction tests are verified.
- [ ] Validation tests are verified.
- [ ] Batch prediction tests are verified.
- [ ] End-to-end testing is completed.
- [ ] README is updated.
- [ ] Installation and execution instructions are verified.
- [ ] Known limitations are documented.

---

## 3. Team Integration and Collaboration

Although each member has individual responsibilities, all members must coordinate during integration.

### Integration sequence

1. **Model verification:** Complete model evaluation and confirm the selected saved model.
2. **Validation integration:** Verify input validation and error handling with the prediction pipeline.
3. **Testing:** Test the individual modules and report issues.
4. **Dashboard integration:** Connect and verify the updated modules with the existing dashboard.
5. **End-to-end verification:** Run the complete application and test all major features.
6. **Documentation:** Update the README and final project documentation.
7. **Final review:** Verify the GitHub repository and prepare the final demonstration.

### GitHub collaboration rules

- Work on separate branches when changes may conflict.
- Pull the latest changes before starting new work.
- Avoid directly overwriting another member's changes.
- Commit related changes with meaningful commit messages.
- Test changes before pushing them.
- Merge changes into `main` after reviewing them.
- Run the complete application after integration.

---

## 4. Expected Final Project Deliverables

By the end of Milestone 4, the repository should contain:

| Area | Expected deliverable |
|---|---|
| Machine learning | Verified and evaluated prediction model |
| Model evaluation | Updated metrics, confusion matrix and comparison results |
| Data processing | Verified preprocessing and feature engineering |
| Validation | Reliable input validation and error handling |
| Prediction | Working single and batch prediction |
| Dashboard | Fully verified six-section Streamlit dashboard |
| Testing | Updated automated tests and test results |
| Documentation | Complete and updated README |
| Integration | All modules working together |
| Final demonstration | Working application ready for presentation |

---

## 5. Final Acceptance Criteria

The project will be considered ready for final submission when:

- [ ] The dataset is processed correctly.
- [ ] The machine learning model trains and evaluates successfully.
- [ ] The selected model is saved and loaded correctly.
- [ ] Model evaluation metrics are verified.
- [ ] Single machine failure prediction works.
- [ ] Batch machine failure prediction works.
- [ ] Input validation and error handling work correctly.
- [ ] Failure analytics produce correct results.
- [ ] Maintenance insights are displayed correctly.
- [ ] All six dashboard sections work.
- [ ] Automated tests pass, or any remaining failures are documented and resolved where possible.
- [ ] End-to-end testing is completed.
- [ ] README and execution instructions are updated.
- [ ] All changes are integrated into the main branch.
- [ ] The application runs successfully from a clean project setup.

---

## 6. Project Execution Command

Run the following command from the project root directory:

```powershell
python -m streamlit run dashboard/app.py
```

---

## 7. Important Project Scope

This project uses the AI4I Predictive Maintenance Dataset to develop and demonstrate machine failure prediction.

The system performs dataset-based analysis and prediction. It does not currently connect to physical industrial machines, live factory sensors, or IoT devices.

Any future real-time monitoring or hardware integration should be described as future scope unless it is actually implemented and tested.

---

## 8. Final Goal

The goal of Milestone 4 is to bring all existing project components together into a reliable, tested, documented, and demonstrable system.

Every team member must complete their assigned technical deliverables and support the final integration and verification process.

**Milestone 4 is complete only when the integrated application has been tested successfully and the final deliverables are available in the GitHub repository.**
