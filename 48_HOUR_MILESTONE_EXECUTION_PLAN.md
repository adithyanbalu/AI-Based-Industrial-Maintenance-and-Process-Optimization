
# 48-Hour Collaborative Development Plan
## AI-Based Industrial Maintenance and Process Optimization System

**Internship:** Infosys Springboard Virtual Internship 7.0

**Team Members:**
1. Adithyan B 
2. Hriday Kaustubh Machiraju
3. Dheepesh N
4. Ganga Vanaparthi

**Development Duration:** 48 Hours

**Internal Target:** Approximately 75% of planned project functionality.

---

# 1. PROJECT OVERVIEW

The objective of this project is to develop an AI-based predictive maintenance and process optimization system that analyzes machine operating conditions and predicts potential machine failures.

The system uses machine operating parameters and machine learning techniques to identify possible failures and provide useful maintenance guidance.

Our current implementation uses the AI4I Predictive Maintenance Dataset.

### Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Pytest
- Git and GitHub

---

# 2. CURRENT PROJECT STATUS – MILESTONE 1

The following activities have already been completed:

- Dataset loading and inspection.
- Data preprocessing and quality analysis.
- Missing-value analysis.
- Duplicate-record analysis.
- Machine failure distribution analysis.
- Failure percentage calculations.
- Data visualization.
- Initial Random Forest classification model.
- Model performance evaluation.

### Dataset Information

| Description | Value |
|---|---:|
| Total Records | 10,000 |
| Total Columns | 14 |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Normal Records | 9,661 |
| Failure Records | 339 |
| Normal Percentage | 96.61% |
| Failure Percentage | 3.39% |

### Initial Machine Learning Model

Algorithm: Random Forest Classifier.

Initial evaluation:

- Accuracy: Approximately 97.95%.
- Failure Precision: Approximately 72.13%.
- Failure Recall: Approximately 64.71%.
- Failure F1-score: Approximately 0.68.
- ROC-AUC: Approximately 0.9612.

Further improvements will focus on detecting actual failures rather than relying on accuracy alone.

---

# 3. 48-HOUR DEVELOPMENT OBJECTIVE

Our internal target is to complete approximately **75% of the planned project functionality within 48 hours**.

Since Milestone 3 is approaching and everyone will be occupied with semester examinations, we must prioritize:

1. Machine learning improvement.
2. Functional Streamlit dashboard.
3. CSV upload functionality.
4. Manual machine input.
5. Data validation.
6. Machine failure analytics.
7. Maintenance recommendations.
8. Integration of all modules.
9. End-to-end testing.
10. Documentation and presentation.

The 75% target is our internal development goal, not an official Infosys evaluation percentage.

All four members will work collaboratively and develop their assigned modules in parallel.

---

# 4. TEAM TASK DIVISION

## MEMBER 1 – ADITHYAN B 

### Main Responsibility: Machine Learning Improvement and Streamlit Dashboard

### A. Machine Learning Improvement

- Compare different machine learning algorithms.
- Implement Logistic Regression.
- Implement Decision Tree Classifier.
- Implement Random Forest Classifier.
- Evaluate models using suitable performance metrics.
- Improve the existing Random Forest model.
- Investigate class imbalance handling techniques.
- Perform hyperparameter tuning where time permits.
- Analyze false positives and false negatives.
- Generate feature importance visualizations.
- Select the most suitable model based on evaluation results.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Special attention will be given to failure recall because missing an actual machine failure is an important concern in predictive maintenance.

### B. Prediction Module

Develop a reusable prediction function.

```python
def predict_machines(input_df):
    """
    Accepts validated machine parameters.
    Returns machine failure predictions.
    """
```

Requirements:

- Load the trained ML model.
- Accept validated machine operating parameters.
- Predict machine failure status.
- Return prediction results.
- Return model probability when available and properly calibrated.

Save the selected model using Joblib.

### C. Streamlit Dashboard

Develop an interactive dashboard containing:

1. Project Overview.
2. Machine Input.
3. CSV Upload.
4. Machine Failure Prediction.
5. Failure Analytics.
6. Model Performance.
7. Maintenance Recommendations.

### Expected Deliverables

- Model comparison results.
- Improved ML model.
- Saved model file.
- Reusable prediction module.
- Streamlit dashboard.
- Dashboard screenshots.
- Integrated application.

---

## MEMBER 2 – HRIDAY KAUSTUBH MACHIRAJU

### Main Responsibility: Data Input, Validation and Machine Monitoring

### A. CSV Upload

- Implement CSV file upload functionality.
- Read uploaded machine data using Pandas.
- Verify required columns.
- Handle incorrect file formats.
- Prepare uploaded records for prediction.

### B. Manual Machine Input

Allow users to enter:

- Product Type.
- Air Temperature [K].
- Process Temperature [K].
- Rotational Speed [rpm].
- Torque [Nm].
- Tool Wear [min].

### C. Data Validation

- Check missing values.
- Validate numerical inputs.
- Check data types.
- Detect invalid inputs.
- Check required columns.
- Display useful error messages.

### D. Machine Monitoring

- Display machine readings in a structured table.
- Display uploaded records.
- Show validated machine parameters.
- Prepare input data in the format required by the prediction module.

### Expected Deliverables

- CSV upload functionality.
- Manual machine input interface.
- Data validation module.
- Machine monitoring table.
- Validated input DataFrame.
- Screenshots.

---

## MEMBER 3 – DHEEPESH N

### Main Responsibility: Failure Analytics and Maintenance Recommendations

### A. Failure Analytics

Analyze machine failure patterns using the available dataset.

Study the relationship between:

- Air Temperature.
- Process Temperature.
- Rotational Speed.
- Torque.
- Tool Wear.

Generate meaningful charts and visualizations.

### B. Sensor Analysis

- Analyze temperature variations.
- Study rotational speed patterns.
- Analyze torque variations.
- Study tool wear patterns.
- Identify unusual operating conditions.
- Generate charts for important machine parameters.

### C. Maintenance Recommendation Module

Develop a transparent rule-based maintenance guidance module.

Example recommendations:

- If tool wear exceeds a documented threshold, suggest tool inspection.
- If temperature readings indicate unusual conditions, suggest checking the cooling system.
- If torque and rotational speed indicate unusual operating conditions, suggest further machine inspection.

Recommendations must be based on documented rules.

They should not be presented as confirmed failure diagnoses.

### D. Integration

- Prepare analytics modules for dashboard integration.
- Coordinate with Adithyan regarding dashboard integration.
- Ensure that recommendations can use the agreed machine input format.

### Expected Deliverables

- Failure analytics module.
- Sensor analysis charts.
- Machine parameter analysis.
- Rule-based maintenance recommendation module.
- Documentation of maintenance rules.
- Screenshots.

---

## MEMBER 4 – GANGA VANAPARTHI

### Main Responsibility: Testing, Integration Support and Documentation

### A. Testing

Test the following:

- Data preprocessing.
- CSV upload.
- Manual machine input.
- Data validation.
- Model loading.
- Prediction functionality.
- Failure analytics.
- Maintenance recommendations.
- Dashboard functionality.

Test different situations:

- Valid inputs.
- Missing inputs.
- Invalid numerical values.
- Incorrect CSV formats.
- Missing required columns.
- Multiple machine records.
- Model loading errors.

### B. Integration Support

- Coordinate with all members.
- Check module compatibility.
- Identify integration errors.
- Report bugs.
- Verify bug fixes.
- Run end-to-end tests.

### C. Documentation

Prepare and update:

- README.md.
- requirements.txt.
- Installation instructions.
- Implementation documentation.
- Testing report.
- Project usage instructions.

### D. Evidence Collection

Collect actual screenshots of:

- Dashboard overview.
- Machine input.
- CSV upload.
- Prediction results.
- Failure analytics.
- Model performance.
- Maintenance recommendations.
- Testing results.

### Expected Deliverables

- Test cases.
- Test execution report.
- Updated README.
- Requirements file.
- Installation instructions.
- Implementation documentation.
- Screenshots and evidence.

---

# 5. PROJECT STRUCTURE AND FILE OWNERSHIP

The following is the proposed target structure.

Existing Milestone 1 files must be preserved. New files should be created inside the appropriate directories.

```text
AI-Based-Industrial-Maintenance/
│
├── data/
│   └── Dataset.csv
│
├── src/
│   ├── data_preprocessing.py       # EXISTING
│   ├── failure_analysis.py         # EXISTING
│   ├── predictive_model.py        # EXISTING
│   │
│   ├── model_comparison.py        # ADITHYAN
│   ├── prediction.py              # ADITHYAN
│   │
│   ├── data_validation.py         # HRIDAY
│   ├── machine_monitoring.py      # HRIDAY
│   │
│   ├── failure_diagnostics.py     # DHEEPESH
│   └── maintenance_advisor.py     # DHEEPESH
│
├── dashboard/
│   ├── app.py                      # ADITHYAN
│   ├── components.py               # ADITHYAN
│   └── styles.py                   # ADITHYAN
│
├── models/
│   └── best_model.pkl              # GENERATED BY ADITHYAN
│
├── outputs/
│   ├── charts/
│   └── reports/
│
├── tests/
│   ├── test_preprocessing.py       # GANGA
│   ├── test_validation.py          # GANGA
│   ├── test_prediction.py          # GANGA
│   └── test_integration.py         # GANGA
│
├── docs/
│   ├── implementation.md           # GANGA
│   └── testing_report.md           # GANGA
│
├── requirements.txt                # GANGA
├── README.md                       # GANGA
└── .gitignore
```

### File Ownership Summary

| File | Responsible Member | Purpose |
|---|---|---|
| data_preprocessing.py | Existing / Adithyan | Dataset preprocessing |
| failure_analysis.py | Dheepesh | Failure distribution analysis |
| predictive_model.py | Adithyan | Existing ML model |
| model_comparison.py | Adithyan | Compare ML algorithms |
| prediction.py | Adithyan | Reusable prediction function |
| data_validation.py | Hriday | Validate machine inputs |
| machine_monitoring.py | Hriday | Display machine readings |
| failure_diagnostics.py | Dheepesh | Failure and sensor analysis |
| maintenance_advisor.py | Dheepesh | Maintenance suggestions |
| dashboard/app.py | Adithyan | Main Streamlit application |
| dashboard/components.py | Adithyan | Dashboard components |
| dashboard/styles.py | Adithyan | Dashboard styling |
| models/best_model.pkl | Adithyan | Selected trained model |
| tests/test_preprocessing.py | Ganga | Preprocessing tests |
| tests/test_validation.py | Ganga | Validation tests |
| tests/test_prediction.py | Ganga | Prediction tests |
| tests/test_integration.py | Ganga | Integration tests |
| docs/implementation.md | Ganga | Implementation documentation |
| docs/testing_report.md | Ganga | Testing report |
| requirements.txt | Ganga | Project dependencies |
| README.md | Ganga | Setup and usage instructions |

### Existing Files – Modification Responsibilities

| Existing File | Responsible Member | Action |
|---|---|---|
| data_preprocessing.py | Adithyan + Ganga | Maintain and test |
| failure_analysis.py | Dheepesh | Extend existing analysis |
| predictive_model.py | Adithyan | Improve existing model |

**Important:** Do not create duplicate modules or rename existing files unnecessarily. Check imports and dependencies before changing the project structure.

---

# 6. COMMON INTEGRATION INTERFACE

All members must use the same six model input features.

```python
MODEL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
```

### Hriday – Validation Interface

```python
def validate_machine_data(input_df):
    """
    Validates machine input.
    Returns validation status and error messages.
    """
```

### Adithyan – Prediction Interface

```python
def predict_machines(input_df):
    """
    Accepts validated machine input.
    Returns failure predictions.
    """
```

### Dheepesh – Maintenance Interface

```python
def generate_maintenance_guidance(machine_readings):
    """
    Returns rule-based maintenance suggestions.
    """
```

### Ganga – Testing

- Test each module independently.
- Verify input and output formats.
- Test communication between modules.
- Perform end-to-end testing.

The final function signatures and return formats must be agreed upon immediately.

---

# 7. APPLICATION INTEGRATION WORKFLOW

The planned application workflow:

1. User opens the Streamlit dashboard.
2. User enters machine readings manually or uploads a CSV file.
3. The validation module checks the input.
4. Validated readings are passed to the prediction module.
5. The ML model predicts machine failure status.
6. The dashboard displays prediction results.
7. The analytics module displays relevant machine information.
8. The maintenance module provides rule-based inspection suggestions.
9. The user views the final results through the dashboard.

### End-to-End Demonstration

```text
CSV UPLOAD / MANUAL INPUT
          |
          v
    DATA VALIDATION
          |
          v
   MACHINE PARAMETERS
          |
          v
  MACHINE FAILURE MODEL
          |
          v
   PREDICTION RESULTS
          |
          v
    FAILURE ANALYTICS
          |
          v
 MAINTENANCE GUIDANCE
          |
          v
   STREAMLIT DASHBOARD
```

Both CSV input and manual input must use the same validation and prediction pipeline.

---

# 8. 48-HOUR EXECUTION TIMELINE

## FIRST 24 HOURS – DEVELOPMENT

### Phase 1: Coordination and Setup

- Confirm responsibilities.
- Review the existing GitHub repository.
- Agree on common input and output formats.
- Confirm dependencies.
- Create feature branches.
- Start parallel development.

### Phase 2: Individual Development

**Adithyan**
- ML model comparison.
- Model improvement.
- Prediction function.
- Dashboard skeleton.

**Hriday**
- CSV upload.
- Manual machine input.
- Data validation.

**Dheepesh**
- Failure analytics.
- Sensor visualizations.
- Maintenance rules.

**Ganga**
- Test case preparation.
- Requirements file.
- Documentation.
- Initial module testing.

### Phase 3: First Integration

- Connect validated inputs to prediction.
- Test prediction function.
- Integrate analytics.
- Check dashboard compatibility.

## NEXT 24 HOURS – INTEGRATION AND COMPLETION

### Phase 4: Application Integration

- Connect CSV upload to validation.
- Connect manual input to validation.
- Connect validation to prediction.
- Display prediction results.
- Integrate failure analytics.
- Integrate maintenance guidance.

### Phase 5: Testing

- Test valid machine readings.
- Test invalid machine readings.
- Test missing columns.
- Test CSV upload.
- Test multiple records.
- Test model loading.
- Run the complete application.

### Phase 6: Presentation Preparation

- Capture actual screenshots.
- Prepare implementation slides.
- Update README.
- Update GitHub.
- Prepare testing evidence.
- Rehearse the demonstration.

---

# 9. INTERNAL 75% COMPLETION TARGET

The following features should be implemented, tested and demonstrated.

| Component | Target |
|---|---|
| Dataset preprocessing | Completed |
| Failure analysis | Completed and extended |
| ML model comparison | Implemented |
| Improved ML model | Evaluated |
| Prediction module | Working |
| CSV upload | Working |
| Manual machine input | Working |
| Data validation | Working |
| Machine monitoring | Working |
| Failure analytics | Working |
| Maintenance guidance | Working |
| Streamlit dashboard | Working |
| Core module integration | Working |
| Basic testing | Completed |
| Documentation | Updated |
| PPT | Prepared |

A feature counts as completed only when it is implemented and tested. Creating an empty file or adding a screenshot alone does not count as implementation.

This checklist is an internal target, not a measured completion percentage.

---

# 10. LOWER-PRIORITY FEATURES

The following activities can be considered after the core prototype is working:

- Real-time IoT sensor integration.
- Physical industrial hardware connectivity.
- Automatic machine shutdown.
- Advanced predictive maintenance scheduling.
- Cloud deployment.
- Advanced dashboard animations.
- Further model optimization.
- Extensive production-level testing.

Prioritize these according to the official project requirements and remaining timeline.

---

# 11. GITHUB COLLABORATION GUIDELINES

Suggested feature branches:

```text
feature/model-improvement
feature/streamlit-dashboard
feature/machine-monitoring
feature/failure-analytics
feature/testing-documentation
```

### Guidelines

- Each member owns the files assigned above.
- Create files inside the correct directories.
- Push changes regularly.
- Use meaningful commit messages.
- Avoid modifying another member's files without discussion.
- Inform the team before modifying shared modules.
- Review changes before merging.
- Test imports and dependencies before merging.
- Keep the main branch stable.

### Suggested Commit Messages

```text
Add model comparison
Implement CSV upload
Add machine input validation
Implement failure analytics
Add maintenance recommendations
Integrate Streamlit dashboard
Add application tests
Update project documentation
```

### Integration Rule

Do not wait until the final hours to merge all modules.

Push small working changes frequently and integrate continuously.

---

# 12. FINAL TEAM CHECKLIST

## Adithyan

- [ ] Compare ML algorithms.
- [ ] Improve the existing ML model.
- [ ] Evaluate failure recall and false negatives.
- [ ] Generate feature importance visualization.
- [ ] Save the trained model.
- [ ] Develop prediction module.
- [ ] Develop Streamlit dashboard.
- [ ] Integrate all modules.

## Hriday

- [ ] Implement CSV upload.
- [ ] Implement manual machine input.
- [ ] Implement data validation.
- [ ] Implement machine monitoring.
- [ ] Test valid and invalid inputs.

## Dheepesh

- [ ] Implement failure analytics.
- [ ] Analyze machine parameters.
- [ ] Generate sensor analysis charts.
- [ ] Implement maintenance guidance.
- [ ] Document maintenance rules.

## Ganga

- [ ] Prepare test cases.
- [ ] Test individual modules.
- [ ] Test integrated application.
- [ ] Report and verify bug fixes.
- [ ] Update README.
- [ ] Update requirements.txt.
- [ ] Prepare testing report.
- [ ] Collect screenshots.

## Entire Team

- [ ] Complete core module integration.
- [ ] Run Streamlit application successfully.
- [ ] Demonstrate CSV-to-prediction workflow.
- [ ] Demonstrate manual-input-to-prediction workflow.
- [ ] Verify analytics and maintenance guidance.
- [ ] Complete end-to-end testing.
- [ ] Update GitHub repository.
- [ ] Prepare PPT.
- [ ] Complete final rehearsal.

---

# 13. IMPORTANT PROJECT LIMITATIONS

The current project uses the AI4I Predictive Maintenance benchmark dataset.

We must clearly distinguish between:

1. Historical dataset analysis.
2. Predictions generated by the trained ML model.
3. Rule-based maintenance suggestions.

We should not claim real-time industrial sensor connectivity, physical IoT integration, automatic machine shutdown, or production deployment unless these features are actually implemented and tested.

The ML model predicts machine failure status. It does not automatically identify the exact physical cause of every failure.

---

# 14. FINAL TEAM GOAL

Our goal is to complete approximately 75% of the planned project functionality within the next 48 hours.

We will achieve this by:

1. Dividing responsibilities clearly.
2. Developing modules in parallel.
3. Following common integration interfaces.
4. Communicating regularly.
5. Testing each module immediately.
6. Integrating the complete application.
7. Preparing the PPT using actual implementation evidence.

**TEAM OBJECTIVE:**

Complete the major development and integration activities before semester examinations so that everyone can focus on academic commitments without unnecessary project pressure.

Let's work collaboratively, avoid unnecessary delays, and complete the implementation together.
