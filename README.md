# 🏭 AI-Based Industrial Maintenance and Process Optimization System

<p align="center">
  <b>Predictive Maintenance Command Center & Telemetry Intelligence</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Internship-Infosys%20Springboard%207.0-blue?style=for-the-badge" alt="Infosys Springboard Internship 7.0"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Domain-Predictive%20Maintenance-green?style=for-the-badge" alt="Predictive Maintenance"/>
  <img src="https://img.shields.io/badge/Model-Random%20Forest%20v2.1-cyan?style=for-the-badge" alt="Random Forest"/>
  <img src="https://img.shields.io/badge/Tests-60%20Passed-brightgreen?style=for-the-badge" alt="60 Tests Passed"/>
</p>

---

## 📌 1. Project Overview

Industrial machinery failures can cause unexpected plant downtime, increased maintenance costs, equipment damage, and production disruptions.

The **AI-Based Industrial Maintenance and Process Optimization System** is a machine learning-based predictive maintenance application developed using the **AI4I 2020 Predictive Maintenance Dataset**, containing 10,000 machine records.

The system analyses machine operating conditions and predicts potential machine failures using machine learning techniques.

It integrates:

- Dataset analysis and preprocessing.
- Data validation and cleaning.
- Feature engineering.
- Machine learning model comparison.
- Improved Random Forest classification.
- Individual machine failure prediction.
- Batch CSV prediction.
- Failure analytics and visualization.
- Maintenance insights.
- Interactive Streamlit dashboard.
- Automated testing and quality assurance.

The main objective is to identify potential failures early and support preventive maintenance decisions.

---

## 🎯 2. Project Objectives

- Analyse industrial machine operating conditions.
- Identify patterns associated with machine failures.
- Develop a machine learning-based failure prediction model.
- Improve failure detection performance.
- Reduce false negatives in machine failure prediction.
- Provide individual and batch prediction functionality.
- Visualize machine failure statistics.
- Provide maintenance-related insights.
- Validate input data and handle invalid files.
- Build an interactive predictive maintenance dashboard.

---

## 👥 3. Project Team & Responsibilities

Developed as part of **Infosys Springboard Virtual Internship 7.0 – Artificial Intelligence Track**.

| Team Member | Responsibilities |
|---|---|
| Adithyan B (Team Lead) | ML Model Comparison, Predictive Model Improvement, Joblib Serialization, Reusable Prediction Module, Streamlit Dashboard Architecture, and Integration |
| Hriday Kaustubh Machiraju | Data Validation and Machine Monitoring |
| Dheepesh N | Failure Diagnostics and Sensor Analysis |
| Ganga Vanaparthi | Testing, Integration Verification, Quality Assurance, and Documentation |

---

# 🚀 4. Project Development – Milestone 1 to Milestone 4

## Milestone 1 – Dataset Analysis and Initial Model Development

### Objective

Understand the predictive maintenance dataset, identify machine operating parameters, analyse failure distribution, and establish an initial machine learning baseline.

### Dataset Used

**AI4I 2020 Predictive Maintenance Dataset**

- Total records: 10,000
- Total attributes: 14
- Normal machine records: 9,661
- Machine failure records: 339
- Missing values: 0
- Duplicate records: 0

### Dataset Features

| Feature | Description |
|---|---|
| UDI | Unique data identifier |
| Product ID | Machine product identifier |
| Type | Product quality type |
| Air temperature [K] | Air temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Machine torque |
| Tool wear [min] | Tool wear time |
| Machine failure | Target variable |
| TWF | Tool Wear Failure |
| HDF | Heat Dissipation Failure |
| PWF | Power Failure |
| OSF | Overstrain Failure |
| RNF | Random Failure |

### Initial Data Analysis

The dataset was analysed to understand:

- Machine failure distribution.
- Operating temperature.
- Rotational speed.
- Torque.
- Tool wear.
- Failure-related patterns.

The initial dataset validation confirmed 10,000 records and 14 attributes, with no missing values or duplicate records.

### Initial Machine Learning Baseline

A baseline Random Forest model was evaluated.

| Evaluation Metric | Baseline Result |
|---|---:|
| Accuracy | 97.95% |
| Precision | 72.13% |
| Recall | 64.71% |
| F1-Score | 0.6822 |
| ROC-AUC | 0.9612 |
| False Negatives | 24 |
| False Positives | 17 |

**Milestone 1 Outcome:** Dataset analysis, data understanding, preprocessing foundations, and initial model evaluation.

---

## Milestone 2 – Data Processing and System Development

### Objective

Develop the data processing and analytical components required for the predictive maintenance system.

### Development Areas

- Dataset loading and processing.
- Data validation.
- Handling machine operating parameters.
- Failure analysis.
- Feature preparation.
- Machine monitoring functionality.
- Preparation of reusable Python modules.

### Main Development Components

| Component | Purpose |
|---|---|
| Data Preprocessing | Loads and prepares machine data |
| Data Validation | Checks required input fields and data quality |
| Feature Engineering | Creates additional machine-related features |
| Failure Analysis | Analyses failure patterns and statistics |
| Machine Monitoring | Supports machine condition analysis |

### Dataset Quality Verification

The preprocessing workflow generated a report and visualization.

Verified dataset results:

| Check | Result |
|---|---:|
| Dataset records | 10,000 |
| Dataset columns | 14 |
| Missing values | 0 |
| Duplicate records | 0 |

**Milestone 2 Outcome:** Development of the data processing, validation, and analytical components used by the predictive maintenance application.

---

## Milestone 3 – Machine Learning Improvement and Dashboard Integration

### Objective

Develop and evaluate improved machine learning models and integrate prediction functionality into an interactive dashboard.

### Model Development

Multiple machine learning algorithms were compared to identify a suitable model for machine failure prediction.

The candidate models included:

- Logistic Regression.
- Balanced Logistic Regression.
- Decision Tree.
- Balanced Decision Tree.
- Random Forest.
- Improved Random Forest.

### Feature Engineering

Machine operating parameters are used to derive additional information for machine condition analysis.

The system incorporates physics-informed features to support failure prediction and machine monitoring.

Failure-cause indicator flags (`TWF`, `HDF`, `PWF`, `OSF`, and `RNF`) are excluded from model input features to avoid target leakage.

### Selected Model

**Improved Random Forest Classifier**

The selected model achieved:

- Accuracy: 98.55%
- Precision: 76.00%
- Recall: 83.82%
- F1-Score: 0.7972
- ROC-AUC: 0.9823
- False Negatives: 11
- False Positives: 18

### Dashboard Integration

The Streamlit application provides the following major sections:

1. Command Center.
2. Failure Prediction.
3. Batch CSV Analysis.
4. Failure Analytics.
5. Model Performance.
6. Maintenance Insights.

### Prediction Functionality

The application supports:

- Individual machine parameter inputs.
- Failure probability estimation.
- Normal and failure risk classification.
- Batch CSV processing.
- Machine failure analytics.
- Maintenance-related recommendations.

**Milestone 3 Outcome:** Machine learning model comparison, improved prediction functionality, and integration of the predictive maintenance dashboard.

---

## Milestone 4 – Testing, Quality Assurance and Documentation

### Objective

Verify application functionality, identify issues, execute automated tests, perform dashboard testing, and update project documentation.

### Assigned Responsibility

**Ganga Vanaparthi – Testing, Quality Assurance and Documentation**

### Activities Performed

- Reviewed the existing automated test suite.
- Executed all automated tests.
- Reviewed feature engineering test coverage.
- Verified dataset dimensions.
- Checked missing values and duplicate records.
- Tested prediction functionality.
- Tested batch CSV prediction.
- Checked CSV input validation.
- Tested validation and error-handling scenarios.
- Tested dashboard modules.
- Performed end-to-end workflow verification.
- Recorded test execution results.
- Updated project README documentation.
- Documented project limitations and future improvements.

### Automated Testing

Testing was performed using **Pytest**.

Command:

```bash
python -m pytest tests/ -v
```

**Recorded Result: 69 passed, 0 failed.**

| Test File | Coverage |
|---|---|
| test_dashboard_integration.py | Dashboard view execution (all 6 views), batch validation distinction, extra columns, advisor integration |
| test_deliverables.py | Project deliverables and maintenance advisor tests |
| test_integration.py | Integration testing |
| test_milestone4_validation.py | Milestone 4 validation scenarios |
| test_prediction.py | Individual and batch prediction tests |
| test_preprocessing.py | Dataset preprocessing tests |
| test_validation.py | Input validation and error-handling tests |
| **Total** | **69 tests passed** |

### Manual Dashboard Testing

| Dashboard Module | Testing Performed | Result |
|---|---|---|
| Command Center | Dashboard loading and overview | PASS |
| Failure Prediction | Individual machine prediction | PASS |
| Batch CSV Analysis | Dataset upload and batch prediction | PASS |
| Failure Analytics | Failure statistics and visualizations | PASS |
| Model Performance | Evaluation metrics display | PASS |
| Maintenance Insights | Machine health and maintenance information | PASS |
| CSV Export | Export functionality checked | Tested |

### Input Validation Testing

**Test 1 – Valid CSV**

- Uploaded the original dataset.
- Required columns were detected.
- Validation passed.
- Batch prediction completed.

Result: PASS

**Test 2 – Malformed CSV**

- Uploaded a malformed CSV file.
- The application rejected the file.
- A CSV parsing error was displayed.

Result: REJECTED

Observation: The file was rejected, but the displayed parser error is technical. More user-friendly error handling can be implemented.

**Test 3 – Missing Required Columns**

- Uploaded a CSV with only `Machine`, `Temperature`, and `Speed`.
- Required column validation failed.
- The application listed the missing columns.
- Prediction was not executed.

Result: PASS

### End-to-End Testing

The following workflow was manually verified:

1. Launch the Streamlit application.
2. Open the Command Center.
3. Enter machine operating parameters.
4. Generate an individual prediction.
5. Upload a valid CSV dataset.
6. Run batch predictions.
7. View failure analytics.
8. Review model performance.
9. Open maintenance insights.

Result: PASS

### Testing Report

Automated test output is saved in:

```text
milestone4_test_results.txt
```

Regenerate the report using:

```powershell
python -m pytest Tests/ -v > milestone4_test_results.txt 2>&1
```

**Milestone 4 Outcome:** 60 passing automated tests, manual verification of the main dashboard modules, CSV validation checks, end-to-end workflow testing, and updated project documentation.

---

# 🧠 5. Machine Learning Model Performance

Evaluation was performed on a 20% stratified test set:

- Total test records: 2,000
- Normal records: 1,932
- Failure records: 68

Target leakage was addressed by excluding failure-cause indicator flags from model input features.

| Model Candidate | Accuracy | Precision (Failure) | Recall (Failure) | F1-Score (Failure) | ROC-AUC | FN | FP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression (Baseline) | 96.75% | 63.64% | 10.29% | 0.1772 | 0.8994 | 61 | 4 |
| Logistic Regression (Balanced) | 82.45% | 14.18% | 82.35% | 0.2419 | 0.9070 | 12 | 339 |
| Decision Tree (Baseline) | 97.85% | 68.66% | 67.65% | 0.6815 | 0.8328 | 22 | 21 |
| Decision Tree (Balanced) | 93.65% | 33.14% | 85.29% | 0.4774 | 0.8761 | 10 | 117 |
| Random Forest (Baseline) | 97.95% | 72.13% | 64.71% | 0.6822 | 0.9612 | 24 | 17 |
| **Improved Random Forest (Selected)** | **98.55%** | **76.00%** | **83.82%** | **0.7972** | **0.9823** | **11** | **18** |

### Resolving the Accuracy Paradox

The dataset contains significantly more normal records than failure records.

Baseline Logistic Regression achieves 96.75% accuracy but detects only 10.29% of actual failures.

For predictive maintenance, missed failures are particularly important because they can result in unexpected machine breakdowns.

The Improved Random Forest model achieves 83.82% failure recall and reduces false negatives from 24 to 11 compared with the baseline Random Forest.

This demonstrates the importance of evaluating precision, recall, F1-score, and false-negative counts alongside accuracy.

---

# 🏗️ 6. Repository Architecture

```text
AI-Based-Industrial-Maintenance-and-Process-Optimization/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── data_validation.py
│   ├── failure_analysis.py
│   ├── features.py
│   ├── machine_monitoring.py
│   ├── model_comparison.py
│   ├── prediction.py
│   └── predictive_model.py
│
├── dashboard/
│   ├── app.py
│   ├── components.py
│   ├── styles.py
│   └── views/
│       ├── overview.py
│       ├── prediction.py
│       ├── batch.py
│       ├── analytics.py
│       ├── performance.py
│       └── maintenance.py
│
├── data/
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── outputs/
│   ├── preprocessing_report.txt
│   ├── first_five_rows.png
│   └── Other generated reports and visualizations
│
├── Tests/
│   ├── test_deliverables.py
│   ├── test_integration.py
│   ├── test_milestone4_validation.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   └── test_validation.py
│
├── docs/
│
├── milestone4_test_results.txt
├── requirements.txt
└── README.md
```

*Note: This is a representative project structure. Generated files and additional documentation may vary depending on the current repository contents.*

---

# ⚡ 7. Getting Started and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/adithyanbalu/AI-Based-Industrial-Maintenance-and-Process-Optimization.git

cd AI-Based-Industrial-Maintenance-and-Process-Optimization
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

Windows CMD:

```cmd
venv\Scripts\activate
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Step 5: Preprocess Dataset

```bash
python src/data_preprocessing.py
```

### Step 6: Run Model Comparison

```bash
python src/model_comparison.py
```

### Step 7: Train and Save the Improved Model

```bash
python src/predictive_model.py
```

### Step 8: Test the Prediction Module

```bash
python src/prediction.py
```

### Step 9: Execute Automated Tests

```bash
python -m pytest Tests/ -v
```

### Step 10: Launch the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL provided by Streamlit in your browser.

---

# 🖥️ 8. Streamlit Command Center Pages

### 1. Command Center / Overview

- Fleet telemetry overview.
- Model status.
- Dataset information.
- KPI cards.
- Machine failure statistics.

### 2. Failure Prediction

- Individual machine parameter inputs.
- Failure probability estimation.
- Risk classification.
- Diagnostic information.
- Maintenance suggestions.

### 3. Batch CSV Analysis

- CSV file upload.
- Required column validation.
- Batch machine predictions.
- Normal and failure counts.
- Failure probability analysis.
- Export functionality.

### 4. Failure Analytics

- Historical failure analysis.
- Failure rate visualization.
- Failure mechanism analysis.
- Machine operating parameter comparison.

### 5. Model Performance

- Algorithm comparison.
- Model evaluation metrics.
- Confusion matrix information.
- False-positive and false-negative analysis.
- ROC-AUC results.

### 6. Maintenance Insights

- Machine health status.
- Warning and critical condition indicators.
- Maintenance guidance.
- Machine operating condition analysis.

---

# 🧪 9. Testing and Quality Assurance Summary

| Testing Area | Verified Result |
|---|---|
| Automated Testing | 60 tests passed |
| Failed Automated Tests | 0 |
| Dataset Validation | 10,000 records, 14 columns |
| Missing Value Check | 0 missing values |
| Duplicate Check | 0 duplicate records |
| Feature Engineering | PASS |
| Individual Prediction | PASS |
| Batch CSV Prediction | PASS |
| Failure Analytics | PASS |
| Model Performance | PASS |
| Maintenance Insights | PASS |
| Invalid CSV Format | Rejected with parser error |
| Missing Required Columns | PASS |
| Dashboard Runtime | PASS |
| End-to-End Testing | PASS |
| CSV Export | Tested |

### Test Execution Report

```text
milestone4_test_results.txt
```

The latest recorded automated test result:

```text
============================= 60 passed =============================
```

### Documentation

Project documentation is maintained alongside the source code and test report.

---

# ⚠️ 10. Project Limitations and Ethical Disclosures

1. **Benchmark Scope:** The system is based on the AI4I 2020 Predictive Maintenance benchmark dataset.
2. **No Hardware Claim:** The project does not claim live physical factory sensor connectivity.
3. **Decision Support:** Predictions and maintenance recommendations are decision-support information, not guaranteed physical diagnoses.
4. **Dataset Dependency:** Prediction quality depends on the quality and characteristics of the training dataset.
5. **Input Validation:** Malformed CSV files are rejected, but parser error messages can be improved.
6. **Generalization:** Model performance may vary for machines and operating conditions outside the training dataset.
7. **No Automated Shutdown:** The application does not directly control or shut down industrial machinery.

---

# 🚀 11. Future Scope

- Real-time IoT sensor integration.
- Live industrial machine monitoring.
- Cloud-based predictive maintenance.
- Advanced machine learning and deep learning models.
- Improved CSV input validation and user-friendly error messages.
- Automated maintenance scheduling.
- Alert and notification systems.
- Model retraining using additional industrial datasets.
- Deployment in practical industrial environments.

---

# 🎯 12. Conclusion

The **AI-Based Industrial Maintenance and Process Optimization System** demonstrates the application of machine learning in predictive maintenance.

The project progressed from dataset analysis and initial model development to data processing, model improvement, dashboard integration, testing, and documentation.

The selected Improved Random Forest model achieved **98.55% accuracy and 83.82% failure recall** in the reported evaluation.

The Streamlit dashboard provides individual machine predictions, batch CSV analysis, failure analytics, model performance evaluation, and maintenance insights.

As part of Milestone 4, the automated test suite recorded **60 passing tests with zero failures**. Manual verification was also performed across the main dashboard modules, CSV validation scenarios, and end-to-end workflow.

The system provides a foundation for reducing unexpected machine downtime and supporting data-driven industrial maintenance decisions.

---

<p align="center">
  <b>Project:</b> AI-Based Industrial Maintenance and Process Optimization System<br/>
  <b>Internship:</b> Infosys Springboard Virtual Internship 7.0<br/>
  <b>Track:</b> Artificial Intelligence<br/>
  <b>Milestones:</b> 1 – 4<br/>
  <b>Final Document:</b> README.md
</p>