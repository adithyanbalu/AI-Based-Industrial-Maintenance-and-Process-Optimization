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
</p>

---

## 📌 Project Overview

Industrial machinery failures create unexpected plant downtime, severe maintenance costs, and supply chain disruptions. This project develops an end-to-end **AI-Based Predictive Maintenance Command Center** built on the **AI4I 2020 Predictive Maintenance benchmark dataset** (10,000 records).

The system integrates data preprocessing, physics-informed domain feature engineering, cost-sensitive machine learning classification, automated batch evaluation, and deterministic rule-based maintenance guidance within a Streamlit industrial analytics platform.

---

## 👥 Project Team & Responsibilities

Developed as part of **Infosys Springboard Virtual Internship 7.0**:

- **Adithyan B (Team Lead):** ML Model Comparison, Predictive Model Improvement, Joblib Serialization, Reusable Prediction Module, Streamlit Dashboard Architecture, and Integration.
- **Hriday Kaustubh Machiraju:** Data Validation and Machine Monitoring.
- **Dheepesh N:** Failure Diagnostics and Sensor Analysis.
- **Ganga Vanaparthi:** Testing, Integration Verification, and Documentation.

---

## 🧠 Machine Learning Model Performance

Evaluation performed on a 20% stratified test set ($N=2,000$ records: $1,932$ normal, $68$ failures). Target leakage was eliminated by strictly excluding failure-cause indicator flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).

| Model Candidate | Accuracy | Precision (Failure) | Recall (Failure) | F1-Score (Failure) | ROC-AUC | Missed Failures (FN) | False Alarms (FP) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | 96.75% | 63.64% | 10.29% | 0.1772 | 0.8994 | 61 | 4 |
| **Logistic Regression (Balanced)** | 82.45% | 14.18% | 82.35% | 0.2419 | 0.9070 | 12 | 339 |
| **Decision Tree (Baseline)** | 97.85% | 68.66% | 67.65% | 0.6815 | 0.8328 | 22 | 21 |
| **Decision Tree (Balanced)** | 93.65% | 33.14% | 85.29% | 0.4774 | 0.8761 | 10 | 117 |
| **Random Forest (Baseline)** | 97.95% | 72.13% | 64.71% | 0.6822 | 0.9612 | 24 | 17 |
| **Improved Random Forest (Selected)** | **98.55%** | **76.00%** | **83.82%** | **0.7972** | **0.9823** | **11** | **18** |

### Resolving the Accuracy Paradox

On this imbalanced dataset (96.61% normal, 3.39% failure), baseline Logistic Regression achieves **96.75% accuracy**, yet misses **61 out of 68 actual breakdowns (Recall: 10.29%)**. In industrial maintenance, missed failures cause catastrophic breakdowns. Our **Improved Random Forest** with balanced weights and physics-informed features achieves **83.82% Recall**, reducing missed failures by over 54% (from 24 down to 11) with only 18 false alarms.

---

## 🏗️ Repository Architecture

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
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── outputs/
│   ├── 01_uploaded_records.png
│   ├── 02_validated_parameters.png
│   ├── 03_workflow.png
│   └── Other generated reports and visualizations
│
├── Tests/
│   ├── test_integration.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   └── test_validation.py
│
├── tests/
│   └── test_deliverables.py
│
├── docs/
│   ├── implementation.md
│   └── testing_report.md
│
├── requirements.txt
└── README.md
```

---

## ⚡ Getting Started & Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/adithyanbalu/AI-Based-Industrial-Maintenance-and-Process-Optimization.git

cd AI-Based-Industrial-Maintenance-and-Process-Optimization
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Preprocess Dataset

```bash
python src/data_preprocessing.py
```

### 5. Run Algorithm Comparison Suite

```bash
python src/model_comparison.py
```

### 6. Train and Save the Improved Model

```bash
python src/predictive_model.py
```

### 7. Test the Inference Engine

```bash
python src/prediction.py
```

### 8. Run Automated Tests

```bash
python -m pytest -v
```

The current testing report records **32 passing automated tests** across preprocessing, validation, prediction, integration, and deliverables testing.

### 9. Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🖥️ Streamlit Command Center Pages

1. **Overview / Command Center:** Fleet telemetry overview, KPI cards, class distribution, machine variant failure rates, failure mechanisms breakdown, and architecture insights.
2. **Machine Failure Prediction:** Real-time parameter inputs, operating presets, online physics telemetry, failure probability gauge, and diagnostic explanations.
3. **Batch CSV Analysis:** CSV file uploader, multi-machine evaluation, missing value handling, batch KPIs, and downloadable prediction results.
4. **Failure Analytics:** Historical failure analysis, operating parameter visualizations, and statistical comparison tables.
5. **Model Performance:** Candidate algorithm comparison, evaluation metrics, confusion matrix, ROC curves, and feature importance.
6. **Maintenance Recommendations:** Rule-based maintenance inspection guidance and an interactive maintenance simulator.

---

## 🧪 Testing and Documentation

Project testing and implementation documentation are available in the `docs/` directory.

- **Testing Report:** [`docs/testing_report.md`](docs/testing_report.md)
- **Implementation Documentation:** [`docs/implementation.md`](docs/implementation.md)

### Testing Summary

- Automated tests: 32 passed.
- Preprocessing tests: 3 passed.
- Validation tests: 9 passed.
- Prediction tests: 9 passed.
- Integration tests: 2 passed.
- Existing deliverables tests: 9 passed.

Manual dashboard testing included normal machine inputs, high-risk predictions, CSV batch processing, failure analytics, model loading error handling, model performance, and maintenance recommendations.

---

## ⚠️ Project Limitations & Ethical Disclosures

1. **Benchmark Scope:** Evaluates the AI4I 2020 Predictive Maintenance benchmark dataset.
2. **No Hardware Claim:** Does not claim live physical factory sensor connectivity or automated plant shutdown actuation.
3. **Decision-Support:** All predictions and maintenance advisories serve as decision-support guidance for qualified plant personnel, not guaranteed physical diagnoses.