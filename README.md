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

- **Adithyan B :** ML Model Comparison, Predictive Model Improvement, Joblib Serialization, Reusable Prediction Module, Streamlit Dashboard Architecture, and Integration.
- **Hriday Kaustubh Machiraju:** Data Validation and Machine Monitoring.
- **Dheepesh N:** Failure Diagnostics and Sensor Analysis.
- **Ganga Vanaparthi:** Testing, Integration Verification, and Documentation.

---

## 🧠 Machine Learning Model Performance

Evaluation performed on a 20% stratified test set ($N=2,000$ records: $1,932$ normal, $68$ failures). Target leakage was eliminated by strictly excluding failure-cause indicator flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).

| Model Candidate | Accuracy | Precision (Failure) | Recall (Failure) | F1-Score (Failure) | ROC-AUC | Missed Failures (FN) | False Alarms (FP) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
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
AI-Based-Industrial-Maintenance/
│
├── data/
│   └── Dataset.csv                     # AI4I 2020 Predictive Maintenance dataset (10,000 rows)
│
├── src/
│   ├── __init__.py
│   ├── features.py                     # IndustrialFeatureEngineer (Physics domain features)
│   ├── data_preprocessing.py           # Dataset inspection, null/duplicate audit, summary reports
│   ├── failure_analysis.py             # Historical failure count and distribution visualization
│   ├── model_comparison.py             # Multi-algorithm comparison suite (LR, DT, RF)
│   ├── predictive_model.py             # Improved Random Forest training & evaluation
│   └── prediction.py                   # Reusable inference engine (single & batch predictions)
│
├── dashboard/
│   ├── app.py                          # Streamlit Industrial AI Command Center (6 views)
│   ├── components.py                   # Reusable UI cards, Plotly charts, integration wrappers
│   └── styles.py                       # Industrial dark theme styling tokens & circular gauges
│
├── models/
│   ├── best_model.pkl                  # Serialized Joblib pipeline
│   └── model_metadata.json             # Metrics, confusion matrix, and feature importances
│
├── outputs/                            # Model reports, comparison CSVs, and visualization charts
│
├── tests/
│   └── test_deliverables.py            # Automated pytest verification suite (9 tests)
│
├── requirements.txt                    # Project dependencies
└── README.md
```

---

## ⚡ Getting Started & Running the Project

### 1. Installation
Clone repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Preprocess Dataset
```bash
python src/data_preprocessing.py
```

### 3. Run Algorithm Comparison Suite
```bash
python src/model_comparison.py
```

### 4. Train & Save the Improved Model
```bash
python src/predictive_model.py
```

### 5. Test Inference Engine
```bash
python src/prediction.py
```

### 6. Run Automated Test Suite
```bash
python -m pytest tests/test_deliverables.py -v
```

### 7. Launch the Streamlit Industrial AI Command Center
```bash
streamlit run dashboard/app.py
```

---

## 🖥️ Streamlit Command Center Pages

1. **Overview / Command Center:** Fleet telemetry overview, 6 KPI cards, donut class distribution, machine variant failure rates, failure mechanisms breakdown, and architecture insights.
2. **Machine Failure Prediction:** Real-time parameter inputs, 5 one-click operating presets, online physics telemetry ($\Delta T$, Spindle Power, Overstrain Index), failure probability gauge, and diagnostic root factor explanations.
3. **Batch CSV Analysis:** File uploader supporting multi-machine evaluation, downloadable template, missing value imputation, batch KPIs, and CSV export.
4. **Failure Analytics:** Empirical scatter plot of spindle speed vs torque operating envelopes, parameter distribution boxplots, and statistical comparison tables.
5. **Model Performance:** Candidate algorithm comparison matrix, grouped metric bar chart, confusion matrix heatmap ($N=2,000$), multi-model ROC curves, and Gini feature importances.
6. **Maintenance Recommendations:** Deterministic, physics-grounded inspection rules (TWF, HDF, PWF, OSF, Nominal) and an interactive maintenance rule simulator.

---

## ⚠️ Project Limitations & Ethical Disclosures

1. **Benchmark Scope:** Evaluates the AI4I 2020 Predictive Maintenance benchmark dataset.
2. **No Hardware Claim:** Does not claim live physical factory sensor connectivity or automated plant shutdown actuation.
3. **Decision-Support:** All predictions and maintenance advisories serve as decision-support guidance for qualified plant personnel, not guaranteed physical diagnoses.
