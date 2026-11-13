# 🏭 AI-Based Industrial Maintenance and Process Optimization System

### An Intelligent Predictive Maintenance & Industrial Analytics Platform

<p align="center">
  <strong>Predict failures. Understand machine behavior. Make smarter maintenance decisions.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Data%20Visualization-Plotly-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Testing-Pytest-green?logo=pytest&logoColor=white" />
</p>

---

## 📌 1. Project Overview

Unexpected industrial machine failures can cause production downtime, increased maintenance expenses, equipment damage, and operational inefficiencies.

The **AI-Based Industrial Maintenance and Process Optimization System** is an intelligent, dataset-driven predictive maintenance platform developed to help identify potential machine failures using machine operating parameters and machine learning.

The system analyzes industrial machine data, applies data preprocessing and feature engineering techniques, evaluates multiple machine learning algorithms, and uses an improved Random Forest model to predict potential machine failures.

It also provides an interactive Streamlit dashboard where users can analyze machine conditions, perform individual and batch predictions, explore failure patterns, evaluate model performance, and access maintenance recommendations.

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, containing 10,000 records and 14 columns.

> **Project scope:** This is a dataset-based predictive maintenance and decision-support system. It does not claim live industrial IoT connectivity, physical sensor integration, or automated factory control.

## 🎯 2. Project Objectives

- Analyze industrial machine operating data to understand failure patterns.
- Identify important parameters associated with machine failures.
- Develop and compare multiple machine learning classification models.
- Improve failure detection using class balancing and feature engineering.
- Predict machine failure risks using an optimized Random Forest model.
- Support individual machine predictions and batch CSV analysis.
- Visualize machine performance, failure distributions, and important features.
- Generate understandable, rule-based maintenance recommendations.
- Build a user-friendly industrial analytics dashboard.
- Validate the system through automated testing and integration checks.

## 🛠️ 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| ML Algorithms | Logistic Regression, Decision Tree, Random Forest |
| Dashboard | Streamlit |
| Data Visualization | Plotly, Matplotlib |
| Model Serialization | Joblib / Pickle |
| Testing | Pytest, Streamlit AppTest |
| Version Control | Git, GitHub |

## 📊 4. Dataset Information

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

The dataset contains simulated industrial machine operating conditions and failure information.

| Property | Description |
|---|---|
| Dataset | AI4I 2020 Predictive Maintenance |
| Total Records | 10,000 |
| Total Columns | 14 |
| Target Variable | Machine Failure |
| Problem Type | Binary Classification |
| Normal Records | 9,661 |
| Failure Records | 339 |
| Normal Distribution | 96.61% |
| Failure Distribution | 3.39% |

### Important Machine Parameters

- Product Type
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (rpm)
- Torque (Nm)
- Tool Wear (min)

The dataset also contains failure-cause indicators. These are excluded from the predictive feature set to prevent target leakage.

## 🧠 5. Machine Learning Approach

The project follows a structured machine learning workflow.

1. **Data Preprocessing:** Load the dataset, inspect its structure, and prepare the data for analysis.
2. **Data Validation:** Check input parameters, missing values, invalid values, and operating ranges.
3. **Exploratory Data Analysis:** Investigate machine characteristics and failure distributions.
4. **Feature Engineering:** Develop additional informative features from machine operating parameters.
5. **Model Development:** Train and compare multiple classification algorithms.
6. **Model Evaluation:** Evaluate accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
7. **Model Selection:** Select the improved Random Forest model based on its overall evaluation performance.
8. **Prediction:** Generate machine failure predictions and risk information.
9. **Maintenance Guidance:** Provide rule-based recommendations based on machine operating conditions.

### Model Comparison

The selected model is an improved Random Forest classifier designed to improve the detection of minority-class machine failures.

The following comparison represents the project's documented evaluation on a stratified 20% test set containing 2,000 records.

| Model | Accuracy | Failure Recall | Failure F1-score |
|---|---:|---:|---:|
| Logistic Regression (Baseline) | 96.75% | 10.29% | 0.1772 |
| Logistic Regression (Balanced) | 82.45% | 82.35% | 0.2419 |
| Decision Tree (Baseline) | 97.85% | 67.65% | 0.6815 |
| Decision Tree (Balanced) | 93.65% | 85.29% | 0.4774 |
| Random Forest (Baseline) | 97.95% | 64.71% | 0.6822 |
| **Improved Random Forest (Selected)** | **98.55%** | **83.82%** | **0.7972** |

### Why Failure Recall Matters

Industrial datasets are often imbalanced, meaning normal machine records greatly outnumber failure records.

A model can achieve high accuracy while still missing a significant number of actual failures.

Therefore, this project emphasizes failure detection and recall alongside accuracy.

The selected Improved Random Forest achieved:

- Accuracy: 98.55%
- Failure Precision: 76.00%
- Failure Recall: 83.82%
- Failure F1-score: 0.7972
- ROC-AUC: 0.9823
- Missed Failures: 11 out of 68

These results demonstrate the importance of evaluating predictive maintenance models beyond accuracy alone.

## 🖥️ 6. Interactive Streamlit Dashboard

The project includes a multi-section industrial analytics dashboard with six major modules.

### 1. Overview – Industrial Command Center

Provides an overview of the dataset and machine failure information.

Features:
- Dataset summary and key performance indicators.
- Machine failure distribution.
- Machine variant analysis.
- Failure mechanism visualization.
- Important industrial insights.

### 2. Single Machine Prediction

Allows users to enter individual machine operating parameters and obtain a failure prediction.

Features:
- Interactive machine parameter inputs.
- Machine type selection.
- Operating parameter validation.
- Failure probability estimation.
- Risk classification.
- Diagnostic explanations.

### 3. Batch CSV Analysis

Allows users to upload a CSV file containing multiple machine records and analyze them together.

Features:
- CSV file upload.
- Input validation and missing-value handling.
- Invalid and out-of-range value detection.
- Batch machine failure predictions.
- Failure risk analysis.
- Downloadable prediction results.
- Preservation of additional customer-provided CSV columns.

### 4. Failure Analytics

Provides insights into machine failure patterns and operating conditions.

Features:
- Failure distribution analysis.
- Machine parameter comparisons.
- Failure mechanism analysis.
- Statistical visualizations.
- Identification of important operating patterns.

### 5. Model Performance

Provides an overview of machine learning model evaluation.

Features:
- Model comparison.
- Accuracy and classification metrics.
- Confusion matrix.
- ROC curves.
- Feature importance visualization.
- Model performance analysis.

### 6. Maintenance Insights

Provides rule-based maintenance guidance using machine operating conditions.

Features:
- Machine condition analysis.
- Tool wear assessment.
- Maintenance recommendations.
- Risk-based inspection guidance.
- Interactive maintenance insights.

## 🏗️ 7. Project Architecture

The repository follows a modular structure to separate data processing, machine learning, dashboard functionality, and testing.

```text
AI-Based-Industrial-Maintenance-and-Process-Optimization/
│
├── .streamlit/
│   └── config.toml
│
├── dashboard/
│   ├── app.py
│   ├── components.py
│   ├── styles.py
│   │
│   └── views/
│       ├── overview.py
│       ├── prediction.py
│       ├── batch.py
│       ├── analytics.py
│       ├── performance.py
│       └── maintenance.py
│
├── data/
│   └── Dataset.csv
│
├── docs/
│   ├── implementation.md
│   └── testing_report.md
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── outputs/
│   └── Generated reports and visualizations
│
├── screenshots/
│   └── Project screenshots
│
├── src/
│   ├── Data preprocessing
│   ├── Data validation
│   ├── Failure analysis
│   ├── Feature engineering
│   ├── Model comparison
│   ├── Prediction
│   └── Predictive modeling
│
├── tests/
│   └── Automated and integration tests
│
├── .gitignore
├── requirements.txt
├── README.md
├── TEAM_TASK_ALLOCATION.md
└── MILESTONE_4_WORK_PLAN.md
```

## ⚡ 8. Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

- Python 3.x
- Git
- VS Code or another Python IDE
- Internet connection for installing dependencies

### Step 1: Clone the Repository

```bash
git clone https://github.com/adithyanbalu/AI-Based-Industrial-Maintenance-and-Process-Optimization.git
```

### Step 2: Navigate to the Project Directory

```bash
cd AI-Based-Industrial-Maintenance-and-Process-Optimization
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 6: Verify Dataset and Model Files

Ensure that the required project files are available:

```text
data/Dataset.csv
models/best_model.pkl
models/model_metadata.json
```

### Step 7: Launch the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open this URL in your browser to access the dashboard.

## 🧪 9. Testing and Verification

Testing is an important part of ensuring the reliability of the predictive maintenance system.

The project includes unit tests, integration tests, input validation tests, prediction tests, and dashboard verification.

### Run the Automated Test Suite

```bash
python -m pytest tests/ -v
```

### Latest Verification Results

| Verification | Result |
|---|---|
| Automated Pytest Suite | 69 Passed |
| Failed Tests | 0 |
| Dashboard Sections | 6 Manually Verified |
| Dashboard AppTest Checks | Passed |
| Batch Validation | Verified |
| Missing and Invalid Value Handling | Verified |
| Extra CSV Column Preservation | Verified |
| GitHub Publication | Successful |

**Latest verified test result: 69 passed, 0 failed.**

These results represent the project's verification at the time of the Milestone 4 final checks.

## 🔒 10. Project Scope and Limitations

The project is developed as a dataset-based predictive maintenance solution.

Important considerations:

- The system uses the AI4I 2020 benchmark dataset.
- It does not connect to live industrial IoT devices or physical sensors.
- Predictions are based on the trained machine learning model and supplied input parameters.
- Maintenance recommendations are generated using rule-based logic.
- Predictions are decision-support outputs and should not replace professional industrial inspection.
- Model performance on benchmark data does not guarantee equivalent performance in real industrial environments.

## 👥 11. Project Team

Developed as part of the **Infosys Springboard Virtual Internship 7.0**.

| S.No. | Team Member |
|:---:|---|
| 1 | Adithyan B |
| 2 | Hriday Kaustubh Machiraju |
| 3 | Dheepesh N |
| 4 | Ganga Vanaparthi |

**Team Effort:** This project was collaboratively developed by all four team members, with contributions throughout the design, development, implementation, testing, and documentation phases.

## 🚀 12. Future Enhancements

Potential future improvements include:

- Integration with real industrial sensor data.
- Real-time machine condition monitoring.
- Time-series-based failure prediction.
- Advanced anomaly detection.
- Remaining Useful Life (RUL) estimation.
- Cloud deployment and remote monitoring.
- Advanced predictive maintenance scheduling.
- Integration with industrial maintenance management systems.

These are potential extensions and are not claimed as current implemented capabilities.

## 🏆 13. Project Outcome

The project demonstrates how machine learning and interactive data visualization can support predictive maintenance workflows.

By combining data preprocessing, feature engineering, model comparison, failure prediction, batch analysis, and maintenance guidance, the system provides an integrated platform for exploring industrial machine health and potential failure risks.

It also demonstrates the importance of handling imbalanced datasets, evaluating failure recall, validating user inputs, and building reliable machine learning applications.

---

## 📌 Repository

**GitHub:**  
https://github.com/adithyanbalu/AI-Based-Industrial-Maintenance-and-Process-Optimization

**Project:** AI-Based Industrial Maintenance and Process Optimization System

**Internship:** Infosys Springboard Virtual Internship 7.0


---

<p align="center">
  <strong>Built with Python, Machine Learning, and Streamlit.</strong>
  <br/>
  <em>Turning industrial data into actionable maintenance insights.</em>
</p>
