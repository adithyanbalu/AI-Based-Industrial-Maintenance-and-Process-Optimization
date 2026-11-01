# Implementation Documentation

## AI-Based Industrial Maintenance and Process Optimization System

### 1. Project Overview

The AI-Based Industrial Maintenance and Process Optimization System is designed to predict potential machine failures using machine operating data and Machine Learning techniques.

The system accepts machine parameters, processes the input data, predicts the probability of machine failure, and displays the prediction results through a Streamlit dashboard.

### 2. Project Architecture

The project consists of the following major components:

1. Data Preprocessing
2. Feature Engineering
3. Predictive Model Training
4. Model Comparison
5. Machine Failure Prediction
6. Failure Analysis
7. Interactive Dashboard

### 3. Module Implementation

#### 3.1 Data Preprocessing – `src/data_preprocessing.py`

This module reads the machine dataset from `data/Dataset.csv`.

It performs initial dataset inspection, including:

- Dataset dimensions
- First five records
- Column data types
- Missing value identification
- Duplicate record identification

It generates a preprocessing report and a visualization of the first five dataset rows.

Output:
- `outputs/preprocessing_report.txt`
- `outputs/first_five_rows.png`

Note: This module performs dataset inspection and reporting. It does not itself perform data cleaning or remove duplicate records.

#### 3.2 Feature Engineering – `src/features.py`

This module defines the machine input features and creates additional engineering features.

The six input features are:

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

Additional calculated features:

- Temp_diff: Difference between process temperature and air temperature.
- Power_W: Mechanical power calculated using rotational speed and torque.
- Strain: Calculated using torque and tool wear.

These additional features help the predictive model identify useful patterns in machine operating conditions.

#### 3.3 Predictive Model Training – `src/predictive_model.py`

This module builds and trains the predictive maintenance model.

The implementation uses a Machine Learning pipeline containing:

- IndustrialFeatureEngineer
- Data preprocessing using imputers and StandardScaler
- OneHotEncoder for categorical features
- RandomForestClassifier

The Random Forest model uses the following configuration:

- Number of estimators: 250
- Maximum depth: 12
- Minimum samples split: 4
- Class weight: Balanced
- Random state: 42

The dataset is divided into training and testing sets using an 80:20 split.

The trained model and metadata are saved for later prediction.

#### 3.4 Model Comparison – `src/model_comparison.py`

This module compares six Machine Learning models:

1. Logistic Regression (Baseline)
2. Logistic Regression (Balanced)
3. Decision Tree (Baseline)
4. Decision Tree (Balanced)
5. Random Forest (Baseline)
6. Improved Random Forest

The models are evaluated using:

- Accuracy
- Failure Precision
- Failure Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Training Time

The module generates comparison reports and visualizations.

Output:
- `outputs/model_comparison.csv`
- `outputs/model_comparison.txt`
- `outputs/model_comparison_metrics.png`
- `outputs/model_roc_curves.png`
- `outputs/model_confusion_matrices.png`

The implementation report identifies Improved Random Forest as the selected model.

#### 3.5 Machine Failure Prediction – `src/prediction.py`

This module performs predictions using the trained Machine Learning model.

Main functionalities:

- Loading the saved model
- Validating machine input data
- Checking required columns
- Checking missing values
- Validating machine types
- Checking numerical values and permitted ranges
- Generating failure predictions
- Calculating failure probabilities
- Assigning risk levels

Risk classification:

- Low Risk: Probability below 30%
- Medium Risk: Probability from 30% to below 65%
- High Risk: Probability of 65% or above

The prediction output includes:

- Prediction
- Prediction Label
- Status
- Failure Probability
- Risk Level
- Diagnostic Notes

The module supports both single-machine and multiple-machine predictions.

#### 3.6 Failure Analysis – `src/failure_analysis.py`

This module analyses historical machine failure information from the dataset.

It calculates:

- Missing value counts
- Duplicate record counts
- Total normal records
- Total failure records
- Failure percentages

It also generates a failure distribution visualization.

Output:
- `outputs/failure_distribution.png`

#### 3.7 Dashboard

The project provides an interactive dashboard using Streamlit.

The dashboard supports:

- Manual machine input
- CSV file upload
- Machine failure prediction
- Batch prediction results
- Failure analytics
- Model performance information
- Maintenance insights

The dashboard displays prediction results and risk levels to help users understand machine conditions.

### 4. Data Flow

1. The system loads machine operating data.
2. The input data is inspected and validated.
3. Feature engineering generates additional machine features.
4. The trained model processes the input.
5. The system predicts machine failure probability.
6. A risk level is assigned.
7. Prediction results are displayed through the dashboard.

### 5. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Pytest

### 6. Model Outputs and Artifacts

The system stores trained model files, metadata, reports, and visualizations in the project directories.

Important directories:

- `data/` – Dataset files
- `src/` – Source code modules
- `models/` – Saved Machine Learning model and metadata
- `outputs/` – Reports and generated visualizations
- `dashboard/` – Dashboard implementation
- `tests/` – Automated test cases
- `docs/` – Project documentation

### 7. Conclusion

The system combines machine data processing, feature engineering, predictive modelling, failure analysis, and an interactive dashboard to support predictive maintenance.

It aims to identify potential machine failures before they occur and provide useful information for maintenance planning.