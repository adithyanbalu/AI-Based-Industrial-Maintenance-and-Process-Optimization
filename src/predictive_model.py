import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Dataset.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

target = "Machine failure"

# 2. Select features and target
# Exclude identifiers and failure-mode columns to prevent target leakage.
excluded_columns = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

X = df.drop(columns=excluded_columns)
y = df[target]

# 3. Separate categorical and numerical features
categorical_features = ["Type"]
numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 4. Preprocess features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 5. Split data while preserving the failure ratio
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 6. Build the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

# 7. Train
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Failure", "Failure"],
    zero_division=0,
)

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
roc_auc = roc_auc_score(y_test, y_prob)

results = f"""
PREDICTIVE MAINTENANCE MODEL RESULTS
====================================
Dataset rows: {len(df)}
Training records: {len(X_train)}
Testing records: {len(X_test)}
Features used: {list(X.columns)}

Model: Random Forest Classifier
Class weighting: balanced

CLASSIFICATION REPORT
{report}

CONFUSION MATRIX
Rows = actual, columns = predicted
{matrix}

ROC-AUC: {roc_auc:.4f}
"""

print(results)
(OUT / "model_results.txt").write_text(results, encoding="utf-8")

print(f"Results saved to: {OUT / 'model_results.txt'}")


# 9. Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(matrix, cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set(
    title="Predictive Maintenance - Confusion Matrix",
    xlabel="Predicted Label",
    ylabel="Actual Label",
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"],
)

for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        ax.text(
            col,
            row,
            str(matrix[row, col]),
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )

fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close(fig)

print(f"Confusion matrix saved to: {OUT / 'confusion_matrix.png'}")

