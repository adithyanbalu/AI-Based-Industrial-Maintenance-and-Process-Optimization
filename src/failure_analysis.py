import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/Dataset.csv")

# Check data quality
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Count machine failures
counts = df["Machine failure"].value_counts().sort_index()

no_failure = counts.get(0, 0)
failure = counts.get(1, 0)
total = len(df)

# Calculate percentages
print("No failure:", no_failure)
print("Failure:", failure)
print("No failure percentage:", round(no_failure / total * 100, 2), "%")
print("Failure percentage:", round(failure / total * 100, 2), "%")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Plot bar chart
counts.rename({
    0: "No Failure",
    1: "Failure"
}).plot(kind="bar")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Status")
plt.ylabel("Number of Records")
plt.tight_layout()

# Save chart
plt.savefig("outputs/failure_distribution.png")
plt.show()
