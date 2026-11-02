import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PARAMETERS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def run_failure_diagnostics(csv_path, output_dir="outputs"):

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # -------------------------------
    # 1. FAILURE ANALYSIS
    # -------------------------------

    counts = df["Machine failure"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    percentages = (counts / len(df)) * 100

    failure_summary = pd.DataFrame({
        "Count": counts.values,
        "Percentage": percentages.round(2).values
    }, index=["No Failure", "Failure"])

    print("\n========== FAILURE DISTRIBUTION ==========")
    print(failure_summary)

    # Failure distribution chart
    plt.figure(figsize=(7, 4))

    sns.barplot(
        x=["No Failure", "Failure"],
        y=counts.values
    )

    plt.title("Machine Failure Distribution")
    plt.xlabel("Failure Status")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "failure_distribution.png"),
        dpi=160
    )
    plt.close()

    # Failure mode analysis
    failure_modes = df[FAILURE_MODES].sum().sort_values(
        ascending=False
    )

    print("\n========== FAILURE MODE ANALYSIS ==========")
    print(failure_modes)

    plt.figure(figsize=(7, 4))

    sns.barplot(
        x=failure_modes.index,
        y=failure_modes.values
    )

    plt.title("Failure Mode Counts")
    plt.xlabel("Failure Mode")
    plt.ylabel("Number of Failures")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "failure_modes.png"),
        dpi=160
    )
    plt.close()

    # Average parameters by failure status
    averages = df.groupby("Machine failure")[PARAMETERS].mean()

    print("\n========== AVERAGE PARAMETERS BY FAILURE ==========")
    print(averages.round(2))

    # Correlation heatmap
    plt.figure(figsize=(9, 6))

    sns.heatmap(
        df[PARAMETERS + ["Machine failure"]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0
    )

    plt.title("Machine Parameter Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "correlation_heatmap.png"),
        dpi=160
    )
    plt.close()

    # -------------------------------
    # 2. SENSOR ANALYSIS
    # -------------------------------

    print("\n========== SENSOR DESCRIPTIVE STATISTICS ==========")

    print(df[PARAMETERS].describe().round(2))

    print("\n========== SENSOR AVERAGES BY FAILURE STATUS ==========")
    print(df.groupby("Machine failure")[PARAMETERS].mean().round(2))

    # Individual sensor distribution charts
    sensor_labels = {
        "Air temperature [K]": "Air Temperature (K)",
        "Process temperature [K]": "Process Temperature (K)",
        "Rotational speed [rpm]": "Rotational Speed (rpm)",
        "Torque [Nm]": "Torque (Nm)",
        "Tool wear [min]": "Tool Wear (min)"
    }

    for column, label in sensor_labels.items():

        plt.figure(figsize=(7, 4))

        sns.histplot(
            data=df,
            x=column,
            hue="Machine failure",
            bins=30,
            kde=True,
            common_norm=False
        )

        plt.title(f"{label} Distribution by Failure Status")
        plt.xlabel(label)
        plt.tight_layout()

        filename = (
            column.lower()
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_")
        )

        plt.savefig(
            os.path.join(output_dir, f"{filename}.png"),
            dpi=160
        )

        plt.close()

    print("\nFailure and sensor diagnostics completed.")
    print(f"All charts saved in: {output_dir}")

    return {
        "failure_summary": failure_summary,
        "failure_modes": failure_modes,
        "parameter_averages": averages,
        "sensor_statistics": df[PARAMETERS].describe()
    }
