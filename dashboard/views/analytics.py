"""Page: Failure Analytics. Every chart and observation is computed from data/Dataset.csv."""

import numpy as np
import streamlit as st

from dashboard.components import (
    COLORS,
    chart_failure_correlation,
    chart_failure_distribution,
    chart_failure_modes,
    chart_operating_envelope,
    chart_parameter_distribution,
    chart_rate_by_bin,
    chart_type_distribution,
    dataset_summary,
    insight_grid,
    kpi_grid,
    load_benchmark_dataset,
    md,
    page_header,
    panel_title,
    section_header,
    show_chart,
    show_table,
)

df = load_benchmark_dataset()
S = dataset_summary(df)
if not S:
    st.error("Dataset not found. Expected at data/Dataset.csv.")
    st.stop()

page_header(
    "Empirical analysis",
    "Failure Analytics",
    f"Patterns in {S['total']:,} historical machine records: where failures happen, which parameters drive them and how risk "
    "changes with tool wear, temperature and torque.",
    chips=[("AI4I 2020 benchmark", "cyan"), (f"{S['failures']} failures analysed", "")],
)

# ---- derived statistics used in KPIs and observations
fail, ok = df[df["Machine failure"] == 1], df[df["Machine failure"] == 0]
rates_type = (S["by_type"]["sum"] / S["by_type"]["count"] * 100).sort_values(ascending=False)
wear_hi = df[df["Tool wear [min]"] >= 200]
wear_lo = df[df["Tool wear [min]"] < 200]
torque_hi = df[df["Torque [Nm]"] > 50]
torque_lo = df[df["Torque [Nm]"] <= 50]
dt_low = df[df["Temp_diff"] < 8.6]
dt_ok = df[df["Temp_diff"] >= 8.6]


def rate(sub):
    return sub["Machine failure"].mean() * 100 if len(sub) else float("nan")


top_mode = max(S["modes"], key=S["modes"].get) if S["modes"] else "—"
kpi_grid([
    dict(title="Failure rate", value=f"{S['fail_rate']:.2f}", unit="%", subtext=f"{S['failures']} of {S['total']:,} records", icon_name="percent",
         accent=COLORS["status_danger"], progress=S["fail_rate"] * 5),
    dict(title="Most failure-prone type", value=f"Type {rates_type.index[0]}", subtext=f"{rates_type.iloc[0]:.2f}% failure rate", icon_name="category",
         accent=COLORS["status_warning"]),
    dict(title="Dominant failure mode", value=top_mode, subtext=f"{S['modes'].get(top_mode, 0)} recorded events", icon_name="local_fire_department",
         accent=COLORS["accent_violet"]),
    dict(title="Mean tool wear", value=f"{fail['Tool wear [min]'].mean():.0f}", unit=" min", subtext=f"failed machines vs {ok['Tool wear [min]'].mean():.0f} min normal",
         icon_name="build", accent=COLORS["brand_cyan"]),
])

# ---- 1. Distribution
section_header("Failure distribution", "How failures split across classes, machine types and mechanisms", "donut_large")
a, b, c = st.columns([1, 1, 1.15], gap="medium")
with a:
    with st.container(key="panel_a1"):
        panel_title("Class split", f"{S['fail_rate']:.2f}% failure rate")
        show_chart(chart_failure_distribution(df), key="an_donut")
with b:
    with st.container(key="panel_a2"):
        panel_title("Failure rate by product type")
        show_chart(chart_type_distribution(df), key="an_type")
with c:
    with st.container(key="panel_a3"):
        panel_title("Failure mechanisms")
        show_chart(chart_failure_modes(df), key="an_modes")

# ---- 2. Machine parameters
section_header("Machine parameter analysis", "Operating envelope and which parameters move with failure", "scatter_plot")
d1, d2 = st.columns([1.5, 1], gap="medium")
with d1:
    with st.container(key="panel_env"):
        panel_title("Speed vs torque operating envelope", "All failures shown; normal records sampled for speed")
        show_chart(chart_operating_envelope(df), key="an_env")
with d2:
    with st.container(key="panel_corr"):
        panel_title("Correlation with failure", "Which parameters move with failure")
        show_chart(chart_failure_correlation(df, height=420), key="an_corr")

# ---- 3. Tool wear
section_header("Tool wear analysis", "Failure risk as the cutting tool ages", "build")
e1, e2 = st.columns([1.5, 1], gap="medium")
with e1:
    with st.container(key="panel_wear"):
        panel_title("Failure rate by tool-wear band", "Bars: failure rate · dotted line: machines in band")
        show_chart(chart_rate_by_bin(df, "Tool wear [min]", [0, 50, 100, 150, 200, 1000], ["0–49", "50–99", "100–149", "150–199", "200+"],
                                      "Tool wear [min]"), key="an_wear")
with e2:
    with st.container(key="panel_wear2"):
        panel_title("Tool wear: normal vs failed")
        show_chart(chart_parameter_distribution(df, "Tool wear [min]", height=320), key="an_wear_box")

# ---- 4. Temperature and torque
section_header("Temperature & torque insights", "Thermal gradient and spindle load against failure rate", "thermostat")
f1, f2 = st.columns(2, gap="medium")
with f1:
    with st.container(key="panel_dt"):
        panel_title("Failure rate by temperature difference (ΔT)", "ΔT = process temperature − air temperature")
        show_chart(chart_rate_by_bin(df, "Temp_diff", [0, 8.6, 9.5, 10.5, 11.5, 100], ["< 8.6 K", "8.6–9.5", "9.5–10.5", "10.5–11.5", "≥ 11.5"],
                                      "ΔT [K]"), key="an_dt")
with f2:
    with st.container(key="panel_torque"):
        panel_title("Failure rate by torque band")
        show_chart(chart_rate_by_bin(df, "Torque [Nm]", [0, 20, 30, 40, 50, 60, 1000], ["< 20", "20–30", "30–40", "40–50", "50–60", "60+"],
                                      "Torque [Nm]"), key="an_torque")

# ---- 5. Explorer
section_header("Parameter explorer", "Compare any parameter between normal and failed machines", "query_stats")
params = ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]", "Temp_diff", "Power_kW", "Strain", "Process temperature [K]", "Air temperature [K]"]
x1, x2 = st.columns([1, 1.2], gap="medium")
with x1:
    with st.container(key="panel_explorer"):
        param = st.selectbox("Parameter", params, key="an_param")
        show_chart(chart_parameter_distribution(df, param, height=300), key="an_box")
with x2:
    with st.container(key="panel_stats"):
        panel_title("Mean values: normal vs failed")
        cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Temp_diff", "Power_kW", "Strain"]
        stats = df.groupby("Failure_Status")[cols].mean().T.round(2)
        stats = stats[["Normal", "Failure"]]
        stats.columns = ["Normal (mean)", "Failure (mean)"]
        stats["Difference"] = (stats["Failure (mean)"] - stats["Normal (mean)"]).round(2)
        show_table(stats)

# ---- 6. Observations
section_header("Analytical observations", "Calculated live from the dataset — nothing is hard-coded", "lightbulb")
obs = [
    ("build", "Tool wear", f"Machines with ≥ 200 min of tool wear fail at {rate(wear_hi):.1f}% versus {rate(wear_lo):.1f}% below 200 min. "
                           f"Failed machines average {fail['Tool wear [min]'].mean():.0f} min of wear against {ok['Tool wear [min]'].mean():.0f} min for normal ones."),
    ("speed", "Torque", f"Torque above 50 Nm shows a {rate(torque_hi):.1f}% failure rate compared with {rate(torque_lo):.1f}% at or below 50 Nm. "
                        f"Mean torque is {fail['Torque [Nm]'].mean():.1f} Nm (failed) vs {ok['Torque [Nm]'].mean():.1f} Nm (normal)."),
    ("thermostat", "Thermal gradient", f"Records with ΔT below 8.6 K fail at {rate(dt_low):.1f}% versus {rate(dt_ok):.1f}% above it "
                                       f"({len(dt_low):,} machines in the low-ΔT group)."),
    ("category", "Machine type", " · ".join(f"Type {t}: {r:.2f}%" for t, r in rates_type.items()) + " failure rate."),
    ("rotate_right", "Rotational speed", f"Failed machines run at {fail['Rotational speed [rpm]'].mean():.0f} rpm on average versus {ok['Rotational speed [rpm]'].mean():.0f} rpm for normal machines."),
    ("bolt", "Power", f"Mean spindle power is {fail['Power_kW'].mean():.2f} kW for failed machines and {ok['Power_kW'].mean():.2f} kW for normal machines."),
]
insight_grid(obs)
