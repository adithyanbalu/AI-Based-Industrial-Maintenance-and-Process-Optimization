"""Page: Command Center (overview). All figures are derived from the real dataset / model files."""

import pandas as pd
import streamlit as st

from dashboard.components import (
    COLORS,
    chart_failure_distribution,
    chart_failure_modes,
    chart_type_distribution,
    dataset_summary,
    insight_grid,
    kpi_grid,
    load_benchmark_dataset,
    load_comparison_data,
    md,
    model_is_available,
    panel_title,
    section_header,
    show_chart,
    show_table,
)

df = load_benchmark_dataset()
comp_df, meta = load_comparison_data()
summary = dataset_summary(df)
metrics = meta.get("metrics", {})
baseline = meta.get("baseline_comparison", {})
model_ok, _ = model_is_available()

if not summary:
    st.error("Dataset not found. Expected at data/Dataset.csv.")
    st.stop()

# ------------------------------------------------------------------ Hero
HERO_ART = """
<svg class="hero-art" width="420" height="420" viewBox="0 0 420 420" fill="none">
  <circle cx="210" cy="210" r="190" stroke="#22D3EE" stroke-opacity=".18" stroke-width="1"/>
  <circle cx="210" cy="210" r="150" stroke="#22D3EE" stroke-opacity=".14" stroke-width="1" stroke-dasharray="3 7"/>
  <circle cx="210" cy="210" r="112" stroke="#22D3EE" stroke-width="26" stroke-opacity=".16" stroke-dasharray="17 13.5"/>
  <circle cx="210" cy="210" r="92" stroke="#22D3EE" stroke-opacity=".5" stroke-width="2"/>
  <circle cx="210" cy="210" r="60" stroke="#3B82F6" stroke-opacity=".5" stroke-width="1.5"/>
  <path d="M120 214h46l18-48 30 92 22-42h54" stroke="#22D3EE" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" stroke-opacity=".9"/>
</svg>
"""
chips = [
    (f"● Model {'online' if model_ok else 'offline'}", "ok" if model_ok else "bad"),
    (f"{summary['total']:,} machine records", "cyan"),
    (meta.get("model_name", "Random Forest"), ""),
    ("AI4I 2020 benchmark", ""),
]
chip_html = "".join(f'<span class="chip {c}">{t}</span>' for t, c in chips)
md(f"""
<div class="hero">
    {HERO_ART}
    <div class="hero-eyebrow">Industrial AI · Command Center</div>
    <h1>Know which machine fails next — before it does.</h1>
    <p>A predictive-maintenance platform that reads spindle speed, torque, temperatures and tool wear,
    estimates the probability of machine failure with a physics-informed Random Forest, and turns it into
    clear maintenance suggestions for plant engineers.</p>
    <div class="chip-row">{chip_html}</div>
</div>
""")

# ------------------------------------------------------------------ KPIs
acc = metrics.get("accuracy")
rec = metrics.get("recall_failure")
kpis = [
    dict(title="Machine records", value=f"{summary['total']:,}", subtext="Records in dataset", icon_name="dns", accent=COLORS["brand_cyan"]),
    dict(title="Normal", value=f"{summary['normal']:,}", subtext=f"{100 - summary['fail_rate']:.2f}% of records", icon_name="check_circle",
         accent=COLORS["status_normal"], progress=100 - summary["fail_rate"]),
    dict(title="Failures", value=f"{summary['failures']:,}", subtext="Machine failure = 1", icon_name="warning", accent=COLORS["status_danger"]),
    dict(title="Failure rate", value=f"{summary['fail_rate']:.2f}", unit="%", subtext="Highly imbalanced classes", icon_name="percent",
         accent=COLORS["status_warning"], progress=summary["fail_rate"] * 5),
]
if acc is not None:
    kpis.append(dict(title="Model accuracy", value=f"{acc * 100:.2f}", unit="%", subtext="Hold-out test set (20%)", icon_name="verified",
                     accent=COLORS["accent_cyan"], progress=acc * 100))
if rec is not None:
    kpis.append(dict(title="Failure recall", value=f"{rec * 100:.1f}", unit="%", subtext=f"{metrics['confusion_matrix']['TP']} of "
                     f"{metrics['confusion_matrix']['TP'] + metrics['confusion_matrix']['FN']} failures caught" if "confusion_matrix" in metrics else "Failure class",
                     icon_name="crisis_alert", accent=COLORS["accent_violet"] if "accent_violet" in COLORS else COLORS["brand_blue"], progress=rec * 100))
kpi_grid(kpis)

# ------------------------------------------------------------------ Fleet health
section_header("Machine health overview", "Distribution of healthy and failed records across the whole dataset", "health_and_safety")
c1, c2, c3 = st.columns([1.05, 1, 1.1], gap="medium")

with c1:
    with st.container(key="panel_health"):
        panel_title("Fleet status", "Share of healthy vs failed records")
        show_chart(chart_failure_distribution(df, height=262), key="ov_donut")
        md(f"""
        <div class="chip-row" style="justify-content:center;margin-top:-4px;">
            <span class="chip ok">● Normal {summary['normal']:,}</span>
            <span class="chip bad">● Failure {summary['failures']:,}</span>
        </div>
        """)
with c2:
    with st.container(key="panel_type"):
        panel_title("Failure rate by machine type", "Quality variants L / M / H")
        show_chart(chart_type_distribution(df), key="ov_type")
with c3:
    with st.container(key="panel_modes"):
        panel_title("Failure mechanisms", "Recorded events per failure mode")
        show_chart(chart_failure_modes(df), key="ov_modes")

# Health by type — segmented bars
by_type = summary["by_type"].reindex(["L", "M", "H"]).dropna()
rows = ""
for t, r in by_type.iterrows():
    n, f = int(r["count"]), int(r["sum"])
    ok_pct = (n - f) / n * 100
    rows += f"""
    <div style="margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:6px;">
            <span style="color:#fff;font-weight:600;">Type {t}</span>
            <span class="num" style="color:{COLORS['text_secondary']};">{n - f:,} healthy · {f} failed · {ok_pct:.2f}% healthy</span>
        </div>
        <div style="display:flex;height:10px;border-radius:8px;overflow:hidden;background:rgba(148,163,184,.14);">
            <div style="width:{ok_pct:.2f}%;background:linear-gradient(90deg,#0aa57a,{COLORS['status_normal']});"></div>
            <div style="width:{100 - ok_pct:.2f}%;background:{COLORS['status_danger']};min-width:3px;"></div>
        </div>
    </div>"""
with st.container(key="panel_typehealth"):
    panel_title("Health by machine type", "Healthy (green) vs failed (red) share per variant")
    md(rows)

# ------------------------------------------------------------------ Insights (computed)
section_header("Key analytical insights", "Every statement below is calculated from the dataset and saved model results", "lightbulb")

items = []
fr = summary["fail_rate"]
items.append(("balance", "Severe class imbalance",
              f"Only {fr:.2f}% of records are failures. A model that always predicts “Normal” would still score {100 - fr:.2f}% accuracy while catching no failures."))

if not comp_df.empty and "Logistic Regression (Baseline)" in set(comp_df["Model"]):
    lr = comp_df[comp_df["Model"] == "Logistic Regression (Baseline)"].iloc[0]
    miss = int(lr["False Negatives (FN)"])
    tot = int(lr["False Negatives (FN)"] + lr["True Positives (TP)"])
    items.append(("report", "Accuracy is misleading",
                  f"Baseline Logistic Regression reaches {lr['Accuracy'] * 100:.2f}% accuracy but only {lr['Recall (Failure)'] * 100:.1f}% recall — it misses {miss} of {tot} failures."))

if baseline and metrics:
    items.append(("neurology", "Physics-informed features help",
                  f"Adding ΔT, mechanical power and an overstrain index lifted failure recall from {baseline['baseline_recall'] * 100:.1f}% to "
                  f"{metrics['recall_failure'] * 100:.1f}% and cut missed failures from {baseline['baseline_fn']} to {baseline['improved_fn']}."))

if not by_type.empty:
    rates = (by_type["sum"] / by_type["count"] * 100)
    hi, lo = rates.idxmax(), rates.idxmin()
    items.append(("category", f"Type {hi} fails most often",
                  f"Type {hi} machines fail at {rates[hi]:.2f}% versus {rates[lo]:.2f}% for Type {lo} ({rates[hi] / rates[lo]:.1f}× higher)."))

if summary["modes"]:
    top_mode = max(summary["modes"], key=summary["modes"].get)
    names = {"HDF": "heat dissipation", "OSF": "overstrain", "PWF": "power", "TWF": "tool wear", "RNF": "random"}
    items.append(("local_fire_department", "Dominant failure mechanism",
                  f"{names.get(top_mode, top_mode).capitalize()} failures ({top_mode}) are the most frequent with {summary['modes'][top_mode]} recorded events."))

items.append(("lock", "No target leakage",
              "Failure-mode flags (TWF, HDF, PWF, OSF, RNF) are excluded from model inputs, so predictions use only live sensor readings."))
insight_grid(items)

# ------------------------------------------------------------------ Session activity
section_header("Recent predictions this session", "Manual predictions appear here after you run them", "history")
hist = st.session_state.get("session_history", [])
if hist:
    h = pd.DataFrame(hist)
    cols = [c for c in ["Timestamp", "Type", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Status", "Failure_Probability", "Risk_Level", "Diagnostic_Notes"] if c in h.columns]
    show_table(h[cols], hide_index=True)
else:
    md("""
    <div class="callout"><div class="callout-title">No predictions yet</div>
    Open <b>Failure Prediction</b> to assess a single machine, or <b>Batch CSV Analysis</b> to score a whole fleet file.</div>
    """)
    qa, qb, qc = st.columns(3)
    try:
        qa.page_link("views/prediction.py", label="Predict a machine", icon=":material/precision_manufacturing:")
        qb.page_link("views/batch.py", label="Upload a CSV", icon=":material/upload_file:")
        qc.page_link("views/maintenance.py", label="Maintenance insights", icon=":material/build_circle:")
    except Exception:
        pass
