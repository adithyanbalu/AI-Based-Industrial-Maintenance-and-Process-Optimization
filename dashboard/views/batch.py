"""Page: Batch CSV Analysis. Uses validate_machine_data + src.prediction.predict_machines (unchanged)."""

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.components import (
    COLORS,
    MODEL_FEATURES,
    chart_batch_distribution,
    chart_batch_risk_tiers,
    chart_batch_scatter,
    esc,
    get_data_validator,
    icon,
    kpi_grid,
    md,
    page_header,
    panel_title,
    section_header,
    show_chart,
    show_table,
    stepper,
)
from src.prediction import InvalidInputError, ModelLoadError, predict_machines

validate_machine_data = get_data_validator()

page_header(
    "Fleet telemetry",
    "Batch CSV Analysis",
    "Upload a CSV of machine readings, check it, score every row with the trained model and download the enriched results.",
    chips=[("6 required columns", "cyan"), ("Missing values imputed safely", "")],
)

STEPS = [("Upload", "Select a CSV file"), ("Validate", "Schema & value checks"), ("Predict", "Score every machine"), ("Export", "Download results")]

# ------------------------------------------------------------------ Template + upload
sample_df = pd.DataFrame([
    {"Type": "M", "Air temperature [K]": 298.1, "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8, "Tool wear [min]": 15},
    {"Type": "L", "Air temperature [K]": 302.5, "Process temperature [K]": 310.2, "Rotational speed [rpm]": 1350, "Torque [Nm]": 64.0, "Tool wear [min]": 220},
    {"Type": "H", "Air temperature [K]": 298.9, "Process temperature [K]": 309.1, "Rotational speed [rpm]": 1420, "Torque [Nm]": 45.2, "Tool wear [min]": 85},
    {"Type": "L", "Air temperature [K]": 303.8, "Process temperature [K]": 311.5, "Rotational speed [rpm]": 1320, "Torque [Nm]": 52.0, "Tool wear [min]": 90},
    {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0, "Rotational speed [rpm]": 2200, "Torque [Nm]": 55.0, "Tool wear [min]": 180},
])

uploaded = st.file_uploader("Drop a machine telemetry CSV here", type=["csv"], key="batch_uploader", label_visibility="collapsed")

# Parse the file
df_upload, parse_error = None, None
if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
    except Exception as exc:  # unreadable / not CSV
        parse_error = str(exc)

# Reset stale results when a different file is chosen
file_sig = (uploaded.name, uploaded.size) if uploaded is not None else None
meta = st.session_state.get("batch_meta")
if meta and meta.get("sig") != file_sig:
    st.session_state.batch_results = None
    st.session_state.batch_meta = None
    meta = None

results = st.session_state.get("batch_results")
if df_upload is None:
    stepper(STEPS, 0)
elif results is None:
    stepper(STEPS, 2)
else:
    stepper(STEPS, 3)

if uploaded is None:
    c_a, c_b = st.columns([2.2, 1], gap="large")
    with c_a:
        with st.container(key="panel_format"):
            panel_title("Expected CSV format", "Extra columns are kept in the output and ignored by the model")
            md('<div class="chip-row">' + "".join(f'<span class="chip cyan">{esc(c)}</span>' for c in MODEL_FEATURES) + "</div>")
            md('<div class="meter-note" style="margin-top:10px;">Type must be L, M or H. Numeric values should be within plausible physical ranges; '
               "missing or out-of-range values are imputed or clipped automatically.</div>")
    with c_b:
        with st.container(key="panel_template"):
            panel_title("Need an example?")
            buf = io.StringIO()
            sample_df.to_csv(buf, index=False)
            st.download_button("Download CSV template", buf.getvalue(), "machine_telemetry_template.csv", "text/csv",
                               icon=":material/download:", key="dl_template")
    st.stop()

if parse_error:
    md(f'<div class="callout danger"><div class="callout-title">{icon("error")}Could not read the file</div>{esc(parse_error)}</div>')
    st.stop()

# ------------------------------------------------------------------ File info + validation
section_header("File & validation", "Checked before any prediction is made", "fact_check")

missing_cols = [c for c in MODEL_FEATURES if c not in df_upload.columns]
try:
    is_valid, error_list, _validated = validate_machine_data(df_upload)
except Exception as exc:  # validator must never crash the page
    is_valid, error_list = False, [f"Validator error: {exc}"]

size_kb = uploaded.size / 1024
kpi_grid([
    dict(title="File", value=f'<span style="font-size:1.15rem">{esc(uploaded.name)}</span>', subtext=f"{size_kb:,.1f} KB", icon_name="description", accent=COLORS["brand_cyan"]),
    dict(title="Records", value=f"{len(df_upload):,}", subtext="Rows detected", icon_name="table_rows", accent=COLORS["accent_cyan"]),
    dict(title="Columns", value=f"{df_upload.shape[1]}", subtext=f"{len(MODEL_FEATURES) - len(missing_cols)} of {len(MODEL_FEATURES)} required found", icon_name="view_column", accent=COLORS["brand_blue"]),
    dict(title="Missing cells", value=f"{int(df_upload.isnull().sum().sum()):,}", subtext="Imputed if present", icon_name="grid_off",
         accent=COLORS["status_warning"] if df_upload.isnull().sum().sum() else COLORS["status_normal"]),
])

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
col_chips = "".join(
    f'<span class="chip {"ok" if c in df_upload.columns else "bad"}">{"✓" if c in df_upload.columns else "✕"} {esc(c)}</span>' for c in MODEL_FEATURES
)
if missing_cols:
    md(f"""
    <div class="callout danger"><div class="callout-title">{icon('error')}Validation failed — required columns are missing</div>
    <div class="chip-row" style="margin:8px 0;">{col_chips}</div>
    Add the missing column(s) <b>{esc(', '.join(missing_cols))}</b> and upload the file again.</div>
    """)
elif is_valid:
    md(f"""
    <div class="callout success"><div class="callout-title">{icon('task_alt')}CSV validation passed</div>
    <div class="chip-row" style="margin:8px 0;">{col_chips}</div>
    All required columns are present and all values are inside plausible physical ranges.</div>
    """)
else:
    items = "".join(f"<li>{esc(e)}</li>" for e in error_list)
    md(f"""
    <div class="callout warning"><div class="callout-title">{icon('warning')}Validation warnings</div>
    <div class="chip-row" style="margin:8px 0;">{col_chips}</div>
    <ul style="margin:4px 0 6px 18px;padding:0;">{items}</ul>
    <b>Safe imputation is enabled</b> — missing values are filled with nominal defaults and out-of-range values are clipped, so every row can be scored.</div>
    """)

with st.expander("Preview uploaded data (first 10 rows)", expanded=False, icon=":material/table_view:"):
    show_table(df_upload.head(10), hide_index=True)

# ------------------------------------------------------------------ Predict
section_header("Run predictions", "Scores every record with the saved model", "play_circle")
run = st.button("RUN BATCH PREDICTION", type="primary", icon=":material/bolt:", key="run_batch", disabled=bool(missing_cols))
if run:
    try:
        with st.spinner(f"Scoring {len(df_upload):,} machines..."):
            res = predict_machines(df_upload, allow_imputation=True)
        st.session_state.batch_results = res
        st.session_state.batch_meta = {"sig": file_sig, "name": uploaded.name, "time": datetime.now().strftime("%Y%m%d_%H%M%S")}
        st.rerun()
    except (InvalidInputError, ModelLoadError) as exc:
        st.error(f"Prediction failed: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error while scoring the file: {exc}")

# ------------------------------------------------------------------ Results
results = st.session_state.get("batch_results")
if results is not None:
    total = len(results)
    fails = int((results["Prediction"] == 1).sum())
    normal = total - fails
    mean_p = float(results["Failure_Probability"].mean()) * 100
    high = int((results["Risk_Level"] == "High Risk / Critical").sum()) if "Risk_Level" in results.columns else 0

    section_header("Batch results", "Summary of the scored fleet", "analytics")
    kpi_grid([
        dict(title="Processed records", value=f"{total:,}", subtext="Machines scored", icon_name="dns", accent=COLORS["brand_cyan"]),
        dict(title="Normal predictions", value=f"{normal:,}", subtext=f"{normal / total * 100:.1f}% of fleet", icon_name="check_circle",
             accent=COLORS["status_normal"], progress=normal / total * 100),
        dict(title="Failure predictions", value=f"{fails:,}", subtext=f"{fails / total * 100:.1f}% of fleet", icon_name="warning",
             accent=COLORS["status_danger"] if fails else COLORS["status_normal"], progress=fails / total * 100),
        dict(title="Mean failure probability", value=f"{mean_p:.2f}", unit="%", subtext=f"{high:,} machines in high-risk tier", icon_name="speed",
             accent=COLORS["status_warning"]),
    ])
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    g1, g2, g3 = st.columns([1, 1, 1.25], gap="medium")
    with g1:
        with st.container(key="panel_bdist"):
            panel_title("Prediction distribution")
            show_chart(chart_batch_distribution(results), key="b_dist")
    with g2:
        with st.container(key="panel_brisk"):
            panel_title("Risk tiers")
            show_chart(chart_batch_risk_tiers(results), key="b_risk")
    with g3:
        with st.container(key="panel_bscatter"):
            panel_title("Operating envelope", "Predicted failures highlighted")
            show_chart(chart_batch_scatter(results), key="b_scatter")

    section_header("Results table", "Original columns plus model output", "table_chart")
    only_fail = st.toggle("Show predicted failures only", value=False, key="batch_only_fail")
    view = results[results["Prediction"] == 1] if only_fail else results
    show_cols = [c for c in ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
                             "Tool wear [min]", "Status", "Failure_Probability", "Risk_Level", "Diagnostic_Notes"] if c in view.columns]
    cfg = {"Failure_Probability": st.column_config.ProgressColumn("Failure probability", min_value=0.0, max_value=1.0, format="%.3f")}
    show_table(view[show_cols], hide_index=True, column_config=cfg, height=420)
    st.caption(f"Showing {len(view):,} of {total:,} records.")

    out = io.StringIO()
    results.to_csv(out, index=False)
    stamp = (st.session_state.get("batch_meta") or {}).get("time", datetime.now().strftime("%Y%m%d_%H%M%S"))
    st.download_button("DOWNLOAD ENRICHED RESULTS (CSV)", out.getvalue(), f"batch_predictions_{stamp}.csv", "text/csv",
                       type="primary", icon=":material/download:", key="dl_results")
