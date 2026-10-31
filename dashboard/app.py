"""
AI-Based Industrial Maintenance and Process Optimization System
Industrial AI — Predictive Maintenance Platform (Streamlit application shell)
Author / Lead Developer: Adithyan B (Team Lead)
Infosys Springboard Virtual Internship 7.0

Run from the project root:
    streamlit run dashboard/app.py

This file only provides the application frame (page config, design system, custom
sidebar and navigation). Each page lives in dashboard/views/.
"""

from pathlib import Path
import sys

import streamlit as st

# Ensure repository root is importable (dashboard.*, src.*)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# MUST be the first Streamlit call
st.set_page_config(
    page_title="Industrial AI | Predictive Maintenance Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="auto",
)

from dashboard.components import (  # noqa: E402
    load_benchmark_dataset,
    load_comparison_data,
    md,
    model_is_available,
)
from dashboard.styles import DASHBOARD_VERSION, apply_custom_css  # noqa: E402

apply_custom_css()

# ---- Session state -----------------------------------------------------------
st.session_state.setdefault("session_history", [])
st.session_state.setdefault("batch_results", None)
st.session_state.setdefault("batch_meta", None)
st.session_state.setdefault("last_prediction", None)

# ---- Pages -------------------------------------------------------------------
PAGES = [
    st.Page("views/overview.py", title="Command Center", icon=":material/space_dashboard:", default=True),
    st.Page("views/prediction.py", title="Failure Prediction", icon=":material/precision_manufacturing:", url_path="prediction"),
    st.Page("views/batch.py", title="Batch CSV Analysis", icon=":material/upload_file:", url_path="batch"),
    st.Page("views/analytics.py", title="Failure Analytics", icon=":material/monitoring:", url_path="analytics"),
    st.Page("views/performance.py", title="Model Performance", icon=":material/model_training:", url_path="performance"),
    st.Page("views/maintenance.py", title="Maintenance Insights", icon=":material/build_circle:", url_path="maintenance"),
]
current_page = st.navigation(PAGES, position="hidden")

# ---- Sidebar -----------------------------------------------------------------
LOGO_SVG = """
<svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="16" cy="16" r="11.2" stroke="#22D3EE" stroke-width="3.4" stroke-dasharray="3.1 2.46"/>
  <circle cx="16" cy="16" r="8.4" stroke="#22D3EE" stroke-width="1.6"/>
  <path d="M8.6 16.4h4l1.7-4.4 2.6 8 1.9-3.6h4.6" stroke="#fff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

df_side = load_benchmark_dataset()
_, meta_side = load_comparison_data()
model_ok, model_err = model_is_available()
model_name = meta_side.get("model_name", "Trained model")
recall_side = meta_side.get("metrics", {}).get("recall_failure")
auc_side = meta_side.get("metrics", {}).get("roc_auc")

with st.sidebar:
    md(f"""
    <div class="brand">
        <div class="brand-mark">{LOGO_SVG}</div>
        <div>
            <div class="brand-name">INDUSTRIAL <b>AI</b></div>
            <div class="brand-sub">Predictive Maintenance Platform</div>
        </div>
    </div>
    <div class="nav-label">Workspace</div>
    """)

    for p in PAGES:
        st.page_link(p)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    md('<div class="nav-label">System status</div>')

    # Model status — derived from an actual load of models/best_model.pkl
    if model_ok:
        extra = ""
        if recall_side is not None and auc_side is not None:
            extra = f"Recall {recall_side * 100:.1f}% · AUC {auc_side:.3f}"
        md(f"""
        <div class="side-card">
            <div class="side-row"><span class="side-title">Model</span>
            <span class="side-val"><span class="dot ok"></span>Online</span></div>
            <div class="side-sub">{model_name}<br>{extra}</div>
        </div>
        """)
    else:
        md(f"""
        <div class="side-card">
            <div class="side-row"><span class="side-title">Model</span>
            <span class="side-val"><span class="dot bad"></span>Offline</span></div>
            <div class="side-sub">Run <code>python src/predictive_model.py</code> to train.</div>
        </div>
        """)

    # Dataset status — derived from the actual dataset file
    if not df_side.empty:
        md(f"""
        <div class="side-card">
            <div class="side-row"><span class="side-title">Dataset</span>
            <span class="side-val"><span class="dot ok"></span>Loaded</span></div>
            <div class="side-sub">AI4I 2020 · {len(df_side):,} records<br>{df_side.shape[1]} columns</div>
        </div>
        """)
    else:
        md("""
        <div class="side-card">
            <div class="side-row"><span class="side-title">Dataset</span>
            <span class="side-val"><span class="dot bad"></span>Missing</span></div>
            <div class="side-sub">Expected at data/Dataset.csv</div>
        </div>
        """)

    n_hist = len(st.session_state.session_history)
    n_batch = len(st.session_state.batch_results) if st.session_state.batch_results is not None else 0
    md(f"""
    <div class="side-card">
        <div class="side-row"><span class="side-title">Session</span></div>
        <div class="side-sub">Predictions run: <b style="color:#fff">{n_hist}</b><br>Batch records scored: <b style="color:#fff">{n_batch:,}</b></div>
    </div>
    """)
    if st.button("Reset session", key="reset_session", icon=":material/restart_alt:"):
        for k in ("session_history", "batch_results", "batch_meta", "last_prediction"):
            st.session_state[k] = [] if k == "session_history" else None
        st.rerun()

    md(f"""
    <div class="side-foot">
        Dashboard v{DASHBOARD_VERSION}<br>
        Infosys Springboard Virtual Internship 7.0<br>
        Lead: Adithyan B
    </div>
    """)

current_page.run()
