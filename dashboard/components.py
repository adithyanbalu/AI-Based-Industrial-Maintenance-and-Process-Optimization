"""
AI-Based Industrial Maintenance and Process Optimization System
Reusable Dashboard UI Components and Analytics Module
Module Owner: Adithyan B (Team Lead)
Infosys Springboard Virtual Internship 7.0
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path before any internal imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import html as _html

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.styles import COLORS, get_plotly_layout

DATA_PATH = ROOT / "data" / "Dataset.csv"
COMPARISON_CSV = ROOT / "outputs" / "model_comparison.csv"
METADATA_JSON = ROOT / "models" / "model_metadata.json"

try:
    from src.features import MODEL_FEATURES
except ImportError:
    try:
        from features import MODEL_FEATURES
    except ImportError:
        MODEL_FEATURES = [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

# ==============================================================================
# INTEGRATION INTERFACE WRAPPERS (FALLBACK / TEAMMATE COMPATIBILITY)
# ==============================================================================

def get_data_validator():
    """
    Uses the project's data_validation module
    and adapts its output for dashboard pages.
    """
    from src.data_validation import validate_records

    def _wrapped_validator(df, *args, **kwargs):
        valid_df, errors = validate_records(df, allow_imputation=kwargs.get("allow_imputation", False))
        is_valid = not errors and valid_df is not None and len(valid_df) == len(df)
        return is_valid, errors, valid_df

    return _wrapped_validator


def get_maintenance_advisor():
    """
    Coordinates with Dheepesh's module (src/maintenance_advisor.py).
    If present, imports generate_maintenance_guidance. Otherwise falls back cleanly.
    """
    fn = None
    try:
        from src.maintenance_advisor import generate_maintenance_guidance
        fn = generate_maintenance_guidance
    except ImportError:
        try:
            from maintenance_advisor import generate_maintenance_guidance
            fn = generate_maintenance_guidance
        except ImportError:
            pass

    if fn is not None:
        def _wrapped_advisor(readings):
            try:
                return fn(readings)
            except Exception:
                return fallback_generate_maintenance_guidance(readings)

        return _wrapped_advisor

    return fallback_generate_maintenance_guidance


def fallback_generate_maintenance_guidance(
    machine_readings: Union[pd.DataFrame, Dict[str, Any], pd.Series],
) -> List[Dict[str, Any]]:
    """
    Rule-based transparent maintenance recommendation engine.
    Derives actionable advisory guidance using AI4I physical failure boundaries:
    1. Tool Wear Failure (TWF): Tool wear [min] >= 200 min
    2. Heat Dissipation Failure (HDF): dT < 8.6 K & Speed < 1380 rpm
    3. Power Failure (PWF): Mechanical Power < 3.5 kW or > 9.0 kW
    4. Overstrain Failure (OSF): Torque * Tool wear exceeding safety limits
    5. Routine Maintenance: Nominal parameters
    """
    if isinstance(machine_readings, pd.DataFrame):
        row = machine_readings.iloc[0].to_dict()
    elif isinstance(machine_readings, pd.Series):
        row = machine_readings.to_dict()
    else:
        row = dict(machine_readings)

    def _safe_float(k: str, default: float) -> float:
        val = row.get(k)
        if val is None or pd.isna(val):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    type_ = str(row.get("Type", "M") or "M").strip().upper()
    if type_ not in {"L", "M", "H"}:
        type_ = "M"

    air_t = _safe_float("Air temperature [K]", 300.0)
    proc_t = _safe_float("Process temperature [K]", 310.0)
    speed = _safe_float("Rotational speed [rpm]", 1500.0)
    torque = _safe_float("Torque [Nm]", 40.0)
    tool_wear = _safe_float("Tool wear [min]", 100.0)

    delta_t = proc_t - air_t
    power_w = speed * torque * (2.0 * np.pi / 60.0)
    strain = torque * tool_wear

    # Strain thresholds from benchmark specification
    strain_thresholds = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
    max_strain = strain_thresholds.get(type_, 12000.0)

    recommendations = []

    # 1. Tool Wear Rule
    if tool_wear >= 200:
        recommendations.append({
            "category": "Tooling System",
            "priority": "CRITICAL / URGENT",
            "condition": f"Tool wear is {tool_wear:.0f} min (Threshold >= 200 min)",
            "mechanism": "Tool Wear Failure (TWF) risk",
            "action": "Immediately schedule cutting tool insert replacement. Do not start high-tolerance batch jobs without tool inspection.",
            "color": COLORS["status_danger"],
        })
    elif tool_wear >= 160:
        recommendations.append({
            "category": "Tooling System",
            "priority": "WARNING / SCHEDULED",
            "condition": f"Tool wear is {tool_wear:.0f} min (Approaching 200 min limit)",
            "mechanism": "Cumulative Tool Degradation",
            "action": "Plan tool swap during the next maintenance window. Measure workpiece dimensional variance.",
            "color": COLORS["status_warning"],
        })

    # 2. Thermal / Heat Dissipation Rule
    if delta_t < 8.6 and speed < 1380:
        recommendations.append({
            "category": "Thermal & Cooling",
            "priority": "CRITICAL / URGENT",
            "condition": f"Temp difference is {delta_t:.1f} K (< 8.6 K) at {speed:.0f} rpm (< 1380 rpm)",
            "mechanism": "Heat Dissipation Failure (HDF) risk",
            "action": "Inspect cooling radiator fans and fluid heat exchanger. Check for coolant particulate clogging or pump degradation.",
            "color": COLORS["status_danger"],
        })
    elif proc_t >= 313.0:
        recommendations.append({
            "category": "Thermal & Cooling",
            "priority": "WARNING / SCHEDULED",
            "condition": f"Process temperature is elevated at {proc_t:.1f} K",
            "mechanism": "Thermal Overload",
            "action": "Verify coolant flow rates and clean air intake filters on spindle enclosure.",
            "color": COLORS["status_warning"],
        })

    # 3. Power Transmission Rule
    if power_w > 9000:
        recommendations.append({
            "category": "Drive & Electrical",
            "priority": "CRITICAL / URGENT",
            "condition": f"Mechanical power is {power_w/1000:.2f} kW (Upper Threshold > 9.0 kW)",
            "mechanism": "Power Failure (PWF) Overload",
            "action": "Reduce spindle feed rate or depth of cut immediately. Check drive inverter current draw and spindle bearings for binding.",
            "color": COLORS["status_danger"],
        })
    elif power_w < 3500 and speed < 1250:
        recommendations.append({
            "category": "Drive & Electrical",
            "priority": "WARNING / SCHEDULED",
            "condition": f"Mechanical power is {power_w/1000:.2f} kW (< 3.5 kW) at low speed",
            "mechanism": "Drive Under-Power / Stall Hazard",
            "action": "Check belt tension, VFD drive parameters, and electrical supply phase balance.",
            "color": COLORS["status_warning"],
        })

    # 4. Overstrain Rule
    if strain > max_strain:
        recommendations.append({
            "category": "Mechanical Structure",
            "priority": "CRITICAL / URGENT",
            "condition": f"Overstrain product (Torque × Wear) is {strain:.0f} (Exceeds {type_} limit of {max_strain:.0f})",
            "mechanism": "Overstrain Failure (OSF) Fracture Hazard",
            "action": "High torque combined with worn tooling causes extreme mechanical stress. Replace tool and reduce cutting speed to avoid catastrophic workpiece/tool breakage.",
            "color": COLORS["status_danger"],
        })

    # 5. Nominal condition
    if not recommendations:
        recommendations.append({
            "category": "Nominal Health",
            "priority": "ROUTINE",
            "condition": "All thermal, mechanical, and tooling parameters are within nominal operational limits.",
            "mechanism": "Healthy Machine Operation",
            "action": "Continue standard preventive maintenance schedule and continuous sensor logging.",
            "color": COLORS["status_normal"],
        })

    return recommendations


# ==============================================================================
# DATA LOADERS & CACHING
# ==============================================================================

@st.cache_data
def load_benchmark_dataset() -> pd.DataFrame:
    """Loads and caches the AI4I 2020 Predictive Maintenance benchmark dataset."""
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at '{DATA_PATH}'.")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    # Add domain features for analytics
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power_kW"] = (df["Rotational speed [rpm]"] * df["Torque [Nm]"] * (2 * np.pi / 60)) / 1000.0
    df["Strain"] = df["Torque [Nm]"] * df["Tool wear [min]"]
    df["Failure_Status"] = np.where(df["Machine failure"] == 1, "Failure", "Normal")
    return df


@st.cache_data
def load_comparison_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads saved model comparison metrics and metadata."""
    comp_df = pd.DataFrame()
    if COMPARISON_CSV.exists():
        comp_df = pd.read_csv(COMPARISON_CSV)

    meta = {}
    if METADATA_JSON.exists():
        try:
            meta = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass

    return comp_df, meta



# ==============================================================================
# UI PRIMITIVES (HTML building blocks used by every page)
# ==============================================================================

from dashboard.styles import CHART_PALETTE  # noqa: E402

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# Physical / rule thresholds (identical to the rule-based advisor and prediction module)
TOOL_WEAR_LIMIT = 200.0
TOOL_WEAR_WARN = 160.0
DELTA_T_LIMIT = 8.6
POWER_MAX_KW = 9.0
POWER_MIN_KW = 3.5
STRAIN_LIMITS = {"L": 11000.0, "M": 12000.0, "H": 13000.0}


def _h(s: str) -> str:
    """Strips indentation/blank lines so Markdown never turns HTML into a code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())


def md(s: str) -> None:
    """Renders an HTML snippet."""
    st.markdown(_h(s), unsafe_allow_html=True)


def esc(value: Any) -> str:
    """HTML-escapes untrusted text (file names, notes)."""
    return _html.escape(str(value))


def icon(name: str) -> str:
    """Material Symbols glyph (font is bundled with Streamlit)."""
    return f'<span class="material-symbols-rounded">{name}</span>'


def show_chart(fig: go.Figure, key: Optional[str] = None) -> None:
    """Version-tolerant st.plotly_chart (new `width` API with fallback)."""
    try:
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    """Version-tolerant st.dataframe."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def page_header(eyebrow: str, title: str, description: str, chips: Optional[List[Tuple[str, str]]] = None) -> None:
    chip_html = ""
    if chips:
        chip_html = '<div class="chip-row" style="margin-top:12px;">' + "".join(
            f'<span class="chip {cls}">{txt}</span>' for txt, cls in chips
        ) + "</div>"
    md(f"""
    <div class="page-head">
        <div>
            <div class="page-eyebrow">{eyebrow}</div>
            <div class="page-title">{title}</div>
            <div class="page-desc">{description}</div>
            {chip_html}
        </div>
    </div>
    """)


def section_header(title: str, subtitle: str = "", icon_name: str = "insights") -> None:
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    md(f"""
    <div class="sec-head">
        <div class="sec-ico">{icon(icon_name)}</div>
        <div><div class="sec-title">{title}</div>{sub}</div>
        <div class="sec-line"></div>
    </div>
    """)


def panel_title(title: str, subtitle: str = "") -> None:
    sub = f'<div class="panel-sub">{subtitle}</div>' if subtitle else ""
    md(f'<div class="panel-title">{title}</div>{sub}')


def kpi_html(
    title: str,
    value: Any,
    subtext: str = "",
    icon_name: Optional[str] = None,
    accent: Optional[str] = None,
    progress: Optional[float] = None,
    unit: str = "",
) -> str:
    style = f' style="--kpi-accent:{accent};"' if accent else ""
    ico = f'<span class="kpi-ico">{icon(icon_name)}</span>' if icon_name else ""
    unit_html = f"<small>{unit}</small>" if unit else ""
    bar = ""
    if progress is not None:
        pct = max(0.0, min(100.0, float(progress)))
        bar = f'<div class="kpi-bar"><span style="width:{pct:.1f}%"></span></div>'
    return (
        f'<div class="kpi-container"{style}>'
        f'<div class="kpi-top"><div class="kpi-label">{title}</div>{ico}</div>'
        f'<div class="kpi-value">{value}{unit_html}</div>'
        f'<div class="kpi-subtext">{subtext}</div>{bar}</div>'
    )


def kpi_grid(items: List[Dict[str, Any]]) -> None:
    """Renders a responsive grid of KPI tiles from a list of kwargs dicts."""
    st.markdown(
        f'<div class="kpi-grid" style="--n:{len(items)};">' + "".join(kpi_html(**it) for it in items) + "</div>",
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: Any, subtext: str = "", badge_text: str = "", badge_type: str = "normal"):
    """Backwards-compatible single KPI tile."""
    accent = {"normal": COLORS["status_normal"], "danger": COLORS["status_danger"],
              "warning": COLORS["status_warning"]}.get(badge_type, COLORS["brand_cyan"])
    sub = f"{subtext}" + (f" · {badge_text}" if badge_text else "")
    st.markdown(kpi_html(title, value, sub, accent=accent), unsafe_allow_html=True)


def render_alert_box(title: str, message: str, level: str = "info"):
    """Coloured callout. level: info | success | warning | danger."""
    ico = {"info": "info", "success": "task_alt", "warning": "warning", "danger": "error"}.get(level, "info")
    md(f"""
    <div class="callout {level}">
        <div class="callout-title">{icon(ico)}{title}</div>
        <div>{message}</div>
    </div>
    """)


def insight_grid(items: List[Tuple[str, str, str]]) -> None:
    """items: (material icon, title, description)."""
    cards = "".join(
        f'<div class="insight"><div class="insight-ico">{icon(i)}</div>'
        f'<div><div class="insight-t">{t}</div><div class="insight-d">{d}</div></div></div>'
        for i, t, d in items
    )
    st.markdown(f'<div class="insight-grid">{cards}</div>', unsafe_allow_html=True)


def meter(name: str, value: float, max_value: float, display: str, color: str,
          marker: Optional[float] = None, note: str = "") -> str:
    pct = max(0.0, min(100.0, value / max_value * 100.0)) if max_value else 0.0
    mark = ""
    if marker is not None:
        mark = f'<div class="meter-mark" style="left:{max(0.0, min(100.0, marker / max_value * 100.0)):.1f}%"></div>'
    note_html = f'<div class="meter-note">{note}</div>' if note else ""
    return (
        f'<div class="meter"><div class="meter-top"><span class="meter-name">{name}</span>'
        f'<span class="meter-val" style="color:{color}">{display}</span></div>'
        f'<div class="meter-track"><div class="meter-fill" style="width:{pct:.1f}%;background:{color}"></div>{mark}</div>'
        f"{note_html}</div>"
    )


def kv_table(pairs: List[Tuple[str, str]]) -> None:
    st.markdown(
        '<div class="kv">' + "".join(f"<div>{k}</div><div>{v}</div>" for k, v in pairs) + "</div>",
        unsafe_allow_html=True,
    )


def priority_badge_class(priority: str) -> str:
    p = str(priority).upper()
    if "CRITICAL" in p or "URGENT" in p:
        return "badge-danger"
    if "WARNING" in p or "SCHEDULED" in p:
        return "badge-warning"
    return "badge-normal"


def render_recommendation(rec: Dict[str, Any]) -> None:
    """Maintenance recommendation card (works with both internal and teammate advisor schemas)."""
    color = rec.get("color", COLORS["brand_cyan"])
    cat = esc(rec.get("category", "Maintenance"))
    mech = esc(rec.get("mechanism", ""))
    pr = esc(rec.get("priority", "ADVISORY"))
    title = f"{cat} — {mech}" if mech else cat
    md(f"""
    <div class="rec-card" style="--rc:{color};">
        <div class="rec-head">
            <span class="rec-title">{title}</span>
            <span class="badge {priority_badge_class(pr)}">{pr}</span>
        </div>
        <div class="rec-line"><b>Condition observed:</b> {esc(rec.get('condition', ''))}</div>
        <div class="rec-line"><b>Suggested action:</b> {esc(rec.get('action', ''))}</div>
    </div>
    """)


def stepper(steps: List[Tuple[str, str]], current: int) -> None:
    """Workflow stepper. `current` is the 0-based active step; earlier steps are done."""
    out = []
    for i, (t, d) in enumerate(steps):
        cls = "done" if i < current else ("active" if i == current else "")
        n = icon("check") if i < current else str(i + 1)
        out.append(f'<div class="step {cls}"><div class="step-n">{n}</div><div><div class="step-t">{t}</div><div class="step-d">{d}</div></div></div>')
    st.markdown('<div class="stepper">' + "".join(out) + "</div>", unsafe_allow_html=True)


def compute_machine_metrics(readings: Dict[str, Any]) -> Dict[str, float]:
    """Derived physical indicators used for the live gauges (same formulas as src.features)."""
    air = float(readings["Air temperature [K]"])
    proc = float(readings["Process temperature [K]"])
    speed = float(readings["Rotational speed [rpm]"])
    torque = float(readings["Torque [Nm]"])
    wear = float(readings["Tool wear [min]"])
    typ = str(readings.get("Type", "M")).upper()
    return {
        "delta_t": proc - air,
        "power_kw": speed * torque * (2.0 * np.pi / 60.0) / 1000.0,
        "strain": torque * wear,
        "strain_limit": STRAIN_LIMITS.get(typ, 12000.0),
        "speed": speed, "torque": torque, "wear": wear, "air": air, "proc": proc,
    }


def risk_color(risk_level: str) -> str:
    if "High" in risk_level:
        return COLORS["status_danger"]
    if "Medium" in risk_level or "Warning" in risk_level:
        return COLORS["status_warning"]
    return COLORS["status_normal"]


# ==============================================================================
# DATA HELPERS (all numbers displayed in the UI are derived here from real files)
# ==============================================================================

def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Counts and rates computed from the actual dataset."""
    if df.empty:
        return {}
    total = len(df)
    fail = int((df["Machine failure"] == 1).sum())
    by_type = df.groupby("Type")["Machine failure"].agg(["count", "sum"])
    modes = {m: int(df[m].sum()) for m in ["TWF", "HDF", "PWF", "OSF", "RNF"] if m in df.columns}
    return {
        "total": total, "failures": fail, "normal": total - fail,
        "fail_rate": fail / total * 100.0, "by_type": by_type, "modes": modes,
    }


def model_is_available() -> Tuple[bool, str]:
    """Checks that the saved model can really be loaded (used for the sidebar status)."""
    try:
        from src.prediction import load_prediction_model
        load_prediction_model()
        return True, ""
    except Exception as exc:  # pragma: no cover - surfaced in UI
        return False, str(exc)


@st.cache_resource(show_spinner=False)
def compute_roc_curves() -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Real ROC curves: re-fits the project's own candidate pipelines (src.model_comparison)
    on the identical stratified split (random_state=42) and records the test-set ROC.
    Nothing is hard-coded; this mirrors outputs/model_roc_curves.png.
    """
    import warnings
    from sklearn.metrics import roc_curve, roc_auc_score
    from src.model_comparison import build_candidate_models, load_and_split_data

    warnings.filterwarnings("ignore")
    _, x_train, x_test, y_train, y_test = load_and_split_data()
    curves = {}
    for name, pipe in build_candidate_models().items():
        pipe.fit(x_train, y_train)
        if not hasattr(pipe, "predict_proba"):
            continue
        prob = pipe.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        curves[name] = (fpr, tpr, float(roc_auc_score(y_test, prob)))
    return curves


# ==============================================================================
# PLOTLY CHART BUILDERS
# ==============================================================================

def _empty_fig(message: str = "No data available", height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False))
    fig.add_annotation(text=message, showarrow=False, font=dict(color=COLORS["text_muted"], size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _status_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Failure_Status" not in out.columns:
        if "Machine failure" in out.columns:
            out["Failure_Status"] = np.where(out["Machine failure"] == 1, "Failure", "Normal")
        elif "Status" in out.columns:
            out["Failure_Status"] = out["Status"]
        else:
            out["Failure_Status"] = "Normal"
    return out


def chart_failure_distribution(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Donut: normal vs failure, with the healthy share printed in the centre."""
    if df.empty or "Machine failure" not in df.columns:
        return _empty_fig()
    counts = df["Machine failure"].value_counts()
    n_ok, n_bad = int(counts.get(0, 0)), int(counts.get(1, 0))
    total = n_ok + n_bad
    fig = go.Figure(go.Pie(
        labels=["Normal", "Failure"], values=[n_ok, n_bad], hole=0.74, sort=False, direction="clockwise",
        marker=dict(colors=[COLORS["status_normal"], COLORS["status_danger"]], line=dict(color=COLORS["bg_main"], width=3)),
        textinfo="none", hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False))
    fig.update_layout(annotations=[
        dict(text=f"<b>{n_ok / total * 100:.2f}%</b>", x=0.5, y=0.54, showarrow=False,
             font=dict(size=28, color="#FFFFFF", family="JetBrains Mono")),
        dict(text="HEALTHY", x=0.5, y=0.40, showarrow=False, font=dict(size=11, color=COLORS["text_muted"])),
    ])
    return fig


def chart_type_distribution(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Failure rate by product type (L / M / H)."""
    if df.empty or "Type" not in df.columns:
        return _empty_fig()
    st_ = df.groupby("Type")["Machine failure"].agg(["count", "sum"]).reindex(["L", "M", "H"]).dropna()
    st_["rate"] = st_["sum"] / st_["count"] * 100.0
    names = {"L": "Type L", "M": "Type M", "H": "Type H"}
    fig = go.Figure(go.Bar(
        x=[names[t] for t in st_.index], y=st_["rate"],
        marker=dict(color=[COLORS["status_danger"], COLORS["status_warning"], COLORS["accent_cyan"]][: len(st_)],
                    line=dict(width=0), cornerradius=8),
        text=[f"{r:.2f}%" for r in st_["rate"]], textposition="outside", cliponaxis=False,
        textfont=dict(color="#FFFFFF", size=12, family="JetBrains Mono"),
        customdata=np.stack([st_["sum"], st_["count"]], axis=-1),
        hovertemplate="<b>%{x}</b><br>Failure rate: %{y:.2f}%<br>%{customdata[0]:.0f} failures of %{customdata[1]:,.0f} machines<extra></extra>",
        width=0.55,
    ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False, y_title="Failure rate (%)"))
    fig.update_yaxes(range=[0, float(st_["rate"].max()) * 1.25])
    return fig


def chart_failure_modes(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Horizontal bars: failure-mode flags counted from the dataset."""
    mode_cols = {
        "HDF": "Heat dissipation (HDF)", "OSF": "Overstrain (OSF)", "PWF": "Power failure (PWF)",
        "TWF": "Tool wear (TWF)", "RNF": "Random (RNF)",
    }
    present = {k: v for k, v in mode_cols.items() if k in df.columns}
    if df.empty or not present:
        return _empty_fig("Failure-mode columns not found in dataset")
    vals = {v: int(df[k].sum()) for k, v in present.items()}
    ordered = sorted(vals.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in ordered], y=[k for k, _ in ordered], orientation="h",
        marker=dict(color=[v for _, v in ordered], colorscale=[[0, "#1E3A5F"], [1, COLORS["brand_cyan"]]], line=dict(width=0), cornerradius=6),
        text=[f"{v}" for _, v in ordered], textposition="outside", cliponaxis=False,
        textfont=dict(color="#FFFFFF", size=12, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>%{x} events<extra></extra>",
    ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False, x_title="Recorded events"))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[0, max(v for _, v in ordered) * 1.18])
    return fig


def chart_operating_envelope(df: pd.DataFrame, height: int = 420) -> go.Figure:
    """Speed vs torque scatter. All failures are plotted; normals are down-sampled for speed."""
    if df.empty:
        return _empty_fig()
    d = _status_series(df)
    normal = d[d["Failure_Status"] == "Normal"].sample(min(2500, int((d["Failure_Status"] == "Normal").sum())), random_state=42)
    fail = d[d["Failure_Status"] == "Failure"]
    hover = "Type %{customdata[0]} · wear %{customdata[1]:.0f} min<br>%{x:.0f} rpm · %{y:.1f} Nm<extra>%{fullData.name}</extra>"
    fig = go.Figure()
    for name, sub, color, size, op in [("Normal", normal, "rgba(56,189,248,0.38)", 5, 1), ("Failure", fail, COLORS["status_danger"], 7, 0.95)]:
        fig.add_trace(go.Scattergl(
            x=sub["Rotational speed [rpm]"], y=sub["Torque [Nm]"], mode="markers", name=name,
            marker=dict(color=color, size=size, opacity=op, line=dict(width=0)),
            customdata=sub[["Type", "Tool wear [min]"]].values, hovertemplate=hover,
        ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=True, x_title="Rotational speed [rpm]", y_title="Torque [Nm]"))
    return fig


def chart_parameter_distribution(df: pd.DataFrame, param: str, height: int = 340) -> go.Figure:
    """Box plot: normal vs failed machines on a chosen parameter."""
    if df.empty or param not in df.columns:
        return _empty_fig()
    d = _status_series(df)
    fig = go.Figure()
    for status, color, fill in [("Normal", COLORS["accent_cyan"], "rgba(56,189,248,0.18)"), ("Failure", COLORS["status_danger"], "rgba(255,77,94,0.20)")]:
        sub = d[d["Failure_Status"] == status][param]
        fig.add_trace(go.Box(
            y=sub, name=status, marker=dict(color=color, size=3, opacity=0.6), line=dict(color=color, width=2),
            fillcolor=fill, boxmean=True, boxpoints="outliers",
            hovertemplate=f"<b>{status}</b><br>%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False, y_title=param))
    return fig


def chart_rate_by_bin(df: pd.DataFrame, col: str, bins: List[float], labels: List[str],
                      x_title: str, height: int = 320, color: str = None) -> go.Figure:
    """Failure rate (bars) and machine count (line) across bins of a numeric column."""
    if df.empty or col not in df.columns:
        return _empty_fig()
    d = df.copy()
    d["bin"] = pd.cut(d[col], bins=bins, labels=labels, right=False, include_lowest=True)
    g = d.groupby("bin", observed=False)["Machine failure"].agg(["count", "sum"])
    g["rate"] = np.where(g["count"] > 0, g["sum"] / g["count"].replace(0, np.nan) * 100.0, 0.0)
    rates = g["rate"].fillna(0)
    top = rates.max()
    colors = [COLORS["status_danger"] if r >= top * 0.85 and r > 0 else (color or COLORS["accent_cyan"]) for r in rates]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(g.index.astype(str)), y=rates, marker=dict(color=colors, line=dict(width=0), cornerradius=6), name="Failure rate (%)",
        text=[f"{r:.1f}%" if c > 0 else "" for r, c in zip(rates, g["count"])], textposition="outside", cliponaxis=False,
        textfont=dict(color="#FFFFFF", size=11, family="JetBrains Mono"),
        customdata=np.stack([g["sum"], g["count"]], axis=-1),
        hovertemplate="<b>%{x}</b><br>Failure rate: %{y:.2f}%<br>%{customdata[0]:.0f} failures / %{customdata[1]:,.0f} machines<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(g.index.astype(str)), y=g["count"], name="Machines", yaxis="y2", mode="lines+markers",
        line=dict(color="rgba(148,163,184,0.7)", width=2, dash="dot"), marker=dict(size=6, color="rgba(148,163,184,0.9)"),
        hovertemplate="%{y:,} machines<extra></extra>",
    ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=True, x_title=x_title, y_title="Failure rate (%)"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, zeroline=False, tickfont=dict(size=10, color=COLORS["text_muted"]),
                                  title=dict(text="Machines", font=dict(size=11, color=COLORS["text_muted"]))))
    fig.update_layout(yaxis=dict(range=[0, max(top * 1.3, 1.0)]))
    return fig


def chart_failure_correlation(df: pd.DataFrame, height: int = 320) -> go.Figure:
    """Point-biserial style correlation of each numeric parameter with Machine failure."""
    cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
            "Tool wear [min]", "Temp_diff", "Power_kW", "Strain"]
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        return _empty_fig()
    corr = df[cols].corrwith(df["Machine failure"]).sort_values()
    colors = [COLORS["status_danger"] if v > 0 else COLORS["accent_cyan"] for v in corr]
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h", marker=dict(color=colors, line=dict(width=0), cornerradius=5),
        text=[f"{v:+.3f}" for v in corr.values], textposition="outside", cliponaxis=False,
        textfont=dict(color="#FFFFFF", size=11, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Correlation with failure: %{x:+.3f}<extra></extra>",
    ))
    lim = max(0.1, float(corr.abs().max()) * 1.8)
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False, x_title="Correlation with failure"))
    fig.update_xaxes(range=[-lim, lim])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def chart_model_metrics(comp_df: pd.DataFrame, height: int = 470) -> go.Figure:
    """Grouped bars for every metric across all candidate models."""
    if comp_df.empty or "Model" not in comp_df.columns:
        return _empty_fig()
    metrics = [("Accuracy", "#64748B"), ("Precision (Failure)", COLORS["status_warning"]),
               ("Recall (Failure)", COLORS["status_danger"]), ("F1-Score (Failure)", COLORS["status_normal"]),
               ("ROC-AUC", COLORS["brand_cyan"])]
    short = [str(m).replace("Logistic Regression", "LogReg").replace("Decision Tree", "DecTree").replace("Random Forest", "RF")
             for m in comp_df["Model"]]
    fig = go.Figure()
    for col, color in metrics:
        if col in comp_df.columns:
            fig.add_trace(go.Bar(
                name=col.replace(" (Failure)", ""), x=short, y=comp_df[col].fillna(0), marker=dict(color=color, line=dict(width=0), cornerradius=4),
                hovertemplate="<b>%{x}</b><br>" + col + ": %{y:.4f}<extra></extra>",
            ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=True, y_title="Score"))
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.06)
    fig.update_yaxes(range=[0, 1.05])
    return fig


def chart_baseline_vs_improved(meta: Dict[str, Any], height: int = 340) -> go.Figure:
    """Baseline Random Forest vs Improved Random Forest, straight from model_metadata.json."""
    m, b = meta.get("metrics", {}), meta.get("baseline_comparison", {})
    if not m or not b:
        return _empty_fig("model_metadata.json has no baseline comparison")
    rows = [("Accuracy", b["baseline_accuracy"], m["accuracy"]), ("Precision", b["baseline_precision"], m["precision_failure"]),
            ("Recall", b["baseline_recall"], m["recall_failure"]), ("F1-Score", b["baseline_f1"], m["f1_failure"]),
            ("ROC-AUC", b["baseline_roc_auc"], m["roc_auc"])]
    names = [r[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Baseline RF", x=names, y=[r[1] for r in rows], marker=dict(color="#475569", cornerradius=6),
                         text=[f"{r[1]:.3f}" for r in rows], textposition="outside", textfont=dict(size=10, family="JetBrains Mono", color=COLORS["text_secondary"]),
                         hovertemplate="Baseline %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Improved RF", x=names, y=[r[2] for r in rows], marker=dict(color=COLORS["brand_cyan"], cornerradius=6),
                         text=[f"{r[2]:.3f}" for r in rows], textposition="outside", textfont=dict(size=10, family="JetBrains Mono", color="#FFFFFF"),
                         hovertemplate="Improved %{x}: %{y:.4f}<extra></extra>"))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=True, y_title="Score"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1.12])
    return fig


def chart_confusion_matrix(cm_data: Union[Dict[str, int], np.ndarray, list], height: int = 340) -> go.Figure:
    """Confusion-matrix heatmap with counts and row-wise rates."""
    if isinstance(cm_data, (np.ndarray, list)) and np.array(cm_data).shape == (2, 2):
        a = np.array(cm_data)
        tn, fp, fn, tp = int(a[0, 0]), int(a[0, 1]), int(a[1, 0]), int(a[1, 1])
    elif isinstance(cm_data, dict) and all(k in cm_data for k in ("TN", "FP", "FN", "TP")):
        tn, fp, fn, tp = (int(cm_data[k]) for k in ("TN", "FP", "FN", "TP"))
    else:
        return _empty_fig("Confusion matrix not available")
    z = [[tn, fp], [fn, tp]]
    row_tot = [tn + fp, fn + tp]
    labels = [["True negative", "False alarm"], ["Missed failure", "Detected failure"]]
    text = [[f"<b>{z[i][j]:,}</b><br>{labels[i][j]}<br>{z[i][j] / row_tot[i] * 100:.1f}% of row" for j in range(2)] for i in range(2)]
    # Colour by correctness rather than magnitude so the small error cells stay visible
    zc = [[1, 0.28], [0.55, 0.85]]
    fig = go.Figure(go.Heatmap(
        z=zc, x=["Predicted Normal", "Predicted Failure"], y=["Actual Normal", "Actual Failure"], text=text, texttemplate="%{text}",
        textfont=dict(size=13, color="#FFFFFF"), showscale=False, xgap=6, ygap=6, hoverinfo="skip",
        colorscale=[[0, "#1a1530"], [0.28, "#5b1d2b"], [0.55, "#6b4a12"], [0.85, "#0b5f6b"], [1, "#0e7490"]],
    ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)")
    fig.update_xaxes(side="top", gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)")
    return fig


def chart_feature_importances(fi_list: Union[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]], height: int = 380) -> go.Figure:
    """Feature importances; physics-derived features are highlighted."""
    if isinstance(fi_list, dict):
        fi_list = fi_list.get("feature_importances", fi_list)
    if fi_list is None or len(fi_list) == 0:
        return _empty_fig("No feature-importance data in model metadata")
    fi_df = pd.DataFrame(fi_list) if not isinstance(fi_list, pd.DataFrame) else fi_list.copy()
    if not {"Feature", "Importance"} <= set(fi_df.columns):
        return _empty_fig()
    fi_df = fi_df.sort_values("Importance")
    derived = {"Power_W", "Strain", "Temp_diff"}
    fig = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
        marker=dict(color=[COLORS["brand_cyan"] if f in derived else "#475A7C" for f in fi_df["Feature"]], line=dict(width=0), cornerradius=5),
        text=[f"{v:.3f}" for v in fi_df["Importance"]], textposition="outside", cliponaxis=False,
        textfont=dict(color="#FFFFFF", size=11, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False, x_title="Gini importance (cyan = physics-derived feature)"))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[0, float(fi_df["Importance"].max()) * 1.18])
    return fig


def chart_roc_comparison_interactive(height: int = 470) -> go.Figure:
    """Real ROC curves of every candidate model (best model highlighted)."""
    try:
        curves = compute_roc_curves()
    except Exception as exc:  # pragma: no cover
        return _empty_fig(f"ROC curves unavailable: {exc}")
    palette = {
        "Improved Random Forest": (COLORS["brand_cyan"], 3.6),
        "Random Forest (Baseline)": ("#3B82F6", 2.0),
        "Logistic Regression (Balanced)": (COLORS["status_warning"], 1.8),
        "Logistic Regression (Baseline)": ("#64748B", 1.8),
        "Decision Tree (Balanced)": ("#EC4899", 1.8),
        "Decision Tree (Baseline)": ("#8B5CF6", 1.8),
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC 0.500)", line=dict(color="#475569", width=1.2, dash="dash"), hoverinfo="skip"))
    for name, (fpr, tpr, auc) in sorted(curves.items(), key=lambda kv: kv[1][2]):
        color, width = palette.get(name, ("#94A3B8", 1.6))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} · {auc:.3f}", line=dict(color=color, width=width, shape="hv"),
                                 hovertemplate=f"<b>{name}</b><br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>"))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=True, x_title="False positive rate", y_title="True positive rate (recall)"))
    fig.update_layout(legend=dict(orientation="h", y=-0.2, yanchor="top", x=0, font=dict(size=10)), margin=dict(l=8, r=8, t=14, b=150))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def chart_batch_distribution(results_df: pd.DataFrame, height: int = 280) -> go.Figure:
    """Donut for uploaded-batch predictions."""
    status_col = "Status" if "Status" in results_df.columns else ("Prediction_Label" if "Prediction_Label" in results_df.columns else None)
    if results_df.empty or status_col is None:
        return _empty_fig()
    counts = results_df[status_col].value_counts()
    labels = list(counts.index)
    colors = [COLORS["status_danger"] if "Fail" in str(l) and "No" not in str(l) else COLORS["status_normal"] for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=list(counts.values), hole=0.7, sort=False,
        marker=dict(colors=colors, line=dict(color=COLORS["bg_main"], width=3)), textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,} machines<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=True))
    fig.update_layout(annotations=[dict(text=f"<b>{len(results_df):,}</b>", x=0.5, y=0.54, showarrow=False, font=dict(size=26, color="#FFF", family="JetBrains Mono")),
                                   dict(text="MACHINES", x=0.5, y=0.40, showarrow=False, font=dict(size=10, color=COLORS["text_muted"]))])
    return fig


def chart_batch_risk_tiers(results_df: pd.DataFrame, height: int = 280) -> go.Figure:
    """Risk-tier counts for uploaded batch."""
    if results_df.empty or "Risk_Level" not in results_df.columns:
        return _empty_fig()
    vc = results_df["Risk_Level"].value_counts()
    tiers = ["Low Risk", "Medium / Warning", "High Risk / Critical"]
    counts = [int(vc.get(t, 0)) for t in tiers]
    fig = go.Figure(go.Bar(
        x=["Low", "Medium", "High"], y=counts, width=0.55,
        marker=dict(color=[COLORS["status_normal"], COLORS["status_warning"], COLORS["status_danger"]], line=dict(width=0), cornerradius=8),
        text=[f"{c:,}" for c in counts], textposition="outside", cliponaxis=False, textfont=dict(color="#FFF", size=12, family="JetBrains Mono"),
        hovertemplate="<b>%{x} risk</b><br>%{y:,} machines<extra></extra>",
    ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=False, y_title="Machines"))
    fig.update_yaxes(range=[0, max(max(counts), 1) * 1.25])
    return fig


def chart_batch_scatter(results_df: pd.DataFrame, height: int = 320) -> go.Figure:
    """Speed vs torque coloured by predicted status."""
    need = {"Rotational speed [rpm]", "Torque [Nm]"}
    status_col = "Status" if "Status" in results_df.columns else ("Prediction_Label" if "Prediction_Label" in results_df.columns else None)
    if results_df.empty or not need <= set(results_df.columns) or status_col is None:
        return _empty_fig()
    fig = go.Figure()
    for lbl in results_df[status_col].unique():
        sub = results_df[results_df[status_col] == lbl]
        bad = "Fail" in str(lbl) and "No" not in str(lbl)
        fig.add_trace(go.Scatter(
            x=sub["Rotational speed [rpm]"], y=sub["Torque [Nm]"], mode="markers", name=str(lbl),
            marker=dict(color=COLORS["status_danger"] if bad else "rgba(56,189,248,0.6)", size=9 if bad else 7, line=dict(width=0)),
            customdata=sub[["Failure_Probability"]].values if "Failure_Probability" in sub.columns else None,
            hovertemplate="%{x:.0f} rpm · %{y:.1f} Nm" + ("<br>P(fail) %{customdata[0]:.1%}" if "Failure_Probability" in sub.columns else "") + "<extra>" + str(lbl) + "</extra>",
        ))
    fig.update_layout(**get_plotly_layout(height=height, show_legend=True, x_title="Rotational speed [rpm]", y_title="Torque [Nm]"))
    return fig
