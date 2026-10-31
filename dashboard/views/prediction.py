"""Page: Machine Failure Prediction. Uses src.prediction.predict_single (unchanged)."""

from datetime import datetime

import streamlit as st

from dashboard.components import (
    COLORS,
    STRAIN_LIMITS,
    compute_machine_metrics,
    esc,
    get_maintenance_advisor,
    icon,
    kv_table,
    md,
    meter,
    page_header,
    panel_title,
    render_recommendation,
    risk_color,
    section_header,
    show_chart,
)
from dashboard.styles import create_probability_gauge
from src.prediction import InvalidInputError, ModelLoadError, predict_single

generate_maintenance_guidance = get_maintenance_advisor()

DEFAULTS = {"in_type": "M", "in_air_t": 298.2, "in_proc_t": 308.7, "in_speed": 1450.0, "in_torque": 42.0, "in_wear": 45.0}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

PRESETS = {
    "Nominal machine": dict(in_type="M", in_air_t=298.1, in_proc_t=308.6, in_speed=1551.0, in_torque=42.8, in_wear=15.0),
    "Tool wear hazard": dict(in_type="L", in_air_t=299.0, in_proc_t=309.2, in_speed=1410.0, in_torque=46.5, in_wear=225.0),
    "Heat dissipation": dict(in_type="L", in_air_t=303.8, in_proc_t=311.5, in_speed=1320.0, in_torque=52.0, in_wear=90.0),
    "Power overload": dict(in_type="H", in_air_t=298.5, in_proc_t=308.7, in_speed=2100.0, in_torque=48.0, in_wear=60.0),
    "Overstrain hazard": dict(in_type="L", in_air_t=301.2, in_proc_t=310.5, in_speed=1380.0, in_torque=66.0, in_wear=210.0),
}


def apply_preset(name: str) -> None:
    for key, val in PRESETS[name].items():
        st.session_state[key] = val


page_header(
    "Prediction workspace",
    "Machine Failure Prediction",
    "Enter live sensor readings for one machine. The trained model returns the failure probability, and the rule engine adds "
    "maintenance suggestions.",
    chips=[("Random Forest · physics-informed", "cyan"), ("Decision threshold 0.50", "")],
)

# ------------------------------------------------------------------ Presets
section_header("Quick scenarios", "Load a typical operating point, then adjust any value", "bolt")
pcols = st.columns(len(PRESETS))
for col, name in zip(pcols, PRESETS):
    col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,))

# ------------------------------------------------------------------ Inputs
section_header("Machine telemetry", "All fields use the units shown in brackets", "tune")
left, right = st.columns([1.7, 1.1], gap="large")

with left:
    with st.container(key="panel_inputs"):
        panel_title("Operating parameters")
        a, b = st.columns(2, gap="medium")
        with a:
            val_type = st.selectbox("Product type", ["L", "M", "H"], key="in_type",
                                    help="L = Low quality variant, M = Medium, H = High quality variant")
            val_air = st.number_input("Air temperature (K)", 275.0, 325.0, step=0.1, format="%.1f", key="in_air_t",
                                      help=f"Ambient air temperature in Kelvin (≈ {st.session_state['in_air_t'] - 273.15:.1f} °C)")
            val_proc = st.number_input("Process temperature (K)", 285.0, 335.0, step=0.1, format="%.1f", key="in_proc_t",
                                       help=f"Process temperature in Kelvin (≈ {st.session_state['in_proc_t'] - 273.15:.1f} °C)")
        with b:
            val_speed = st.number_input("Rotational speed (rpm)", 800.0, 3500.0, step=10.0, format="%.0f", key="in_speed")
            val_torque = st.number_input("Torque (Nm)", 2.0, 120.0, step=0.5, format="%.1f", key="in_torque")
            val_wear = st.number_input("Tool wear (min)", 0.0, 300.0, step=1.0, format="%.0f", key="in_wear",
                                       help="Cumulative cutting time of the current tool")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        run = st.button("PREDICT MACHINE CONDITION", type="primary", icon=":material/bolt:", key="run_prediction")

current = {
    "Type": val_type,
    "Air temperature [K]": float(val_air),
    "Process temperature [K]": float(val_proc),
    "Rotational speed [rpm]": float(val_speed),
    "Torque [Nm]": float(val_torque),
    "Tool wear [min]": float(val_wear),
}

# Live physics panel (updates as the inputs change)
with right:
    with st.container(key="panel_live"):
        panel_title("Live derived indicators", "Computed from the inputs with the same formulas the model uses")
        m = compute_machine_metrics(current)
        dt_bad = m["delta_t"] < 8.6 and m["speed"] < 1380
        pw_bad = m["power_kw"] > 9.0 or (m["power_kw"] < 3.5 and m["speed"] < 1250)
        st_bad = m["strain"] > m["strain_limit"]
        wr_bad = m["wear"] >= 200
        ok, bad = COLORS["status_normal"], COLORS["status_danger"]
        md(
            meter("Temperature difference ΔT", m["delta_t"], 15.0, f"{m['delta_t']:.2f} K", bad if dt_bad else ok, marker=8.6,
                  note="HDF risk when below 8.6 K at low speed (&lt; 1380 rpm)")
            + meter("Mechanical power", m["power_kw"], 12.0, f"{m['power_kw']:.2f} kW", bad if pw_bad else ok, marker=9.0,
                    note="Nominal band 3.5 – 9.0 kW")
            + meter("Overstrain index (torque × wear)", m["strain"], m["strain_limit"] * 1.25, f"{m['strain']:,.0f}", bad if st_bad else ok,
                    marker=m["strain_limit"], note=f"Limit for type {val_type}: {m['strain_limit']:,.0f} Nm·min")
            + meter("Tool wear", m["wear"], 260.0, f"{m['wear']:.0f} min", bad if wr_bad else (COLORS["status_warning"] if m["wear"] >= 160 else ok),
                    marker=200.0, note="Tool-wear failure risk from 200 min")
        )

# ------------------------------------------------------------------ Run prediction
if run:
    try:
        with st.spinner("Running telemetry through the trained model..."):
            result = predict_single(**current)
            guidance = generate_maintenance_guidance(current)
        stamp = datetime.now().strftime("%H:%M:%S")
        entry = dict(result)
        entry["Timestamp"] = stamp
        st.session_state.session_history.insert(0, entry)
        st.session_state.last_prediction = {"input": dict(current), "result": result, "guidance": guidance, "time": stamp}
    except InvalidInputError as exc:
        st.session_state.last_prediction = None
        st.error(f"Validation error: {exc}")
    except ModelLoadError as exc:
        st.session_state.last_prediction = None
        st.error(f"Model error: {exc}")

# ------------------------------------------------------------------ Result
lp = st.session_state.get("last_prediction")
if lp:
    res, inp, guidance = lp["result"], lp["input"], lp["guidance"]
    stale = inp != current
    is_fail = int(res["Prediction"]) == 1
    prob = float(res["Failure_Probability"])

    section_header("Prediction result", f"Generated at {lp['time']}", "fact_check")
    if stale:
        md('<div class="chip-row" style="margin-bottom:12px;"><span class="chip warn">Inputs changed since this prediction — press Predict to refresh</span></div>')

    cls = "bad" if is_fail else "ok"
    status_txt = "FAILURE PREDICTED" if is_fail else "NORMAL"
    ico = "dangerous" if is_fail else "verified_user"
    sub = ("The model expects this machine to fail at the current operating point." if is_fail
           else "The model does not expect a failure at the current operating point.")
    md(f"""
    <div class="result-hero {cls}">
        <div class="rh-eyebrow">Model verdict</div>
        <div class="rh-status">{icon(ico)}<div class="rh-text">{status_txt}</div></div>
        <div class="rh-meta">{sub} Estimated failure probability <b style="color:#fff">{prob * 100:.1f}%</b>
        &nbsp;·&nbsp; risk tier <b style="color:{risk_color(res['Risk_Level'])}">{esc(res['Risk_Level'])}</b></div>
    </div>
    """)

    c1, c2, c3 = st.columns([1.05, 1, 1.15], gap="medium")
    with c1:
        with st.container(key="panel_gauge"):
            panel_title("Failure probability", "White marker = 50% decision threshold")
            show_chart(create_probability_gauge(prob, "P(failure)"), key="pred_gauge")
    with c2:
        with st.container(key="panel_summary"):
            panel_title("Machine input summary")
            kv_table([
                ("Product type", esc(inp["Type"])),
                ("Air temperature", f"{inp['Air temperature [K]']:.1f} K"),
                ("Process temperature", f"{inp['Process temperature [K]']:.1f} K"),
                ("Rotational speed", f"{inp['Rotational speed [rpm]']:.0f} rpm"),
                ("Torque", f"{inp['Torque [Nm]']:.1f} Nm"),
                ("Tool wear", f"{inp['Tool wear [min]']:.0f} min"),
            ])
    with c3:
        with st.container(key="panel_diag"):
            panel_title("Diagnostic notes", "Rule-based explanation of the risk factors")
            notes = [n.strip() for n in str(res["Diagnostic_Notes"]).split("|") if n.strip()]
            md("".join(
                f'<div class="rec-line" style="margin:0 0 9px 0;display:flex;gap:9px;">{icon("chevron_right")}<span>{esc(n)}</span></div>'
                for n in notes
            ))
            md('<div class="meter-note" style="margin-top:10px;">Probabilistic decision support, not a guaranteed diagnosis.</div>')

    section_header("Suggested maintenance actions", "Rule-based suggestions from the maintenance advisor — to be confirmed by a qualified technician", "build_circle")
    for rec in guidance:
        render_recommendation(rec)
else:
    md("""
    <div class="callout" style="margin-top:18px;"><div class="callout-title">Ready for a prediction</div>
    Choose a scenario or enter your own readings, then press <b>Predict machine condition</b>. The result panel with
    probability gauge, input summary and maintenance suggestions appears here.</div>
    """)
