"""Page: Maintenance Insights. Integrates the existing rule-based advisor (suggestions, not diagnoses)."""

import streamlit as st

from dashboard.components import (
    COLORS,
    compute_machine_metrics,
    esc,
    get_maintenance_advisor,
    icon,
    insight_grid,
    md,
    meter,
    page_header,
    panel_title,
    priority_badge_class,
    render_alert_box,
    render_recommendation,
    risk_color,
    section_header,
)
from src.prediction import InvalidInputError, ModelLoadError, predict_single

generate_maintenance_guidance = get_maintenance_advisor()

page_header(
    "Decision support",
    "Maintenance Insights",
    "Turns sensor readings into risk factors and suggested maintenance actions. Suggestions come from transparent, rule-based "
    "thresholds and the model's failure probability — they support, but never replace, a qualified technician's judgement.",
    chips=[("Rule-based suggestions", "warn"), ("Not a guaranteed diagnosis", "")],
)

# ------------------------------------------------------------------ Source of readings
last = st.session_state.get("last_prediction")
options = ["Interactive simulator"] + (["Latest prediction"] if last else [])
default_idx = options.index("Latest prediction") if last else 0
source = st.radio("Machine readings", options, index=default_idx, horizontal=True, key="mt_source")

if source == "Latest prediction" and last:
    readings = dict(last["input"])
    md(f'<div class="chip-row" style="margin:4px 0 14px 0;"><span class="chip cyan">Using readings from the prediction made at {esc(last["time"])}</span></div>')
else:
    section_header("Simulator", "Adjust the readings to see which rules trigger", "tune")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        s_type = st.selectbox("Product type", ["L", "M", "H"], key="mt_type")
        s_wear = st.slider("Tool wear (min)", 0, 260, 210, key="mt_wear")
    with c2:
        s_air = st.slider("Air temperature (K)", 295.0, 305.0, 303.5, 0.1, key="mt_air")
        s_proc = st.slider("Process temperature (K)", 305.0, 315.0, 311.0, 0.1, key="mt_proc")
    with c3:
        s_speed = st.slider("Rotational speed (rpm)", 1100, 2800, 1320, 10, key="mt_speed")
        s_torque = st.slider("Torque (Nm)", 10.0, 75.0, 62.0, 0.5, key="mt_torque")
    readings = {
        "Type": s_type, "Air temperature [K]": float(s_air), "Process temperature [K]": float(s_proc),
        "Rotational speed [rpm]": float(s_speed), "Torque [Nm]": float(s_torque), "Tool wear [min]": float(s_wear),
    }

guidance = generate_maintenance_guidance(readings)
m = compute_machine_metrics(readings)

# Model verdict for these readings (real model output)
verdict = None
if source == "Latest prediction" and last:
    verdict = last["result"]
else:
    try:
        verdict = predict_single(**readings)
    except (InvalidInputError, ModelLoadError):
        verdict = None

n_crit = sum(1 for g in guidance if "CRITICAL" in str(g.get("priority", "")).upper())
n_warn = sum(1 for g in guidance if "WARNING" in str(g.get("priority", "")).upper())
if n_crit:
    sev_cls, sev_txt, sev_ico = "bad", "ACTION REQUIRED", "report"
elif n_warn:
    sev_cls, sev_txt, sev_ico = "warn", "ATTENTION ADVISED", "warning"
else:
    sev_cls, sev_txt, sev_ico = "ok", "HEALTHY", "verified_user"

# ------------------------------------------------------------------ Machine status
section_header("Machine status", "Combined view of model verdict and rule-based checks", "monitor_heart")
s1, s2 = st.columns([1.1, 1], gap="medium")
with s1:
    model_line = ""
    if verdict is not None:
        p = float(verdict["Failure_Probability"])
        model_line = (f'Model verdict: <b style="color:#fff">{esc(verdict["Status"])}</b> · failure probability '
                      f'<b style="color:#fff">{p * 100:.1f}%</b> · <b style="color:{risk_color(verdict["Risk_Level"])}">{esc(verdict["Risk_Level"])}</b>')
    md(f"""
    <div class="result-hero {sev_cls}" style="margin-bottom:0;">
        <div class="rh-eyebrow">Maintenance status</div>
        <div class="rh-status">{icon(sev_ico)}<div class="rh-text">{sev_txt}</div></div>
        <div class="rh-meta">{n_crit} critical · {n_warn} warning rule(s) triggered<br>{model_line}</div>
    </div>
    """)
with s2:
    with st.container(key="panel_risk"):
        panel_title("Risk factors", "Marker = rule threshold")
        ok_c, bad_c, warn_c = COLORS["status_normal"], COLORS["status_danger"], COLORS["status_warning"]
        wear_c = bad_c if m["wear"] >= 200 else (warn_c if m["wear"] >= 160 else ok_c)
        dt_c = bad_c if (m["delta_t"] < 8.6 and m["speed"] < 1380) else ok_c
        pw_c = bad_c if (m["power_kw"] > 9.0 or (m["power_kw"] < 3.5 and m["speed"] < 1250)) else ok_c
        st_c = bad_c if m["strain"] > m["strain_limit"] else ok_c
        md(
            meter("Tool wear", m["wear"], 260.0, f"{m['wear']:.0f} min", wear_c, marker=200.0)
            + meter("Thermal gradient ΔT", m["delta_t"], 15.0, f"{m['delta_t']:.2f} K", dt_c, marker=8.6)
            + meter("Mechanical power", m["power_kw"], 12.0, f"{m['power_kw']:.2f} kW", pw_c, marker=9.0)
            + meter("Overstrain index", m["strain"], m["strain_limit"] * 1.25, f"{m['strain']:,.0f}", st_c, marker=m["strain_limit"])
        )

# ------------------------------------------------------------------ Watchlist (tool wear / temperature / torque)
section_header("Warnings by area", "Tool wear, temperature and torque observations for these readings", "fact_check")


def area(title, ico, level, message):
    cls = {"ok": "badge-normal", "warn": "badge-warning", "bad": "badge-danger"}[level]
    lab = {"ok": "OK", "warn": "WARNING", "bad": "CRITICAL"}[level]
    return (f'<div class="insight"><div class="insight-ico">{icon(ico)}</div><div>'
            f'<div class="insight-t">{title} <span class="badge {cls}" style="margin-left:6px;">{lab}</span></div>'
            f'<div class="insight-d">{message}</div></div></div>')


# tool wear
if m["wear"] >= 200:
    w = ("bad", f"Tool wear is {m['wear']:.0f} min, at or beyond the 200 min limit.")
elif m["wear"] >= 160:
    w = ("warn", f"Tool wear is {m['wear']:.0f} min and approaching the 200 min limit.")
else:
    w = ("ok", f"Tool wear is {m['wear']:.0f} min, below the 160 min watch level.")
# temperature
if m["delta_t"] < 8.6 and m["speed"] < 1380:
    t = ("bad", f"ΔT is {m['delta_t']:.1f} K (&lt; 8.6 K) at {m['speed']:.0f} rpm — poor heat dissipation.")
elif m["proc"] >= 313.0:
    t = ("warn", f"Process temperature is elevated at {m['proc']:.1f} K.")
else:
    t = ("ok", f"Process temperature {m['proc']:.1f} K and ΔT {m['delta_t']:.1f} K are within normal limits.")
# torque
if m["torque"] > 60:
    q = ("warn", f"Torque of {m['torque']:.1f} Nm is high (&gt; 60 Nm) — elevated mechanical load.")
elif m["torque"] < 15:
    q = ("warn", f"Torque of {m['torque']:.1f} Nm is unusually low (&lt; 15 Nm).")
else:
    q = ("ok", f"Torque of {m['torque']:.1f} Nm is within the usual range.")
if m["strain"] > m["strain_limit"]:
    q = ("bad", f"Torque × wear is {m['strain']:,.0f}, above the {m['strain_limit']:,.0f} limit for type {esc(readings['Type'])} — overstrain risk.")

md('<div class="insight-grid">' + area("Tool wear", "build", *w) + area("Temperature", "thermostat", *t) + area("Torque & load", "speed", *q) + "</div>")

# ------------------------------------------------------------------ Actions
section_header("Suggested maintenance actions", "Rule-based suggestions — confirm with a qualified technician before acting", "engineering")
for rec in guidance:
    render_recommendation(rec)

# ------------------------------------------------------------------ Rule reference
with st.expander("Documented rules behind the suggestions", expanded=False, icon=":material/menu_book:"):
    insight_grid([
        ("build", "Tool wear (TWF)", "Critical at tool wear ≥ 200 min; warning from 160 min. Action: schedule insert replacement."),
        ("thermostat", "Heat dissipation (HDF)", "Critical when ΔT &lt; 8.6 K and speed &lt; 1380 rpm; warning when process temperature ≥ 313 K. Action: check coolant and radiator."),
        ("bolt", "Power (PWF)", "Critical above 9.0 kW; warning below 3.5 kW at low speed. Action: inspect drive current, belts and bearings."),
        ("construction", "Overstrain (OSF)", "Critical when torque × wear exceeds 11,000 (L), 12,000 (M) or 13,000 (H). Action: replace tool and reduce load."),
        ("task_alt", "Nominal", "No rule triggered: continue the standard preventive-maintenance schedule."),
    ])
render_alert_box("Advisory notice", "These are rule-based suggestions derived from the AI4I benchmark's published failure conditions. They are decision-support aids, "
                 "not guaranteed diagnoses, and must be confirmed by qualified maintenance staff.", "info")
