"""
AI-Based Industrial Maintenance and Process Optimization System
Dashboard Design System: colour tokens, global CSS, Plotly theme.
Infosys Springboard Virtual Internship 7.0

This module is presentation-only. It contains no ML or data logic.
"""

import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------------
# Colour tokens
# ------------------------------------------------------------------------------
COLORS = {
    # Surfaces (deep midnight navy -> charcoal)
    "bg_main": "#070B16",
    "bg_card": "#0E1526",
    "bg_card_elevated": "#141D33",
    "bg_sidebar": "#080D1B",
    "bg_input": "#0A1122",
    # Borders
    "border_subtle": "rgba(148, 163, 184, 0.13)",
    "border_light": "rgba(148, 163, 184, 0.24)",
    "border_focus": "#22D3EE",
    # Text
    "text_primary": "#F1F5F9",
    "text_secondary": "#9AA9C0",
    "text_muted": "#64748B",
    # Brand
    "brand_blue": "#3B82F6",
    "brand_cyan": "#22D3EE",
    "accent_cyan": "#38BDF8",
    "accent_violet": "#8B5CF6",
    # Status
    "status_normal": "#10D9A0",
    "status_normal_bg": "rgba(16, 217, 160, 0.12)",
    "status_warning": "#F5A524",
    "status_warning_bg": "rgba(245, 165, 36, 0.12)",
    "status_danger": "#FF4D5E",
    "status_danger_bg": "rgba(255, 77, 94, 0.12)",
}

# Categorical palette for charts (distinct, colour-blind friendly ordering)
CHART_PALETTE = ["#22D3EE", "#8B5CF6", "#F5A524", "#FF4D5E", "#10D9A0", "#3B82F6"]

DASHBOARD_VERSION = "2.0.0"


def _root_vars() -> str:
    """Expose colour tokens as CSS custom properties."""
    lines = [f"--{k.replace('_', '-')}: {v};" for k, v in COLORS.items()]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


_STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.mono, .kpi-value, .num { font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace; }

/* ---------- App canvas: midnight gradient + faint engineering grid ---------- */
.stApp {
    background-color: var(--bg-main);
    background-image:
        radial-gradient(900px 480px at 88% -8%, rgba(34, 211, 238, 0.10), transparent 60%),
        radial-gradient(760px 420px at 6% 4%, rgba(59, 130, 246, 0.10), transparent 60%),
        linear-gradient(rgba(148,163,184,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,0.035) 1px, transparent 1px);
    background-size: auto, auto, 44px 44px, 44px 44px;
    background-attachment: fixed;
    color: var(--text-primary);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px; }

h1, h2, h3, h4, h5 { color: var(--text-primary); letter-spacing: -0.01em; }
p, li, label, span { color: inherit; }
hr { border-color: var(--border-subtle) !important; }
code {
    background: rgba(34, 211, 238, 0.09); color: var(--brand-cyan);
    padding: 1px 6px; border-radius: 5px; font-size: 0.82em;
    font-family: 'JetBrains Mono', monospace;
}
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: #1f2b46; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #2b3b60; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1020 0%, #070C18 100%);
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0.4rem; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0 0.9rem 1rem 0.9rem; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { display: none; }

.brand { display: flex; align-items: center; gap: 12px; padding: 14px 6px 18px 6px; }
.brand-mark {
    width: 44px; height: 44px; border-radius: 13px; flex: 0 0 44px;
    display: grid; place-items: center;
    background: linear-gradient(140deg, rgba(34,211,238,0.22), rgba(59,130,246,0.18));
    border: 1px solid rgba(34, 211, 238, 0.45);
    box-shadow: 0 0 22px rgba(34, 211, 238, 0.22), inset 0 0 14px rgba(34,211,238,0.10);
}
.brand-name { font-weight: 800; letter-spacing: 0.14em; font-size: 1.02rem; color: #fff; line-height: 1.1; }
.brand-name b { color: var(--brand-cyan); font-weight: 800; }
.brand-sub { font-size: 0.68rem; color: var(--text-muted); letter-spacing: 0.04em; margin-top: 3px; }

.nav-label {
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.16em; color: var(--text-muted);
    text-transform: uppercase; padding: 6px 8px 8px 8px;
}

/* Navigation links (st.page_link) */
section[data-testid="stSidebar"] div[data-testid="stPageLink"] { width: 100%; }
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
    width: 100%; border-radius: 11px; padding: 0.5rem 0.8rem; margin-bottom: 0;
    border: 1px solid transparent; transition: all .16s ease;
    color: var(--text-secondary);
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p {
    font-size: 0.88rem; font-weight: 500; color: inherit;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
    background: rgba(148, 163, 184, 0.07); color: #fff; transform: translateX(2px);
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: linear-gradient(90deg, rgba(34,211,238,0.16), rgba(59,130,246,0.05));
    border: 1px solid rgba(34, 211, 238, 0.35);
    box-shadow: inset 3px 0 0 var(--brand-cyan), 0 0 18px rgba(34,211,238,0.08);
    color: #fff;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] p { font-weight: 700; }
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] span[data-testid="stIconMaterial"] {
    color: var(--brand-cyan);
}

.side-card {
    background: rgba(20, 29, 51, 0.55); border: 1px solid var(--border-subtle);
    border-radius: 12px; padding: 11px 13px; margin-bottom: 9px;
}
.side-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.side-title { font-size: 0.66rem; letter-spacing: 0.13em; font-weight: 700; color: var(--text-muted); text-transform: uppercase; }
.side-val { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); }
.side-sub { font-size: 0.7rem; color: var(--text-muted); margin-top: 3px; line-height: 1.45; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 7px; }
.dot.ok { background: var(--status-normal); box-shadow: 0 0 0 3px rgba(16,217,160,0.18), 0 0 10px var(--status-normal); }
.dot.bad { background: var(--status-danger); box-shadow: 0 0 0 3px rgba(255,77,94,0.18); }
.side-foot { font-size: 0.68rem; color: var(--text-muted); line-height: 1.6; padding: 10px 6px 0 6px;
    border-top: 1px solid var(--border-subtle); margin-top: 14px; }

/* ---------- Hero / page headers ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px;
    border: 1px solid rgba(34, 211, 238, 0.22);
    background:
        radial-gradient(520px 260px at 100% 0%, rgba(34,211,238,0.20), transparent 65%),
        radial-gradient(420px 260px at 0% 100%, rgba(59,130,246,0.18), transparent 65%),
        linear-gradient(135deg, #0E1A33 0%, #0A1226 60%, #0B1630 100%);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 22px;
}
.hero-art { position: absolute; right: -50px; top: -60px; opacity: 0.38; pointer-events: none; }
.hero-eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--brand-cyan); }
.hero-eyebrow::before { content: ""; width: 26px; height: 2px; background: var(--brand-cyan); border-radius: 2px; }
.hero h1 { font-size: 2.35rem; font-weight: 800; line-height: 1.12; margin: 12px 0 10px 0; max-width: 760px;
    background: linear-gradient(90deg, #fff 30%, #9fe8f7 100%); -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; }
.hero p { color: var(--text-secondary); max-width: 640px; font-size: 0.97rem; line-height: 1.6; margin: 0 0 18px 0; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { display: inline-flex; align-items: center; gap: 7px; padding: 6px 12px; border-radius: 999px;
    font-size: 0.74rem; font-weight: 600; color: var(--text-secondary);
    background: rgba(148,163,184,0.08); border: 1px solid var(--border-subtle); }
.chip.ok { color: var(--status-normal); background: var(--status-normal-bg); border-color: rgba(16,217,160,.35); }
.chip.cyan { color: var(--brand-cyan); background: rgba(34,211,238,0.09); border-color: rgba(34,211,238,.32); }
.chip.warn { color: var(--status-warning); background: var(--status-warning-bg); border-color: rgba(245,165,36,.35); }
.chip.bad { color: var(--status-danger); background: var(--status-danger-bg); border-color: rgba(255,77,94,.35); }

.page-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 18px; flex-wrap: wrap;
    padding-bottom: 18px; margin-bottom: 22px; border-bottom: 1px solid var(--border-subtle); position: relative; }
.page-head::after { content: ""; position: absolute; left: 0; bottom: -1px; width: 140px; height: 2px;
    background: linear-gradient(90deg, var(--brand-cyan), transparent); }
.page-eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: var(--brand-cyan); }
.page-title { font-size: 1.85rem; font-weight: 800; margin: 6px 0 4px 0; color: #fff; line-height: 1.15; }
.page-desc { font-size: 0.92rem; color: var(--text-secondary); max-width: 780px; line-height: 1.55; }

.sec-head { display: flex; align-items: center; gap: 12px; margin: 30px 0 14px 0; }
.sec-ico { width: 32px; height: 32px; border-radius: 9px; display: grid; place-items: center;
    background: rgba(34,211,238,0.10); border: 1px solid rgba(34,211,238,0.28); color: var(--brand-cyan); }
.sec-ico .material-symbols-rounded { font-size: 19px; }
.sec-title { font-size: 1.08rem; font-weight: 700; color: #fff; line-height: 1.2; }
.sec-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
.sec-line { flex: 1; height: 1px; background: linear-gradient(90deg, var(--border-light), transparent); }

.material-symbols-rounded {
    font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
    letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
    -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle;
}

/* ---------- KPI tiles ---------- */
.kpi-grid { display: grid; gap: 14px; grid-template-columns: repeat(var(--n, 4), minmax(0, 1fr)); }
@media (max-width: 1250px) { .kpi-grid { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); } }
.kpi-container {
    position: relative; overflow: hidden; border-radius: 16px; padding: 16px 16px 15px 16px;
    background: linear-gradient(160deg, rgba(20,29,51,0.92), rgba(12,19,36,0.92));
    border: 1px solid var(--border-subtle); transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.kpi-container::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--kpi-accent, var(--brand-cyan)), transparent 80%); }
.kpi-container:hover { transform: translateY(-2px); border-color: var(--border-light); box-shadow: 0 14px 34px rgba(0,0,0,.35); }
.kpi-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.kpi-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); }
.kpi-ico { color: var(--kpi-accent, var(--brand-cyan)); opacity: .9; }
.kpi-ico .material-symbols-rounded { font-size: 20px; }
.kpi-value { font-size: 1.95rem; font-weight: 700; line-height: 1.1; margin: 10px 0 6px 0; color: #fff; letter-spacing: -0.02em; }
.kpi-value small { font-size: 0.95rem; color: var(--text-secondary); font-weight: 500; margin-left: 2px; }
.kpi-subtext { font-size: 0.76rem; color: var(--text-secondary); }
.kpi-bar { height: 5px; background: rgba(148,163,184,0.14); border-radius: 6px; margin-top: 12px; overflow: hidden; }
.kpi-bar > span { display: block; height: 100%; border-radius: 6px; background: var(--kpi-accent, var(--brand-cyan)); }

/* ---------- Panels ---------- */
.panel-title { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 2px; }
.panel-sub { font-size: 0.76rem; color: var(--text-muted); margin-bottom: 8px; }

[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > .element-container > .stMarkdown .panel-title),
div[class*="st-key-panel"] {
    background: linear-gradient(165deg, rgba(17, 25, 45, 0.88), rgba(11, 17, 32, 0.88));
    border: 1px solid var(--border-subtle) !important; border-radius: 18px !important;
    padding: 6px 8px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
div[class*="st-key-panel"] { padding: 16px 18px 12px 18px !important; }

/* ---------- Status & result ---------- */
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; white-space: nowrap; }
.badge-normal { color: var(--status-normal); background: var(--status-normal-bg); border: 1px solid rgba(16,217,160,.4); }
.badge-warning { color: var(--status-warning); background: var(--status-warning-bg); border: 1px solid rgba(245,165,36,.4); }
.badge-danger { color: var(--status-danger); background: var(--status-danger-bg); border: 1px solid rgba(255,77,94,.4); }
.badge-info { color: var(--brand-cyan); background: rgba(34,211,238,.10); border: 1px solid rgba(34,211,238,.4); }

.result-hero { position: relative; overflow: hidden; border-radius: 20px; padding: 26px 28px; margin-bottom: 16px;
    border: 1px solid var(--rh-border); background: linear-gradient(135deg, var(--rh-bg), rgba(10,16,32,0.9) 70%);
    box-shadow: 0 0 40px var(--rh-glow); }
.result-hero.ok { --rh-border: rgba(16,217,160,.45); --rh-bg: rgba(16,217,160,.14); --rh-glow: rgba(16,217,160,.10); --rh-c: var(--status-normal); }
.result-hero.bad { --rh-border: rgba(255,77,94,.5); --rh-bg: rgba(255,77,94,.16); --rh-glow: rgba(255,77,94,.14); --rh-c: var(--status-danger); }
.result-hero.warn { --rh-border: rgba(245,165,36,.5); --rh-bg: rgba(245,165,36,.14); --rh-glow: rgba(245,165,36,.10); --rh-c: var(--status-warning); }
.rh-eyebrow { font-size: 0.68rem; letter-spacing: 0.16em; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); }
.rh-status { display: flex; align-items: center; gap: 14px; margin: 8px 0 6px 0; }
.rh-status .material-symbols-rounded { font-size: 46px; color: var(--rh-c); }
.rh-text { font-size: 2.05rem; font-weight: 800; letter-spacing: -0.01em; color: var(--rh-c); line-height: 1.05; }
.rh-meta { font-size: 0.84rem; color: var(--text-secondary); }

.rec-card { border-radius: 14px; padding: 15px 18px; margin-bottom: 11px; background: rgba(14, 21, 38, 0.9);
    border: 1px solid var(--border-subtle); border-left: 4px solid var(--rc, var(--brand-cyan)); transition: transform .16s ease, border-color .16s ease; }
.rec-card:hover { transform: translateX(3px); border-color: var(--border-light); border-left-color: var(--rc, var(--brand-cyan)); }
.rec-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.rec-title { font-weight: 700; color: #fff; font-size: 0.95rem; }
.rec-line { font-size: 0.82rem; color: var(--text-secondary); margin-top: 5px; line-height: 1.5; }
.rec-line b { color: var(--text-primary); font-weight: 600; }

.callout { border-radius: 14px; padding: 14px 18px; font-size: 0.86rem; line-height: 1.6; color: var(--text-secondary);
    border: 1px solid var(--border-subtle); background: rgba(20, 29, 51, 0.55); border-left: 4px solid var(--brand-cyan); margin: 6px 0 10px 0; }
.callout b, .callout strong { color: var(--text-primary); }
.callout-title { color: #fff; font-weight: 700; font-size: 0.92rem; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }
.callout.warning { border-left-color: var(--status-warning); background: rgba(245,165,36,.07); }
.callout.danger { border-left-color: var(--status-danger); background: rgba(255,77,94,.07); }
.callout.success { border-left-color: var(--status-normal); background: rgba(16,217,160,.07); }

.insight-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.insight { display: flex; gap: 13px; padding: 15px 16px; border-radius: 14px; background: rgba(14,21,38,.85);
    border: 1px solid var(--border-subtle); transition: border-color .16s ease; }
.insight:hover { border-color: rgba(34,211,238,.35); }
.insight-ico { flex: 0 0 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
    background: rgba(34,211,238,.10); color: var(--brand-cyan); }
.insight-ico .material-symbols-rounded { font-size: 19px; }
.insight-t { font-weight: 700; font-size: 0.86rem; color: #fff; margin-bottom: 3px; }
.insight-d { font-size: 0.79rem; color: var(--text-secondary); line-height: 1.5; }

/* Meters */
.meter { margin-bottom: 15px; }
.meter-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 7px; gap: 8px; }
.meter-name { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); }
.meter-val { font-size: 0.86rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.meter-track { position: relative; height: 8px; border-radius: 8px; background: rgba(148,163,184,.14); }
.meter-fill { height: 100%; border-radius: 8px; }
.meter-mark { position: absolute; top: -4px; width: 2px; height: 16px; background: #fff; opacity: .75; border-radius: 2px; }
.meter-note { font-size: 0.7rem; color: var(--text-muted); margin-top: 5px; }

/* Key-value summary */
.kv { display: grid; grid-template-columns: 1fr auto; gap: 0; }
.kv > div { padding: 9px 0; border-bottom: 1px dashed rgba(148,163,184,.16); font-size: 0.84rem; }
.kv > div:nth-child(odd) { color: var(--text-secondary); }
.kv > div:nth-child(even) { color: #fff; font-weight: 600; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

/* Workflow stepper */
.stepper { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.step { flex: 1 1 180px; display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 14px;
    background: rgba(14,21,38,.8); border: 1px solid var(--border-subtle); }
.step-n { width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; font-weight: 700;
    font-size: 0.82rem; background: rgba(148,163,184,.12); color: var(--text-muted); flex: 0 0 30px; }
.step-t { font-size: 0.82rem; font-weight: 700; color: var(--text-secondary); line-height: 1.2; }
.step-d { font-size: 0.7rem; color: var(--text-muted); }
.step.done { border-color: rgba(16,217,160,.4); }
.step.done .step-n { background: var(--status-normal); color: #04130e; }
.step.done .step-t { color: #fff; }
.step.active { border-color: rgba(34,211,238,.55); box-shadow: 0 0 20px rgba(34,211,238,.10); }
.step.active .step-n { background: var(--brand-cyan); color: #04141a; }
.step.active .step-t { color: #fff; }

/* Model table */
.model-table { width: 100%; border-collapse: separate; border-spacing: 0 6px; font-size: 0.82rem; }
.model-table th { text-align: right; font-size: 0.66rem; letter-spacing: 0.11em; text-transform: uppercase; color: var(--text-muted);
    font-weight: 700; padding: 4px 12px; white-space: nowrap; }
.model-table th:first-child, .model-table td:first-child { text-align: left; }
.model-table td { padding: 11px 12px; background: rgba(14,21,38,.9); text-align: right; color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; border-top: 1px solid var(--border-subtle); border-bottom: 1px solid var(--border-subtle); }
.model-table td:first-child { font-family: 'Inter', sans-serif; font-weight: 600; border-left: 1px solid var(--border-subtle);
    border-radius: 12px 0 0 12px; }
.model-table td:last-child { border-right: 1px solid var(--border-subtle); border-radius: 0 12px 12px 0; }
.model-table tr.best td { background: linear-gradient(90deg, rgba(34,211,238,.14), rgba(14,21,38,.9)); border-color: rgba(34,211,238,.45); }
.model-table td.top { color: var(--brand-cyan); font-weight: 700; }
.model-table td.weak { color: var(--status-warning); }
.model-table td.bad { color: var(--status-danger); }

.tag { font-family: 'Inter'; font-size: 0.6rem; padding: 2px 8px; border-radius: 999px; margin-left: 8px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; vertical-align: middle; }
.tag.best { background: rgba(34,211,238,.16); color: var(--brand-cyan); border: 1px solid rgba(34,211,238,.45); }
.tag.base { background: rgba(148,163,184,.12); color: var(--text-secondary); border: 1px solid var(--border-light); }
.tag.imp { background: rgba(139,92,246,.16); color: #b9a2ff; border: 1px solid rgba(139,92,246,.45); }

/* ---------- Streamlit widgets ---------- */
div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
    width: 100%; border-radius: 12px; font-weight: 600; padding: 0.6rem 1rem;
    background: rgba(20, 29, 51, 0.85); color: var(--text-primary); border: 1px solid var(--border-light);
    transition: all .16s ease;
}
div[data-testid="stButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--brand-cyan); color: #fff; background: rgba(34,211,238,.09); transform: translateY(-1px);
}
button[data-testid="stBaseButton-primary"], div[data-testid="stButton"] > button[kind="primary"], div[data-testid="stDownloadButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%) !important; border: 1px solid rgba(255,255,255,.18) !important;
    color: #031018 !important; font-weight: 800 !important; letter-spacing: 0.03em; font-size: 0.98rem !important;
    padding: 0.8rem 1.2rem !important; box-shadow: 0 10px 28px rgba(34, 211, 238, 0.28);
}
button[data-testid="stBaseButton-primary"]:hover, div[data-testid="stButton"] > button[kind="primary"]:hover {
    filter: brightness(1.1); transform: translateY(-2px); box-shadow: 0 14px 34px rgba(34, 211, 238, 0.4); color: #031018 !important;
}
button[data-testid="stBaseButton-primary"] p { color: #031018 !important; font-weight: 800; }

label[data-testid="stWidgetLabel"] p, .stSelectbox label p, .stNumberInput label p, .stSlider label p {
    font-size: 0.78rem !important; font-weight: 600 !important; color: var(--text-secondary) !important; letter-spacing: 0.02em;
}
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
    background-color: var(--bg-input) !important; border-color: var(--border-light) !important; border-radius: 10px !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: var(--brand-cyan) !important; box-shadow: 0 0 0 3px rgba(34,211,238,.14) !important;
}
input { font-family: 'JetBrains Mono', monospace !important; }
div[data-testid="stNumberInput"] button { background: transparent !important; border-color: var(--border-subtle) !important; }

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(34, 211, 238, 0.45); border-radius: 18px; padding: 30px 20px;
    background: radial-gradient(420px 160px at 50% 0%, rgba(34,211,238,.10), transparent 70%), rgba(10, 17, 34, 0.7);
    transition: all .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--brand-cyan); background-color: rgba(34,211,238,.06); }

div[data-testid="stDataFrame"] { border: 1px solid var(--border-subtle); border-radius: 14px; overflow: hidden; }
div[data-testid="stPlotlyChart"] { border-radius: 12px; }
div[data-testid="stSlider"] [role="slider"] { background: var(--brand-cyan) !important; box-shadow: 0 0 0 4px rgba(34,211,238,.18); }
details[data-testid="stExpander"], div[data-testid="stExpander"] { border: 1px solid var(--border-subtle) !important; border-radius: 14px !important; background: rgba(14,21,38,.6); }
div[data-testid="stAlert"] { border-radius: 12px; }

/* Segmented control / radio-pills / tabs */
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; font-size: 0.86rem; color: var(--text-secondary); }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--brand-cyan); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--brand-cyan); height: 3px; border-radius: 3px; }

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.4rem; }
    .hero { padding: 24px 22px; border-radius: 18px; }
    .hero h1 { font-size: 1.7rem; }
    .hero-art { opacity: 0.25; }
    .page-title { font-size: 1.45rem; }
    .kpi-value { font-size: 1.7rem; }
    .rh-text { font-size: 1.5rem; }
    .model-table { display: block; overflow-x: auto; white-space: nowrap; }
}
"""


def apply_custom_css() -> None:
    """Injects the global design system CSS into the Streamlit app."""
    st.markdown(f"<style>{_root_vars()}{_STATIC_CSS}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Plotly theme
# ------------------------------------------------------------------------------
_GRID = "rgba(148, 163, 184, 0.10)"
_AXIS = "rgba(148, 163, 184, 0.28)"


def get_plotly_layout(
    title: str = "",
    height: int = 320,
    show_legend: bool = True,
    x_title: str = "",
    y_title: str = "",
) -> dict:
    """Shared dark, transparent Plotly layout so every chart matches the UI."""
    return dict(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=13, color=COLORS["text_secondary"])) if title else None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text_secondary"], size=12),
        margin=dict(l=8, r=8, t=44 if title else 14, b=8),
        showlegend=show_legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color=COLORS["text_secondary"]), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=11, color=COLORS["text_muted"])),
            gridcolor=_GRID, linecolor=_AXIS, zeroline=False, tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=11, color=COLORS["text_muted"])),
            gridcolor=_GRID, linecolor=_AXIS, zeroline=False, tickfont=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#0B1226", bordercolor=COLORS["brand_cyan"],
            font=dict(family="Inter, sans-serif", size=12, color=COLORS["text_primary"]),
        ),
    )


def create_probability_gauge(prob: float, title: str = "Failure Probability") -> go.Figure:
    """Semi-circular gauge for failure probability (green / amber / red zones)."""
    prob_pct = float(prob) * 100.0
    if prob < 0.30:
        bar_color = COLORS["status_normal"]
    elif prob < 0.65:
        bar_color = COLORS["status_warning"]
    else:
        bar_color = COLORS["status_danger"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "valueformat": ".1f",
                    "font": {"size": 40, "color": COLORS["text_primary"], "family": "JetBrains Mono"}},
            title={"text": title.upper(), "font": {"size": 11, "color": COLORS["text_muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_muted"],
                         "tickvals": [0, 30, 65, 100], "tickfont": {"size": 10}},
                "bar": {"color": bar_color, "thickness": 0.30},
                "bgcolor": "rgba(20, 29, 51, 0.6)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16, 217, 160, 0.16)"},
                    {"range": [30, 65], "color": "rgba(245, 165, 36, 0.16)"},
                    {"range": [65, 100], "color": "rgba(255, 77, 94, 0.18)"},
                ],
                "threshold": {"line": {"color": "#FFFFFF", "width": 3}, "thickness": 0.8, "value": 50},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"], family="Inter"),
        height=250, margin=dict(l=24, r=24, t=48, b=6),
    )
    return fig
