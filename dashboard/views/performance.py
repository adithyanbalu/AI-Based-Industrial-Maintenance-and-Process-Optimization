"""Page: Model Performance. Metrics come from outputs/model_comparison.csv and models/model_metadata.json."""

import pandas as pd
import streamlit as st

from dashboard.components import (
    COLORS,
    chart_baseline_vs_improved,
    chart_confusion_matrix,
    chart_feature_importances,
    chart_model_metrics,
    chart_roc_comparison_interactive,
    esc,
    kpi_grid,
    load_comparison_data,
    md,
    meter,
    page_header,
    panel_title,
    render_alert_box,
    section_header,
    show_chart,
)

comp_df, meta = load_comparison_data()
metrics = meta.get("metrics", {})
base = meta.get("baseline_comparison", {})
cm = metrics.get("confusion_matrix", {})

if comp_df.empty and not metrics:
    st.error("No evaluation results found. Run `python src/model_comparison.py` and `python src/predictive_model.py` first.")
    st.stop()

best_name = None
if not comp_df.empty:
    best_name = comp_df.loc[comp_df["F1-Score (Failure)"].idxmax(), "Model"]

page_header(
    "Machine-learning evaluation",
    "Model Performance",
    "Six candidate algorithms evaluated on the same stratified 20% hold-out set. The metric that matters most is recall on the "
    "failure class — a missed failure is far costlier than a false alarm.",
    chips=[(f"Selected: {meta.get('model_name', best_name)}", "cyan"), ("Test set N = 2,000", ""), ("Stratified split · random_state 42", "")],
)

# ------------------------------------------------------------------ Selected model KPIs
if metrics:
    section_header("Selected model", meta.get("model_name", ""), "workspace_premium")
    kpi_grid([
        dict(title="Accuracy", value=f"{metrics['accuracy'] * 100:.2f}", unit="%", subtext="Overall, both classes", icon_name="verified", accent=COLORS["accent_cyan"], progress=metrics["accuracy"] * 100),
        dict(title="Precision", value=f"{metrics['precision_failure'] * 100:.1f}", unit="%", subtext="Failure class · alarms that were real", icon_name="target", accent=COLORS["status_warning"], progress=metrics["precision_failure"] * 100),
        dict(title="Recall", value=f"{metrics['recall_failure'] * 100:.1f}", unit="%", subtext="Failure class · failures caught", icon_name="crisis_alert", accent=COLORS["status_danger"], progress=metrics["recall_failure"] * 100),
        dict(title="F1-score", value=f"{metrics['f1_failure']:.4f}", subtext="Failure class · balance", icon_name="balance", accent=COLORS["status_normal"], progress=metrics["f1_failure"] * 100),
        dict(title="ROC-AUC", value=f"{metrics['roc_auc']:.4f}", subtext="Ranking quality", icon_name="show_chart", accent=COLORS["brand_cyan"], progress=metrics["roc_auc"] * 100),
    ])

# ------------------------------------------------------------------ Algorithm comparison table
if not comp_df.empty:
    section_header("Algorithm comparison", "All candidates are shown, including the weaker ones. Cyan = best in column · amber = weak", "compare_arrows")

    cols = [("Accuracy", "Accuracy"), ("Precision (Failure)", "Precision"), ("Recall (Failure)", "Recall"),
            ("F1-Score (Failure)", "F1"), ("ROC-AUC", "ROC-AUC"), ("False Negatives (FN)", "Missed (FN)"), ("False Positives (FP)", "False alarms (FP)")]
    best_per_col = {}
    for c, _ in cols:
        if c in comp_df.columns:
            best_per_col[c] = comp_df[c].min() if c in ("False Negatives (FN)", "False Positives (FP)") else comp_df[c].max()

    head = "<tr><th>Model</th>" + "".join(f"<th>{lbl}</th>" for _, lbl in cols) + "</tr>"
    body = ""
    for _, r in comp_df.iterrows():
        name = str(r["Model"])
        tag = ""
        if name == best_name:
            tag += '<span class="tag best">Best</span>'
        if "Improved" in name:
            tag += '<span class="tag imp">Improved</span>'
        elif "Baseline" in name:
            tag += '<span class="tag base">Baseline</span>'
        elif "Balanced" in name:
            tag += '<span class="tag base">Balanced</span>'
        tds = ""
        for c, _ in cols:
            if c not in comp_df.columns:
                continue
            v = r[c]
            cls = ""
            if v == best_per_col[c]:
                cls = "top"
            elif c in ("Precision (Failure)", "Recall (Failure)", "F1-Score (Failure)"):
                cls = "bad" if v < 0.25 else ("weak" if v < 0.5 else "")
            txt = f"{int(v)}" if "(FN)" in c or "(FP)" in c else f"{v:.4f}"
            tds += f'<td class="{cls}">{txt}</td>'
        body += f'<tr class="{"best" if name == best_name else ""}"><td>{esc(name)}{tag}</td>{tds}</tr>'
    md(f'<table class="model-table"><thead>{head}</thead><tbody>{body}</tbody></table>')

    # Honest trade-off note derived from the table
    weak = comp_df[(comp_df["Recall (Failure)"] > 0.8) & (comp_df["Precision (Failure)"] < 0.5)]
    if not weak.empty:
        names = ", ".join(weak["Model"])
        render_alert_box("Trade-off to be aware of",
                         f"{esc(names)} reach high recall but with very low precision (many false alarms). "
                         f"The improved Random Forest gives the best balance between the two.", "warning")

    g1, g2 = st.columns([1.15, 1], gap="medium")
    with g1:
        with st.container(key="panel_metrics"):
            panel_title("Metric comparison", "Every metric for every algorithm")
            show_chart(chart_model_metrics(comp_df), key="pf_metrics")
    with g2:
        with st.container(key="panel_roc"):
            panel_title("ROC curves", "Recomputed from the project's candidate pipelines on the test split")
            show_chart(chart_roc_comparison_interactive(), key="pf_roc")

# ------------------------------------------------------------------ Failure-class performance + confusion matrix
if metrics and cm:
    section_header("Failure-class performance", "How well the selected model finds real failures", "fact_check")
    h1, h2 = st.columns([1, 1], gap="medium")
    tp, fn, fp, tn = cm["TP"], cm["FN"], cm["FP"], cm["TN"]
    with h1:
        with st.container(key="panel_cm"):
            panel_title("Confusion matrix", f"{meta.get('model_name', '')} · test set N = {tp + fn + fp + tn:,}")
            show_chart(chart_confusion_matrix(cm), key="pf_cm")
    with h2:
        with st.container(key="panel_fc"):
            panel_title("Failure-class results")
            md(
                meter("Recall — failures caught", metrics["recall_failure"], 1.0, f"{tp} of {tp + fn}  ·  {metrics['recall_failure'] * 100:.1f}%", COLORS["status_normal"],
                      note=f"{fn} real failures were missed (false negatives)")
                + meter("Precision — alarms that were real", metrics["precision_failure"], 1.0, f"{tp} of {tp + fp}  ·  {metrics['precision_failure'] * 100:.1f}%", COLORS["status_warning"],
                        note=f"{fp} healthy machines were flagged by mistake (false positives)")
                + meter("F1-score", metrics["f1_failure"], 1.0, f"{metrics['f1_failure']:.3f}", COLORS["brand_cyan"],
                        note="Harmonic mean of precision and recall for the failure class")
                + meter("Specificity — healthy machines cleared", tn / (tn + fp), 1.0, f"{tn:,} of {tn + fp:,}  ·  {tn / (tn + fp) * 100:.1f}%", COLORS["accent_cyan"],
                        note="Share of normal machines correctly left alone")
            )

# ------------------------------------------------------------------ Baseline vs improved
if metrics and base:
    section_header("Baseline vs improved model", "Same algorithm family; the improved version adds balanced class weights, tuned depth and physics-derived features", "trending_up")
    i1, i2 = st.columns([1.5, 1], gap="medium")
    with i1:
        with st.container(key="panel_bi"):
            panel_title("Baseline Random Forest vs Improved Random Forest")
            show_chart(chart_baseline_vs_improved(meta), key="pf_bi")
    with i2:
        with st.container(key="panel_delta"):
            panel_title("What changed")
            rows = [("Recall", base["baseline_recall"], metrics["recall_failure"]), ("Precision", base["baseline_precision"], metrics["precision_failure"]),
                    ("F1-score", base["baseline_f1"], metrics["f1_failure"]), ("ROC-AUC", base["baseline_roc_auc"], metrics["roc_auc"]),
                    ("Accuracy", base["baseline_accuracy"], metrics["accuracy"])]
            md("".join(
                f'<div class="meter"><div class="meter-top"><span class="meter-name">{n}</span>'
                f'<span class="meter-val num" style="color:{COLORS["status_normal"] if b_ <= a_ else COLORS["status_danger"]}">{b_:.4f} → {a_:.4f} ({(a_ - b_) * 100:+.2f} pp)</span></div></div>'
                for n, b_, a_ in rows
            ))
            md(f'<div class="callout success" style="margin-top:4px;"><b>Missed failures:</b> {base["baseline_fn"]} → {base["improved_fn"]} '
               f'({base["baseline_fn"] - base["improved_fn"]} fewer breakdowns undetected).</div>')

# ------------------------------------------------------------------ Feature importances
fi = meta.get("feature_importances", [])
if fi:
    section_header("What the model relies on", "Feature importance of the selected Random Forest", "psychology")
    with st.container(key="panel_fi"):
        show_chart(chart_feature_importances(fi), key="pf_fi")

# ------------------------------------------------------------------ Accuracy paradox (computed)
if not comp_df.empty and "Logistic Regression (Baseline)" in set(comp_df["Model"]):
    lr = comp_df[comp_df["Model"] == "Logistic Regression (Baseline)"].iloc[0]
    tot_f = int(lr["False Negatives (FN)"] + lr["True Positives (TP)"])
    render_alert_box(
        "Why accuracy alone is not enough",
        f"Baseline Logistic Regression scores {lr['Accuracy'] * 100:.2f}% accuracy yet catches only {int(lr['True Positives (TP)'])} of {tot_f} failures "
        f"({lr['Recall (Failure)'] * 100:.1f}% recall). On imbalanced industrial data, failure-class recall, precision and ROC-AUC are the metrics that decide model quality.",
        "info",
    )
