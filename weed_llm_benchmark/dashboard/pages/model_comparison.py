"""Model Comparison — Baseline vs freeze vs LoRA vs two-pass."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dashboard.components.charts import COLORS


def render(data):
    st.title("⚖️ Model Comparison")
    st.markdown("Compare anti-forgetting methods on the same dataset.")

    results = data.get("results", {})

    # Try to load the three-way comparison
    lora_comp = results.get("framework_lora_eval_comparison", {})

    if lora_comp and "results" in lora_comp:
        methods = lora_comp["results"]
        rows = []
        for name, metrics in methods.items():
            rows.append({
                "Method": name.replace("_", " ").title(),
                "Old F1": metrics.get("old_f1", 0),
                "New F1": metrics.get("new_f1", 0),
                "Old mAP50": metrics.get("old_map50", 0),
                "New mAP50": metrics.get("new_map50", 0),
                "Old mAP50-95": metrics.get("old_map50_95", 0),
                "New mAP50-95": metrics.get("new_map50_95", 0),
                "Forgetting": "Yes" if metrics.get("forgetting") else "No",
                "Params%": metrics.get("trainable_params", "100%"),
            })
        df = pd.DataFrame(rows)

        # Grouped bar chart
        metrics_to_plot = st.multiselect(
            "Select Metrics", ["Old F1", "New F1", "Old mAP50", "New mAP50"],
            default=["Old F1", "New F1", "Old mAP50", "New mAP50"])

        fig = go.Figure()
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#00BCD4"]
        for i, metric in enumerate(metrics_to_plot):
            fig.add_trace(go.Bar(name=metric, x=df["Method"], y=df[metric],
                                 marker_color=colors[i % len(colors)]))

        fig.add_hline(y=0.90, line_dash="dash", line_color=COLORS["threshold"],
                       annotation_text="Forgetting Threshold (F1=0.90)")
        fig.update_layout(barmode="group", template="plotly_white", height=450,
                          title="Method Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Parameter efficiency scatter
        st.subheader("Parameter Efficiency")
        st.markdown("Trainable parameters vs detection improvement")
        st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.info("No three-way comparison data found. Run the evaluation job first.")

    # Also show experiments from memory grouped by method
    st.subheader("All Methods Tested (from Memory)")
    memory = data.get("memory", {})
    experiments = memory.get("experiments", [])
    if experiments:
        rows = []
        for e in experiments:
            r = e.get("result", {})
            s = e.get("strategy", {})
            rows.append({
                "Iter": e.get("iteration"),
                "Method": s.get("name", "?")[:35],
                "Old F1": round(r.get("old_f1", 0), 4),
                "New F1": round(r.get("new_f1", 0), 4),
                "Forgetting": "Yes" if r.get("forgetting") else "No",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
