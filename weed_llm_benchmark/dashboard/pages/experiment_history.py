"""Experiment History — mAP/F1 curves over iterations."""

import streamlit as st
import pandas as pd
from dashboard.data_loader import get_experiments_df
from dashboard.components.charts import metric_line_chart, forgetting_scatter


def render(data):
    st.title("📊 Experiment History")

    experiments = get_experiments_df()
    if not experiments:
        st.warning("No experiments found in memory.json")
        return

    df = pd.DataFrame(experiments)
    baseline = data.get("memory", {}).get("baseline", {})

    # Metric selector
    metric_group = st.radio("Metric Group", ["F1", "mAP@0.5", "mAP@0.5:0.95"], horizontal=True)

    if metric_group == "F1":
        y_cols = ["old_f1", "new_f1"]
        threshold = 0.90
    elif metric_group == "mAP@0.5":
        y_cols = ["old_map50", "new_map50"]
        threshold = None
    else:
        y_cols = ["old_map50_95", "new_map50_95"]
        threshold = None

    # Main chart
    fig = metric_line_chart(experiments, y_cols,
                            f"{metric_group} Over Iterations", threshold=threshold)
    st.plotly_chart(fig, use_container_width=True)

    # Forgetting scatter
    st.subheader("Forgetting Analysis")
    fig2 = forgetting_scatter(experiments)
    st.plotly_chart(fig2, use_container_width=True)

    # Summary stats
    st.subheader("Summary Statistics")
    if len(df) > 1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best New F1", f"{df['new_f1'].max():.4f}",
                       delta=f"{df['new_f1'].max() - baseline.get('new_f1', 0):+.4f}")
        with col2:
            st.metric("Best New mAP50", f"{df['new_map50'].max():.4f}",
                       delta=f"{df['new_map50'].max() - baseline.get('new_map50', 0):+.4f}")
        with col3:
            no_forget = len(df[~df["forgetting"]])
            st.metric("No-Forgetting Runs", f"{no_forget}/{len(df)}")

    # Full table
    st.subheader("All Experiments")
    st.dataframe(df.round(4), use_container_width=True, hide_index=True)
