"""Species Comparison — Old vs New species side-by-side."""

import streamlit as st
import plotly.graph_objects as go
from dashboard.data_loader import get_experiments_df


def render(data):
    st.title("🔬 Old vs New Species Comparison")

    experiments = get_experiments_df()
    baseline = data.get("memory", {}).get("baseline", {})

    if not experiments:
        st.warning("No experiments found")
        return

    # Side-by-side bars for latest experiment
    latest = experiments[-1]

    st.subheader("Latest Experiment vs Baseline")
    fig = go.Figure()

    metrics = ["F1", "mAP50", "mAP50-95"]
    old_baseline = [baseline.get("old_f1", 0), baseline.get("old_map50", 0), baseline.get("old_map50_95", 0)]
    new_baseline = [baseline.get("new_f1", 0), baseline.get("new_map50", 0), baseline.get("new_map50_95", 0)]
    old_latest = [latest["old_f1"], latest["old_map50"], latest.get("old_map50_95", 0)]
    new_latest = [latest["new_f1"], latest["new_map50"], latest.get("new_map50_95", 0)]

    fig.add_trace(go.Bar(name="Old (Baseline)", x=metrics, y=old_baseline, marker_color="#90CAF9"))
    fig.add_trace(go.Bar(name="Old (Latest)", x=metrics, y=old_latest, marker_color="#1565C0"))
    fig.add_trace(go.Bar(name="New (Baseline)", x=metrics, y=new_baseline, marker_color="#FFE0B2"))
    fig.add_trace(go.Bar(name="New (Latest)", x=metrics, y=new_latest, marker_color="#E65100"))

    fig.update_layout(barmode="group", template="plotly_white", height=450,
                      title=f"Baseline vs {latest['name'][:30]}")
    st.plotly_chart(fig, use_container_width=True)

    # Delta cards
    st.subheader("Changes from Baseline")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Old Species (8 known)**")
        st.metric("F1", f"{latest['old_f1']:.4f}",
                   delta=f"{latest['old_f1'] - baseline.get('old_f1', 0):+.4f}")
        st.metric("mAP50", f"{latest['old_map50']:.4f}",
                   delta=f"{latest['old_map50'] - baseline.get('old_map50', 0):+.4f}")

    with col2:
        st.markdown("**New Species (4 unseen)**")
        st.metric("F1", f"{latest['new_f1']:.4f}",
                   delta=f"{latest['new_f1'] - baseline.get('new_f1', 0):+.4f}")
        st.metric("mAP50", f"{latest['new_map50']:.4f}",
                   delta=f"{latest['new_map50'] - baseline.get('new_map50', 0):+.4f}")

    # Species list
    st.subheader("Species Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**8 Known Species (YOLO trained on)**")
        for s in ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
                   "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge"]:
            st.markdown(f"- {s}")
    with col2:
        st.markdown("**4 Unseen Species (held out)**")
        for s in ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]:
            st.markdown(f"- {s}")
