"""Label Quality — VLM comparison, consensus stats, filtering rates."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


VLM_DATA = {
    "Florence-2-base": {"precision": 0.789, "recall": 0.519, "mAP50": 0.434, "params": "0.23B"},
    "Florence-2-large": {"precision": 0.692, "recall": 0.431, "mAP50": 0.329, "params": "0.77B"},
    "OWLv2-large": {"precision": 0.194, "recall": 0.943, "mAP50": 0.184, "params": "0.4B"},
    "InternVL2-8B": {"precision": 0.545, "recall": 0.354, "mAP50": 0.208, "params": "8B"},
    "Qwen2.5-VL-3B": {"precision": 0.333, "recall": 0.249, "mAP50": 0.196, "params": "3B"},
    "MiniCPM-V-4.5": {"precision": 0.407, "recall": 0.340, "mAP50": 0.192, "params": "4.5B"},
    "Qwen2.5-VL-7B": {"precision": 0.334, "recall": 0.214, "mAP50": 0.176, "params": "7B"},
}


def render(data):
    st.title("🏷️ Label Quality")

    # VLM Precision vs Recall
    st.subheader("VLM Precision vs Recall")
    rows = [{"Model": k, **v} for k, v in VLM_DATA.items()]
    df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=df["Model"], y=df["precision"],
                         marker_color="#2196F3"))
    fig.add_trace(go.Bar(name="Recall", x=df["Model"], y=df["recall"],
                         marker_color="#FF9800"))
    fig.update_layout(barmode="group", template="plotly_white", height=400,
                      title="VLM Precision vs Recall on CottonWeedDet12")
    st.plotly_chart(fig, use_container_width=True)

    # Best pair highlight
    st.info("**Best Consensus Pair**: Florence-2-base (P=0.789) + OWLv2 (R=0.943) — "
            "complementary precision-recall")

    # Label noise stats
    st.subheader("Label Noise & Filtering")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw FP Rate", "27.4%", help="Florence-2 false positive rate")
    col2.metric("After Filter (conf>0.7)", "16.3% removed")
    col3.metric("After Filter (conf>0.8)", "22.8% removed")

    st.markdown("""
    **Filtering pipeline:**
    1. VLMs detect weeds → ~27% are false positives
    2. Multi-VLM consensus (2+ models agree) → reduces noise
    3. YOLO self-training filter (conf>0.8) → removes additional noise
    4. Remaining labels used for YOLO training
    """)

    # Full table
    st.subheader("All VLM Results")
    st.dataframe(df.round(3), use_container_width=True, hide_index=True)
