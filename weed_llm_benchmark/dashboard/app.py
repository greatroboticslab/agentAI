"""
Weed Optimizer Dashboard — Real-time monitoring for the autonomous agent framework.

Usage:
    cd weed_llm_benchmark
    streamlit run dashboard/app.py

Features:
    - Brain decision timeline (actions, reasoning, tool calls)
    - Experiment history with mAP/F1 curves
    - Label quality visualization (consensus, filtering, VLM comparison)
    - Training progress (loss, validation metrics per epoch)
    - Old vs New species comparison
    - Model comparison (baseline vs freeze vs LoRA vs two-pass)
    - Memory / lessons learned display
    - Cluster job status
"""

import streamlit as st
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Weed Optimizer Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Navigation ---
st.sidebar.title("🌿 Weed Optimizer")
st.sidebar.markdown("**Autonomous Agent Framework**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🧠 Brain Timeline",
        "📊 Experiment History",
        "🏷️ Label Quality",
        "📈 Training Progress",
        "🔬 Species Comparison",
        "⚖️ Model Comparison",
        "💾 Memory & Lessons",
        "🖥️ Cluster Status",
        "🏗️ Architecture",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("MTSU Great Robotics Lab")
st.sidebar.caption("PSC Bridges-2 · V100-32GB")

# --- Load Data ---
from dashboard.data_loader import load_all_data

data = load_all_data()

# --- Route to Pages ---
if page == "🏠 Overview":
    from dashboard.pages.overview import render
    render(data)
elif page == "🧠 Brain Timeline":
    from dashboard.pages.brain_timeline import render
    render(data)
elif page == "📊 Experiment History":
    from dashboard.pages.experiment_history import render
    render(data)
elif page == "🏷️ Label Quality":
    from dashboard.pages.label_quality import render
    render(data)
elif page == "📈 Training Progress":
    from dashboard.pages.training_progress import render
    render(data)
elif page == "🔬 Species Comparison":
    from dashboard.pages.species_comparison import render
    render(data)
elif page == "⚖️ Model Comparison":
    from dashboard.pages.model_comparison import render
    render(data)
elif page == "💾 Memory & Lessons":
    from dashboard.pages.memory_display import render
    render(data)
elif page == "🖥️ Cluster Status":
    from dashboard.pages.cluster_status import render
    render(data)
elif page == "🏗️ Architecture":
    from dashboard.pages.architecture import render
    render(data)
