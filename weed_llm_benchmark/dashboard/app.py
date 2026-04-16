"""
Weed Optimizer Dashboard — Real-time monitoring for the autonomous agent framework.

Usage:
    cd weed_llm_benchmark
    streamlit run dashboard/app.py
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Weed Optimizer Dashboard",
    page_icon="🌿",
    layout="wide",
)

# Load data once
from dashboard.data_loader import load_all_data

@st.cache_data(ttl=60)
def get_data():
    return load_all_data()

data = get_data()

# --- Single-page app with tabs instead of multi-page ---
st.sidebar.title("🌿 Weed Optimizer")
st.sidebar.markdown("**Autonomous Agent Framework**")
st.sidebar.markdown("MTSU Great Robotics Lab")
st.sidebar.markdown("PSC Bridges-2 · V100-32GB")
st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

tab_names = [
    "🏠 Overview",
    "🧠 Brain Timeline",
    "📊 Experiments",
    "🏷️ Labels",
    "⚖️ Models",
    "🔬 Species",
    "💾 Memory",
    "🏗️ Architecture",
    "🖥️ Cluster",
]

tabs = st.tabs(tab_names)

with tabs[0]:
    from dashboard.pages_impl import render_overview
    render_overview(data)

with tabs[1]:
    from dashboard.pages_impl import render_brain_timeline
    render_brain_timeline(data)

with tabs[2]:
    from dashboard.pages_impl import render_experiment_history
    render_experiment_history(data)

with tabs[3]:
    from dashboard.pages_impl import render_label_quality
    render_label_quality(data)

with tabs[4]:
    from dashboard.pages_impl import render_model_comparison
    render_model_comparison(data)

with tabs[5]:
    from dashboard.pages_impl import render_species_comparison
    render_species_comparison(data)

with tabs[6]:
    from dashboard.pages_impl import render_memory_display
    render_memory_display(data)

with tabs[7]:
    from dashboard.pages_impl import render_architecture
    render_architecture(data)

with tabs[8]:
    from dashboard.pages_impl import render_cluster_status
    render_cluster_status(data)
