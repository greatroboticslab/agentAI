"""Training Progress — YOLO training loss and validation curves."""

import streamlit as st
import os
import glob


def render(data):
    st.title("📈 Training Progress")
    st.markdown("YOLO training metrics per epoch (from training runs).")

    base_dir = data.get("base_dir", "")
    framework_dir = os.path.join(base_dir, "results", "framework")

    # Find YOLO training result directories
    train_dirs = sorted(glob.glob(os.path.join(framework_dir, "yolo_*/train*/results.csv")))
    lora_dirs = sorted(glob.glob(os.path.join(framework_dir, "yolo_lora_*/train*/results.csv")))
    all_csvs = train_dirs + lora_dirs

    if not all_csvs:
        st.info("No training CSV files found locally. Training results are on the cluster.")
        st.markdown("""
        To view training progress:
        1. Sync results from cluster: `scp -r byler@bridges2:/.../results/framework/yolo_* results/framework/`
        2. Refresh this page
        """)
        return

    # Let user pick which training run to view
    csv_names = [os.path.relpath(c, framework_dir) for c in all_csvs]
    selected = st.selectbox("Select Training Run", csv_names)
    csv_path = os.path.join(framework_dir, selected)

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Loss curves
        st.subheader("Training Loss")
        loss_cols = [c for c in df.columns if "loss" in c.lower() and "train" in c.lower()]
        if loss_cols:
            st.line_chart(df[loss_cols])

        # Validation metrics
        st.subheader("Validation Metrics")
        val_cols = [c for c in df.columns if "mAP" in c or "precision" in c.lower() or "recall" in c.lower()]
        if val_cols:
            st.line_chart(df[val_cols])

        # Raw data
        with st.expander("Raw Data"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
