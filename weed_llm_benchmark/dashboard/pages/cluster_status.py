"""Cluster Status — SLURM job history and log viewer."""

import streamlit as st
import os


def render(data):
    st.title("🖥️ Cluster Status")

    logs = data.get("slurm_logs", [])

    # Job history from log files
    st.subheader(f"SLURM Job Logs ({len(logs)} found)")
    if logs:
        for log in reversed(logs):
            with st.expander(f"📄 {log['name']} ({log['size_mb']} MB)"):
                st.caption(f"Modified: {log['modified']}")
                # Show last 50 lines
                try:
                    with open(log["path"]) as f:
                        lines = f.readlines()
                    st.code("".join(lines[-50:]), language="text")
                except Exception as e:
                    st.error(f"Cannot read: {e}")
    else:
        st.info("No SLURM logs found locally.")

    st.divider()

    # Cluster info
    st.subheader("Cluster Configuration")
    st.markdown("""
    | Resource | Value |
    |----------|-------|
    | System | PSC Bridges-2 |
    | GPU | 8× V100-SXM2-32GB per node |
    | System RAM | 515 GB |
    | CPU | 40 cores per node |
    | Storage | /ocean shared filesystem |
    | Conda envs | `bench` (transformers 4.57), `compat` (4.46) |
    | Ollama | v0.20.6 |
    | Brain | Gemma 4 31B (MoE, Q4_K_M) |
    """)

    st.subheader("SSH Command (manual sync)")
    st.code("""
# Sync framework results from cluster
scp -r byler@bridges2.psc.edu:/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/ results/framework/

# Check running jobs
ssh byler@bridges2.psc.edu 'squeue -u byler'
    """, language="bash")
