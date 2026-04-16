"""Overview page — KPI cards, latest results, system status."""

import streamlit as st
import pandas as pd


def render(data):
    st.title("🌿 Weed Optimizer Framework — Overview")
    st.markdown("**Autonomous agent for YOLO weed detection optimization**")

    memory = data.get("memory", {})
    baseline = memory.get("baseline", {})
    best = memory.get("current_best", {})
    experiments = memory.get("experiments", [])
    lessons = memory.get("learned_lessons", [])

    # --- KPI Cards ---
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Experiments", len(experiments))
    with col2:
        st.metric("Lessons Learned", len(lessons) + len(memory.get("hard_lessons", [])))
    with col3:
        best_new = best.get("new_f1", 0)
        base_new = baseline.get("new_f1", 0)
        delta = round(best_new - base_new, 4) if base_new else None
        st.metric("Best New F1", f"{best_new:.4f}", delta=f"{delta:+.4f}" if delta else None)
    with col4:
        best_old = best.get("old_f1", 0)
        st.metric("Best Old F1", f"{best_old:.4f}",
                   delta="No Forgetting" if best_old >= 0.90 else "Forgetting",
                   delta_color="normal" if best_old >= 0.90 else "inverse")

    st.divider()

    # --- Baseline vs Best ---
    st.subheader("Baseline vs Current Best")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline (8-species YOLO, no adaptation)**")
        if baseline:
            bcols = st.columns(3)
            bcols[0].metric("Old F1", f"{baseline.get('old_f1', 0):.4f}")
            bcols[1].metric("Old mAP50", f"{baseline.get('old_map50', 0):.4f}")
            bcols[2].metric("New mAP50", f"{baseline.get('new_map50', 0):.4f}")
        else:
            st.info("No baseline data available")

    with col2:
        st.markdown("**Current Best**")
        if best:
            bcols = st.columns(3)
            bcols[0].metric("Old F1", f"{best.get('old_f1', 0):.4f}")
            bcols[1].metric("Old mAP50", f"{best.get('old_map50', 0):.4f}")
            bcols[2].metric("New mAP50", f"{best.get('new_map50', 0):.4f}")
        else:
            st.info("No experiments completed yet")

    st.divider()

    # --- Recent Experiments Table ---
    st.subheader("Recent Experiments")
    if experiments:
        rows = []
        for e in experiments[-10:]:
            r = e.get("result", {})
            s = e.get("strategy", {})
            rows.append({
                "Iter": e.get("iteration", 0),
                "Name": s.get("name", "?")[:40],
                "Old F1": round(r.get("old_f1", 0), 4),
                "New F1": round(r.get("new_f1", 0), 4),
                "Old mAP50": round(r.get("old_map50", 0), 4),
                "New mAP50": round(r.get("new_map50", 0), 4),
                "Forgetting": "Yes" if r.get("forgetting") else "No",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No experiments yet")

    # --- SLURM Logs ---
    st.subheader("Cluster Logs")
    logs = data.get("slurm_logs", [])
    if logs:
        for log in logs[-5:]:
            st.text(f"{log['name']}  ({log['size_mb']} MB, {log['modified'][:16]})")
    else:
        st.info("No SLURM logs found locally. Sync from cluster to view.")
