"""All dashboard page implementations in one file (avoids Streamlit multi-page detection)."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
from dashboard.data_loader import get_experiments_df, parse_slurm_actions, load_memory
from dashboard.components.charts import (
    metric_line_chart, forgetting_scatter, action_timeline,
    tool_usage_pie, COLORS, TOOL_COLORS,
)


# ======================================================================
# OVERVIEW
# ======================================================================
def render_overview(data):
    st.header("🌿 Weed Optimizer Framework")
    memory = data.get("memory", {})
    baseline = memory.get("baseline", {})
    best = memory.get("current_best", {})
    experiments = memory.get("experiments", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments", len(experiments))
    c2.metric("Lessons", len(memory.get("learned_lessons", [])) + len(memory.get("hard_lessons", [])))
    best_new = best.get("new_f1", 0)
    base_new = baseline.get("new_f1", 0)
    c3.metric("Best New F1", f"{best_new:.4f}",
              delta=f"{best_new - base_new:+.4f}" if base_new else None)
    best_old = best.get("old_f1", 0)
    c4.metric("Best Old F1", f"{best_old:.4f}",
              delta="OK" if best_old >= 0.90 else "Forgetting",
              delta_color="normal" if best_old >= 0.90 else "inverse")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline")
        if baseline:
            st.json(baseline)
    with col2:
        st.subheader("Current Best")
        if best:
            st.json(best)

    if experiments:
        st.subheader("Recent Experiments")
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
                "Forget": "Yes" if r.get("forgetting") else "No",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ======================================================================
# BRAIN TIMELINE
# ======================================================================
def render_brain_timeline(data):
    st.header("🧠 Brain Decision Timeline")

    logs = data.get("slurm_logs", [])
    if not logs:
        st.warning("No SLURM logs found locally. Sync from cluster to view Brain actions.")
        st.code("scp byler@bridges2:/.../results/framework/slurm_ollama_*.out results/framework/")
        return

    log_names = [l["name"] for l in logs]
    selected = st.selectbox("Select Job Log", log_names, index=len(log_names) - 1)
    selected_path = next(l["path"] for l in logs if l["name"] == selected)

    actions = parse_slurm_actions(selected_path)
    if not actions:
        st.info("No Brain actions found in this log.")
        return

    brain_actions = [a for a in actions if a["action"] not in
                     ("ROUND_START", "EVAL_RESULT", "FILTER_RESULT", "LORA_INJECT")]
    rounds = [a for a in actions if a["action"] == "ROUND_START"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Actions", len(brain_actions))
    c2.metric("Rounds", len(rounds))
    c3.metric("Unique Tools", len(set(a["action"] for a in brain_actions)))

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(action_timeline(actions), use_container_width=True)
    with col2:
        st.plotly_chart(tool_usage_pie(actions), use_container_width=True)

    st.subheader("Action Details")
    for a in actions:
        action = a.get("action", "?")
        reasoning = a.get("reasoning", a.get("detail", ""))
        time = a.get("time", "")
        if action == "ROUND_START":
            st.markdown(f"---\n### Round {a.get('detail', '?')}")
        elif action == "EVAL_RESULT":
            st.success(f"📊 {time} — {reasoning}")
        elif action == "FILTER_RESULT":
            st.warning(f"🔍 {time} — {reasoning}")
        elif action == "LORA_INJECT":
            st.info(f"🔧 {time} — {reasoning}")
        else:
            icon = {"train_yolo": "🏋️", "freeze_train": "🧊", "lora_train": "🔗",
                    "two_pass_train": "🔄", "evaluate": "📊", "analyze_failure": "🔬",
                    "inspect_labels": "👁️", "generate_consensus": "🤝",
                    "filter_labels": "🔍", "done": "✅"}.get(action, "▶️")
            st.markdown(f"{icon} **{time}** — `{action}` {reasoning[:100]}")


# ======================================================================
# EXPERIMENT HISTORY
# ======================================================================
def render_experiment_history(data):
    st.header("📊 Experiment History")
    experiments = get_experiments_df()
    if not experiments:
        st.warning("No experiments found")
        return

    metric = st.radio("Metric", ["F1", "mAP@0.5", "mAP@0.5:0.95"], horizontal=True)
    y_map = {"F1": ["old_f1", "new_f1"],
             "mAP@0.5": ["old_map50", "new_map50"],
             "mAP@0.5:0.95": ["old_map50_95", "new_map50_95"]}
    threshold = 0.90 if metric == "F1" else None

    st.plotly_chart(metric_line_chart(experiments, y_map[metric], f"{metric} Over Iterations", threshold),
                    use_container_width=True)
    st.plotly_chart(forgetting_scatter(experiments), use_container_width=True)
    st.dataframe(pd.DataFrame(experiments).round(4), hide_index=True)


# ======================================================================
# LABEL QUALITY
# ======================================================================
def render_label_quality(data):
    st.header("🏷️ Label Quality")

    vlm_data = {
        "Florence-2-base": {"precision": 0.789, "recall": 0.519, "mAP50": 0.434},
        "Florence-2-large": {"precision": 0.692, "recall": 0.431, "mAP50": 0.329},
        "OWLv2-large": {"precision": 0.194, "recall": 0.943, "mAP50": 0.184},
        "InternVL2-8B": {"precision": 0.545, "recall": 0.354, "mAP50": 0.208},
        "Qwen2.5-VL-3B": {"precision": 0.333, "recall": 0.249, "mAP50": 0.196},
        "MiniCPM-V-4.5": {"precision": 0.407, "recall": 0.340, "mAP50": 0.192},
        "Qwen2.5-VL-7B": {"precision": 0.334, "recall": 0.214, "mAP50": 0.176},
    }
    df = pd.DataFrame([{"Model": k, **v} for k, v in vlm_data.items()])

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=df["Model"], y=df["precision"], marker_color="#2196F3"))
    fig.add_trace(go.Bar(name="Recall", x=df["Model"], y=df["recall"], marker_color="#FF9800"))
    fig.update_layout(barmode="group", template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Best Pair**: Florence-2 (P=0.789) + OWLv2 (R=0.943)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw FP Rate", "27.4%")
    c2.metric("Filter conf>0.7", "16.3% removed")
    c3.metric("Filter conf>0.8", "22.8% removed")


# ======================================================================
# MODEL COMPARISON
# ======================================================================
def render_model_comparison(data):
    st.header("⚖️ Model Comparison")

    results = data.get("results", {})
    lora_comp = results.get("framework_lora_eval_comparison", {})

    if lora_comp and "results" in lora_comp:
        methods = lora_comp["results"]
        rows = []
        for name, m in methods.items():
            rows.append({
                "Method": name.replace("_", " ").title(),
                "Old F1": m.get("old_f1", 0), "New F1": m.get("new_f1", 0),
                "Old mAP50": m.get("old_map50", 0), "New mAP50": m.get("new_map50", 0),
                "Params": m.get("trainable_params", "100%"),
            })
        df = pd.DataFrame(rows)

        fig = go.Figure()
        for i, col in enumerate(["Old F1", "New F1", "Old mAP50", "New mAP50"]):
            fig.add_trace(go.Bar(name=col, x=df["Method"], y=df[col],
                                 marker_color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"][i]))
        fig.add_hline(y=0.90, line_dash="dash", line_color="red", annotation_text="Forgetting Threshold")
        fig.update_layout(barmode="group", template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, hide_index=True)
    else:
        st.info("No comparison data. Run lora evaluation job first.")


# ======================================================================
# SPECIES COMPARISON
# ======================================================================
def render_species_comparison(data):
    st.header("🔬 Old vs New Species")
    experiments = get_experiments_df()
    baseline = data.get("memory", {}).get("baseline", {})
    if not experiments:
        st.warning("No experiments")
        return

    latest = experiments[-1]
    metrics = ["F1", "mAP50", "mAP50-95"]
    old_b = [baseline.get("old_f1", 0), baseline.get("old_map50", 0), baseline.get("old_map50_95", 0)]
    new_b = [baseline.get("new_f1", 0), baseline.get("new_map50", 0), baseline.get("new_map50_95", 0)]
    old_l = [latest["old_f1"], latest["old_map50"], latest.get("old_map50_95", 0)]
    new_l = [latest["new_f1"], latest["new_map50"], latest.get("new_map50_95", 0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Old Baseline", x=metrics, y=old_b, marker_color="#90CAF9"))
    fig.add_trace(go.Bar(name="Old Latest", x=metrics, y=old_l, marker_color="#1565C0"))
    fig.add_trace(go.Bar(name="New Baseline", x=metrics, y=new_b, marker_color="#FFE0B2"))
    fig.add_trace(go.Bar(name="New Latest", x=metrics, y=new_l, marker_color="#E65100"))
    fig.update_layout(barmode="group", template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Known (8 species)**: Carpetweeds, Crabgrass, PalmerAmaranth, PricklySida, Purslane, Ragweed, Sicklepod, SpottedSpurge")
    with c2:
        st.markdown("**Unseen (4 species)**: Eclipta, Goosegrass, Morningglory, Nutsedge")


# ======================================================================
# MEMORY & LESSONS
# ======================================================================
def render_memory_display(data):
    st.header("💾 Memory & Lessons")
    memory = data.get("memory", {})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔴 Hard Lessons")
        for l in memory.get("hard_lessons", []):
            sev = l.get("severity", "info")
            icon = {"critical": "🚫", "high": "⚠️"}.get(sev, "ℹ️")
            with st.expander(f"{icon} {l.get('id', '?')}: {l.get('lesson', '')[:80]}"):
                st.markdown(l.get("lesson", ""))
                st.caption(f"Evidence: {l.get('evidence', '')}")

    with c2:
        st.subheader("🧠 Learned Lessons")
        for l in reversed(memory.get("learned_lessons", [])):
            with st.expander(f"📝 {l.get('id', '?')} — {l.get('timestamp', '')[:16]}"):
                st.markdown(l.get("lesson", ""))

    st.divider()
    st.subheader("Baseline & Best")
    c1, c2 = st.columns(2)
    with c1:
        st.json(memory.get("baseline", {}))
    with c2:
        st.json(memory.get("current_best", {}))


# ======================================================================
# ARCHITECTURE
# ======================================================================
def render_architecture(data):
    st.header("🏗️ Framework Architecture")
    st.markdown("""
```
┌──────────────────────────────────────────────────────────────┐
│                  SuperBrain (Gemma 4 / Qwen3)                │
│   See state → Think → Choose tool → See result → Repeat     │
│                      (14 tools)                              │
└─────────────────────────┬────────────────────────────────────┘
                          │
      ┌───────────────────┼───────────────────────┐
      ▼                   ▼                       ▼
 Label Tools          Training Tools         Eval + Analysis
 ├ VLM inference      ├ train_yolo           ├ evaluate
 ├ consensus          ├ freeze_train         ├ analyze_failure
 ├ filter_labels      ├ lora_train           ├ inspect_labels
 ├ plant.id API       ├ distill_train        └ search_models
 └ run_external       └ two_pass_train

      ↓                   ↓                       ↓
┌──────────────────────────────────────────────────────────────┐
│                    Memory (Persistent JSON)                   │
│   10 hard lessons + learned lessons + experiment history      │
└──────────────────────────────────────────────────────────────┘
```
    """)

    st.subheader("Components (17 files, ~4,500 lines)")
    components = {
        "🧠 brain.py (650)": "Gemma 4 via Ollama. Native function calling. 14 tools.",
        "🔄 orchestrator.py (780)": "Agent loop + strategy mode. Job chaining. Forced progression.",
        "🔗 lora_yolo.py (211)": "Conv2d LoRA. 4 modes: head/backbone/hybrid/all.",
        "📊 evaluator.py (296)": "Dual-conf: mAP@0.001 + F1@0.25.",
        "🏷️ label_gen.py (240)": "Multi-VLM consensus + external models.",
        "🔍 label_filter.py (158)": "YOLO self-training filter (conf>0.8).",
        "🌐 web_identifier.py (252)": "plant.id API + cache.",
        "🔎 model_discovery.py (338)": "HuggingFace search + download.",
        "💾 memory.py (270)": "10 hard lessons + persistent history.",
        "🛡️ monitor.py (198)": "Forgetting detection + strategy validation.",
    }
    for name, desc in components.items():
        with st.expander(name):
            st.markdown(desc)


# ======================================================================
# CLUSTER STATUS
# ======================================================================
def render_cluster_status(data):
    st.header("🖥️ Cluster Status")

    logs = data.get("slurm_logs", [])
    st.subheader(f"SLURM Logs ({len(logs)} found)")
    if logs:
        for log in reversed(logs[-5:]):
            with st.expander(f"📄 {log['name']} ({log['size_mb']} MB)"):
                try:
                    with open(log["path"]) as f:
                        lines = f.readlines()
                    st.code("".join(lines[-30:]), language="text")
                except Exception as e:
                    st.error(str(e))
    else:
        st.info("No logs locally. Run: `scp byler@bridges2:/.../slurm_*.out results/framework/`")

    st.subheader("Cluster Info")
    st.markdown("""
    | Resource | Value |
    |----------|-------|
    | System | PSC Bridges-2 |
    | GPU | 8× V100-32GB per node |
    | RAM | 515 GB |
    | Ollama | v0.20.6 |
    | Brain | Gemma 4 31B (MoE, Q4) |
    """)
