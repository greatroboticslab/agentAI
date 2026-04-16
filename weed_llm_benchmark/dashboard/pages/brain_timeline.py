"""Brain Timeline — Visualize every action the Brain took."""

import streamlit as st
from dashboard.data_loader import parse_slurm_actions
from dashboard.components.charts import action_timeline, tool_usage_pie


def render(data):
    st.title("🧠 Brain Decision Timeline")
    st.markdown("Every action the Brain chose, with reasoning and results.")

    logs = data.get("slurm_logs", [])
    if not logs:
        st.warning("No SLURM logs found. Sync from cluster first.")
        return

    # Select log file
    log_names = [l["name"] for l in logs]
    selected = st.selectbox("Select Job Log", log_names, index=len(log_names) - 1)
    selected_path = next(l["path"] for l in logs if l["name"] == selected)

    # Parse actions
    actions = parse_slurm_actions(selected_path)
    if not actions:
        st.info("No Brain actions found in this log.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    brain_actions = [a for a in actions if a["action"] not in
                     ("ROUND_START", "EVAL_RESULT", "FILTER_RESULT", "LORA_INJECT")]
    rounds = [a for a in actions if a["action"] == "ROUND_START"]

    col1.metric("Total Actions", len(brain_actions))
    col2.metric("Rounds", len(rounds))
    unique = len(set(a["action"] for a in brain_actions))
    col3.metric("Unique Tools Used", unique)

    st.divider()

    # Timeline and pie side by side
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(action_timeline(actions, f"Actions from {selected}"),
                        use_container_width=True)
    with col2:
        st.plotly_chart(tool_usage_pie(actions), use_container_width=True)

    # Detailed action table
    st.subheader("Action Details")
    for i, a in enumerate(actions):
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
