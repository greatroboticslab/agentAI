"""Memory & Lessons — Hard lessons and learned lessons display."""

import streamlit as st


def render(data):
    st.title("💾 Memory & Lessons Learned")

    memory = data.get("memory", {})
    hard_lessons = memory.get("hard_lessons", [])
    learned_lessons = memory.get("learned_lessons", [])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🔴 Hard Lessons ({len(hard_lessons)})")
        st.markdown("*Never violated — learned from 18+ sessions of experiments*")
        for l in hard_lessons:
            severity = l.get("severity", "info")
            icon = {"critical": "🚫", "high": "⚠️", "info": "ℹ️"}.get(severity, "ℹ️")
            color = {"critical": "red", "high": "orange", "info": "blue"}.get(severity, "gray")

            with st.expander(f"{icon} [{severity.upper()}] {l.get('id', '?')}"):
                st.markdown(f"**{l.get('lesson', '')}**")
                st.caption(f"Evidence: {l.get('evidence', '')}")
                constraint = l.get("constraint", {})
                if constraint:
                    st.json(constraint)

    with col2:
        st.subheader(f"🧠 Learned Lessons ({len(learned_lessons)})")
        st.markdown("*Discovered by the Brain during optimization*")
        if learned_lessons:
            for l in reversed(learned_lessons):
                with st.expander(f"📝 {l.get('id', '?')} — {l.get('timestamp', '')[:16]}"):
                    st.markdown(l.get("lesson", ""))
                    st.caption(f"Evidence: {l.get('evidence', '')}")
                    st.caption(f"Severity: {l.get('severity', 'info')}")
        else:
            st.info("No lessons learned yet (run the framework first)")

    st.divider()

    # Baseline and current best
    st.subheader("📌 Performance Bookmarks")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline**")
        baseline = memory.get("baseline", {})
        if baseline:
            st.json(baseline)
        else:
            st.info("No baseline set")

    with col2:
        st.markdown("**Current Best**")
        best = memory.get("current_best", {})
        if best:
            st.json(best)
        else:
            st.info("No best result yet")

    # Meta info
    meta = memory.get("meta", {})
    if meta:
        st.subheader("🔧 Meta")
        st.json(meta)
