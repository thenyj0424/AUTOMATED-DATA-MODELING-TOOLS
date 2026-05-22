import streamlit as st


def render_activity_panel(max_items: int = 20) -> None:
    with st.sidebar.expander("AI Activity", expanded=False):
        items = st.session_state.get("agent_activity", [])[:max_items]
        if not items:
            st.caption("No activity yet.")
        else:
            for item in items:
                time = item.get("time", "--:--")
                level = item.get("level", "info")
                msg = item.get("message", "")
                prefix = "INFO" if level == "info" else ("ERROR" if level == "error" else "WARN")
                st.markdown(f"**[{time}] {prefix}:** {msg}")

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Clear", key="clear_agent_activity"):
                st.session_state["agent_activity"] = []
        with cols[1]:
            if st.button("Hide", key="dismiss_activity_panel"):
                st.session_state["agent_activity_panel_hidden"] = True

        if st.button("Undo Last AI Action", key="undo_agent_action"):
            st.session_state["agent_request_undo_pending"] = True

        if st.session_state.get("agent_request_undo_pending"):
            st.warning("Undo the most recent AI-driven change?")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Confirm Undo", key="confirm_undo_agent_action"):
                    st.session_state["agent_request_undo"] = True
                    st.session_state.pop("agent_request_undo_pending", None)
            with c2:
                if st.button("Cancel", key="cancel_undo_agent_action"):
                    st.session_state.pop("agent_request_undo_pending", None)
