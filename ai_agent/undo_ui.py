import streamlit as st


def render_undo_panel():
    """Show recent AI-driven changes and provide a 'Revert All AI Actions' control.
    The actual revert is performed by the main loop to avoid circular imports.
    """
    stack = st.session_state.get("agent_undo_stack", [])
    if not stack:
        st.info("No AI-driven actions to undo.")
        return
    st.markdown("**Recent AI-driven actions (most recent last)**")
    # Show up to 40 most recent entries with per-entry Revert buttons
    start = max(0, len(stack) - 40)
    for idx in range(start, len(stack)):
        key, existed, old_value = stack[idx]
        label = f"{key} — {'present' if existed else 'absent'}"
        cols = st.columns([8, 1])
        with cols[0]:
            st.write(f"{idx+1}. {label}")
        with cols[1]:
            if st.button("Revert", key=f"revert_entry_{idx}"):
                st.session_state["agent_request_revert_index_pending"] = idx

        # If a revert is pending for this index, show confirmation
        pending = st.session_state.get("agent_request_revert_index_pending")
        if pending is not None and pending == idx:
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Confirm Revert", key=f"confirm_revert_{idx}"):
                    st.session_state["agent_request_revert_index"] = idx
                    st.session_state.pop("agent_request_revert_index_pending", None)
                    st.success(f"Revert confirmed for {key}.")
            with c2:
                if st.button("Cancel", key=f"cancel_revert_{idx}"):
                    st.session_state.pop("agent_request_revert_index_pending", None)
                    st.info("Revert cancelled.")

    st.markdown("---")
    if st.button("Revert All AI Actions", key="revert_all_ai"):
        st.session_state["agent_request_revert_all_pending"] = True

    # Confirmation for revert all
    if st.session_state.get("agent_request_revert_all_pending"):
        st.warning("Are you sure? This will revert ALL AI-driven changes. This action cannot be undone easily.")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Confirm Revert All", key="confirm_revert_all"):
                st.session_state["agent_request_revert_all"] = True
                st.session_state.pop("agent_request_revert_all_pending", None)
                st.success("Revert all confirmed. The app will now apply reverts.")
        with c2:
            if st.button("Cancel", key="cancel_revert_all"):
                st.session_state.pop("agent_request_revert_all_pending", None)
                st.info("Revert all cancelled.")
