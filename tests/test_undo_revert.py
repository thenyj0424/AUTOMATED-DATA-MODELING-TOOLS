import streamlit as st
from main import push_undo_snapshot, undo_last_agent_action, revert_agent_action_at


def setup_state():
    # Clear any existing session keys used by tests
    for k in list(st.session_state.keys()):
        if k.startswith("test_") or k.startswith("agent_"):
            del st.session_state[k]


def test_push_and_undo_last():
    setup_state()
    # Simulate adding a key
    st.session_state["foo"] = 42
    push_undo_snapshot("foo", 99)
    # After push, undo should revert to old value 99
    msg = undo_last_agent_action()
    assert "Reverted" in msg
    assert st.session_state.get("foo") == 99


def test_push_and_revert_index():
    setup_state()
    # Simulate two pushes
    st.session_state["bar"] = "x"
    push_undo_snapshot("bar", "old_x")
    st.session_state["baz"] = "y"
    push_undo_snapshot("baz", "old_y")
    # Revert the first entry (index 0 corresponds to earliest pushed)
    # Since push_undo_snapshot appends, index 0 will be first pushed
    msg = revert_agent_action_at(0)
    assert "Reverted" in msg
