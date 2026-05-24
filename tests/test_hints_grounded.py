import pandas as pd
import streamlit as st

from ai_agent.copilot_utils import build_hybrid_step_hint


def clear_hint_state():
    for key in ["target_col", "eda_target_col", "dataset_readiness_bundle", "agent_preferences", "agent_goal"]:
        if key in st.session_state:
            del st.session_state[key]


def test_step_three_hint_does_not_leak_stale_target():
    clear_hint_state()
    df = pd.DataFrame({"Age": [21, 25], "Income": [50000, 62000], "Purchased": [0, 1]})
    st.session_state["target_col"] = "Sales"
    hint = build_hybrid_step_hint(3, df, None)
    assert "Sales" not in hint
    assert "Age" in hint or "Income" in hint or "Purchased" in hint
    assert "current upload" in hint.lower()
