import streamlit as st

from ai_agent.copilot_utils import is_requirement_message, build_copilot_context


def test_is_requirement_message_detects_labelled():
    assert is_requirement_message("Requirement: The model must be explainable and fast.")
    assert is_requirement_message("Req: Save this as a requirement")


def test_is_requirement_message_detects_declarative():
    assert is_requirement_message("The analytics pipeline must refresh nightly and export CSVs to /data")
    assert not is_requirement_message("Can you suggest a model for this dataset?")


def test_build_copilot_context_contains_system_policy():
    # Ensure session_state has minimal keys used in context builder
    st.session_state.clear()
    ctx = build_copilot_context(1, None, None)
    assert "System policy" in ctx
    assert "requirement" in ctx.lower()
