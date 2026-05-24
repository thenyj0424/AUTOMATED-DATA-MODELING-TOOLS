import streamlit as st

from ai_agent.copilot_utils import (
    build_copilot_context,
    format_preferences_for_context,
    update_preferences_from_text,
)


def setup_state():
    for key in list(st.session_state.keys()):
        if key.startswith("agent_"):
            del st.session_state[key]


def test_update_preferences_from_text_detects_ml_preference():
    prefs, updates = update_preferences_from_text("I prefer ML model and care about accuracy", {})
    assert prefs.get("preferred_model_family") == "ML"
    assert prefs.get("preferred_model_name") == "Random Forest"
    assert prefs.get("priority") == "accuracy"
    assert any("Random Forest" in item for item in updates)


def test_update_preferences_from_text_switches_to_statistical_model():
    prefs, updates = update_preferences_from_text(
        "I want statistical model instead",
        {"preferred_model_family": "ML", "preferred_model_name": "Random Forest"},
    )
    assert prefs.get("preferred_model_family") == "Statistical"
    assert prefs.get("preferred_model_name") in {"Logistic Regression", "Linear Regression"}
    assert any("Statistical" in item for item in updates)


def test_build_copilot_context_includes_stored_preferences():
    setup_state()
    st.session_state["agent_preferences"] = {
        "preferred_model_family": "ML",
        "priority": "speed",
    }
    st.session_state["agent_goal"] = "Recommend a model"
    text = build_copilot_context(step=3, df=None, summary=None)
    assert "Stored user preferences" in text
    assert "Model family preference: ML" in text
    assert "Optimization priority: speed" in text


def test_format_preferences_for_context_empty():
    assert format_preferences_for_context({}) == ""
