import streamlit as st
import pandas as pd

from ai_agent.copilot_utils import (
    build_copilot_context,
    build_dataset_analysis_context,
    build_meta_help_reply,
    condense_user_facing_text,
    is_dataset_question,
    is_meta_help_request,
    is_requirement_message,
    sanitize_user_facing_text,
    strip_internal_plan_payload,
)
from ai_agent.data_utils import build_summary


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
    assert "Auto AI" in ctx
    assert "requirement" in ctx.lower()


def test_sanitize_user_facing_text_removes_self_interpretation():
    text = sanitize_user_facing_text("I think the user wants a faster model because the user would like better accuracy.")
    assert "the user wants" not in text.lower()
    assert "the request is" in text.lower()


def test_meta_help_reply_is_conversational():
    assert is_meta_help_request("explain my request")
    reply = build_meta_help_reply("explain my request", "Use a time series model")
    assert "Auto AI" not in reply
    assert "checklist" in reply.lower() or "steps" in reply.lower()


def test_strip_internal_plan_payload_removes_json():
    text = "No changes are being made. {\"changes\": {}, \"activities\": [\"Noted\"], \"priority_source\": \"user\"} Please provide further clarification."
    cleaned = strip_internal_plan_payload(text)
    assert "{" not in cleaned
    assert "}" not in cleaned
    assert "priority_source" not in cleaned
    assert "activities" not in cleaned


def test_condense_user_facing_text_trims_long_reply():
    text = "First. Second. Third. Fourth. Fifth. Sixth."
    cleaned = condense_user_facing_text(text, max_sentences=3, max_words=20)
    assert cleaned.count(".") <= 3
    assert "Fourth" not in cleaned


def test_apply_requirement_to_state_sets_default_ml_model():
    # Ensure session_state is clean
    for k in list(st.session_state.keys()):
        if k.startswith("model_") or k in {"problem_type", "agent_requirements"}:
            del st.session_state[k]
    from ai_agent.copilot_utils import apply_requirement_to_state
    notes = apply_requirement_to_state("Please use any ML model for this task")
    # Do not force `model_family` or `model_name`; instead a flexible hint should be set.
    assert "model_name" not in st.session_state
    assert st.session_state.get("model_selection_hint", {}).get("family") == "ML"
    assert any("prefer ML models" in n or "preference hint" in n for n in notes)


def test_apply_requirement_to_state_sets_statistical_hint():
    for k in list(st.session_state.keys()):
        if k.startswith("model_") or k in {"problem_type", "agent_requirements", "model_selection_hint"}:
            del st.session_state[k]
    from ai_agent.copilot_utils import apply_requirement_to_state
    notes = apply_requirement_to_state("Please prefer statistical methods for this analysis")
    assert "model_name" not in st.session_state
    assert st.session_state.get("model_selection_hint", {}).get("family") == "Statistical"
    assert any("Statistical" in n for n in notes)


def test_apply_requirement_to_state_sets_performance_hint():
    for k in list(st.session_state.keys()):
        if k.startswith("model_") or k in {"problem_type", "agent_requirements", "model_selection_hint"}:
            del st.session_state[k]
    from ai_agent.copilot_utils import apply_requirement_to_state
    notes = apply_requirement_to_state("I need a fast, lightweight solution prioritizing speed over accuracy")
    hint = st.session_state.get("model_selection_hint", {})
    # prefer speed; we represent it as a tradeoff flag in the hint
    assert hint.get("family") in (None, "ML", "Statistical") or hint.get("flexible") is True
    assert any("speed" in n.lower() or "fast" in n.lower() for n in notes) or hint.get("speed") == "high"


def test_dataset_question_detection_and_context():
    st.session_state.clear()
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    summary = build_summary(df)
    st.session_state["target_col"] = "target"
    assert is_dataset_question("Can you summarize this dataset and suggest a model?", 1, df, summary)
    bundle = build_dataset_analysis_context(1, df, summary, "Can you summarize this dataset and suggest a model?")
    assert bundle["target_col"] == "target"
    assert "recommended_sections" in bundle["prompt_context"]
