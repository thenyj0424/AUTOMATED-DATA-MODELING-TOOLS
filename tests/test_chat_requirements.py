import streamlit as st
import pandas as pd

from ai_agent.copilot_utils import (
    build_copilot_context,
    build_dataset_analysis_context,
    build_dataset_readiness_bundle,
    build_meta_help_reply,
    build_supported_tool_summary,
    build_unsupported_tool_reply,
    condense_user_facing_text,
    format_conversation_memory_for_context,
    is_short_acknowledgement,
    is_dataset_question,
    is_meta_help_request,
    is_requirement_message,
    is_unsupported_statistical_request,
    is_statistical_diagnostic_request,
    sanitize_user_facing_text,
    strip_internal_plan_payload,
    update_conversation_memory_from_assistant,
    update_conversation_memory_from_user,
)
from ai_agent.data_utils import build_summary


def test_is_requirement_message_detects_labelled():
    assert is_requirement_message("Requirement: The model must be explainable and fast.")
    assert is_requirement_message("Req: Save this as a requirement")


def test_is_requirement_message_detects_declarative():
    assert is_requirement_message("The analytics pipeline must refresh nightly and export CSVs to /data")
    assert not is_requirement_message("Can you suggest a model for this dataset?")


def test_is_requirement_message_rejects_question_like_statements():
    assert not is_requirement_message("What are the requirements for this model setup")
    assert not is_requirement_message("I need to know which model should I use for this dataset")


def test_is_requirement_message_detects_direct_instruction_language():
    assert is_requirement_message("Please use a Random Forest model and keep it fast")
    assert is_requirement_message("We need to forecast this series monthly")


def test_unsupported_statistical_request_detection_and_reply():
    assert is_unsupported_statistical_request("Conduct a hypothesis testing to know whether the mean lot size is less than 3 or not")
    assert not is_unsupported_statistical_request("Run a normality check on the residuals")
    reply = build_unsupported_tool_reply("Conduct a hypothesis testing to know whether the mean lot size is less than 3 or not")
    assert "unavailable" in reply.lower()
    assert "hypothesis" in reply.lower()


def test_statistical_diagnostic_request_detects_natural_language_normality_questions():
    assert is_statistical_diagnostic_request("Is house price normally distributed?")
    assert is_statistical_diagnostic_request("Can you use shapiro wilk to know whether lot size is normally distributed or not?")
    assert is_statistical_diagnostic_request("Is house price constant variance?")
    assert is_statistical_diagnostic_request("heterokedrasticity of house price")
    assert not is_statistical_diagnostic_request("Conduct a hypothesis testing to know whether the mean lot size is less than 3 or not")


def test_supported_tool_summary_mentions_available_tool_families():
    summary = build_supported_tool_summary()
    assert "ML models" in summary
    assert "statistical diagnostics" in summary
    assert "stationarity" in summary.lower()


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
    notes = apply_requirement_to_state("Please use an ML model for this task")
    assert st.session_state.get("model_family") == "ML"
    assert st.session_state.get("model_name") == "Random Forest"
    assert st.session_state.get("model_selection_hint", {}).get("family") == "ML"
    assert any("Random Forest" in n for n in notes)


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


def test_apply_requirement_to_state_defaults_holt_winters_for_general_time_series():
    for k in list(st.session_state.keys()):
        if k.startswith("model_") or k.startswith("hw_") or k in {"problem_type", "agent_requirements", "model_selection_hint"}:
            del st.session_state[k]
    from ai_agent.copilot_utils import apply_requirement_to_state
    notes = apply_requirement_to_state("Forecast this series")
    assert st.session_state.get("problem_type") == "time_series"
    assert st.session_state.get("time_series_model") == "Holt-Winters"
    assert st.session_state.get("hw_trend") == "add"
    assert st.session_state.get("hw_seasonal") == "add"
    assert any("Holt-Winters" in n or "time_series_model" in n for n in notes)


def test_dataset_question_detection_and_context():
    st.session_state.clear()
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    summary = build_summary(df)
    st.session_state["target_col"] = "target"
    assert is_dataset_question("Can you summarize this dataset and suggest a model?", 1, df, summary)
    bundle = build_dataset_analysis_context(1, df, summary, "Can you summarize this dataset and suggest a model?")
    assert bundle["target_col"] == "target"
    assert bundle["bundle"]["high_level"]["model_name"]
    assert "tool_outputs" in bundle["bundle"]
    assert "recommended_sections" in bundle["prompt_context"]


def test_dataset_analysis_context_runs_statistical_tools_only_when_requested():
    st.session_state.clear()
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "x2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            "target": [1.0, 1.9, 3.1, 4.0, 5.0, 6.2, 7.1, 8.0, 9.1, 9.8, 11.0, 12.2],
        }
    )
    summary = build_summary(df)
    st.session_state["target_col"] = "target"
    plain = build_dataset_analysis_context(1, df, summary, "Which model is suitable?")
    stats = build_dataset_analysis_context(1, df, summary, "Can you run stationarity and VIF checks?")
    normality = build_dataset_analysis_context(1, df, summary, "Is house price normally distributed?")
    variance = build_dataset_analysis_context(1, df, summary, "Is house price constant variance?")
    assert plain["on_demand_tools"] == {}
    assert "multicollinearity_analysis" in stats["on_demand_tools"]
    assert "stationarity_analysis" in stats["on_demand_tools"]
    assert "normality_analysis" in normality["on_demand_tools"]
    assert "heteroscedasticity_analysis" in variance["on_demand_tools"]


def test_dataset_readiness_bundle_and_copilot_context_use_cache():
    st.session_state.clear()
    df = pd.DataFrame({"feature_a": [1, 2, 3], "feature_b": [3, 2, 1], "target": [0, 1, 0]})
    summary = build_summary(df)
    bundle = build_dataset_readiness_bundle(df, summary)
    st.session_state["dataset_readiness_bundle"] = bundle
    st.session_state["agent_goal"] = "show me what to do next"
    ctx = build_copilot_context(1, df, summary)
    assert "Dataset readiness" in ctx
    assert bundle["high_level"]["model_name"] in ctx
    assert bundle["signals"]["correlation_ready"] is True


def test_conversation_memory_tracks_ack_and_recommendation():
    st.session_state.clear()
    update_conversation_memory_from_assistant("I recommend Random Forest for this dataset.")
    update_conversation_memory_from_user("OK")
    memory = st.session_state.get("agent_conversation_memory", {})
    assert memory.get("last_user_ack") is True
    assert "random forest" in str(memory.get("last_recommendation", "")).lower()
    compact = format_conversation_memory_for_context(memory)
    assert "last_recommendation" in compact
    assert "last_confirmation" in compact


def test_short_ack_detection():
    assert is_short_acknowledgement("OK") is True
    assert is_short_acknowledgement("go ahead") is True
    assert is_short_acknowledgement("Can you suggest a model?") is False


def test_copilot_context_includes_conversation_memory():
    st.session_state.clear()
    st.session_state["agent_conversation_memory"] = {
        "last_recommendation": "Decision Tree",
        "last_user_confirmation": "ok",
        "last_assistant_reply": "I recommend Decision Tree for this dataset.",
    }
    ctx = build_copilot_context(1, None, None)
    assert "Conversation memory" in ctx
    assert "Decision Tree" in ctx
