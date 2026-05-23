import streamlit as st
from main import (
    clear_workflow_state,
    push_undo_snapshot,
    undo_last_agent_action,
    revert_agent_action_at,
    normalize_selector_state,
)


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


def test_normalize_selector_state_maps_legacy_alias():
    setup_state()
    st.session_state["time_series_model"] = "holt_winters_additive"
    normalize_selector_state(
        "time_series_model",
        ["ARIMA", "SARIMA", "Holt-Winters"],
        aliases={"holt_winters_additive": "Holt-Winters"},
        default="ARIMA",
    )
    assert st.session_state["time_series_model"] == "Holt-Winters"


def test_normalize_selector_state_defaults_when_invalid():
    setup_state()
    st.session_state["tuning_method"] = "BayesianSearch"
    normalize_selector_state(
        "tuning_method",
        ["None", "Grid Search", "Manual"],
        default="Grid Search",
    )
    assert st.session_state["tuning_method"] == "Grid Search"


def test_clear_workflow_state_clears_stale_model_setup():
    setup_state()
    st.session_state.update(
        {
            "agent_goal": "use random forest",
            "agent_requirements": [{"text": "use random forest"}],
            "agent_step_autorun": {3: True},
            "model_family": "ML",
            "model_name": "Random Forest",
            "feature_selection": "RFE",
            "tuning_method": "Grid Search",
            "time_series_model": "holt_winters_additive",
            "problem_type": "classification",
            "target_col": "target",
            "selected_features": ["a", "b"],
        }
    )
    clear_workflow_state(keep_original=False)
    assert st.session_state.get("step") == 0
    assert "agent_goal" not in st.session_state
    assert "agent_requirements" not in st.session_state
    assert "agent_step_autorun" not in st.session_state
    assert "model_family" not in st.session_state
    assert "model_name" not in st.session_state
    assert "feature_selection" not in st.session_state
    assert "tuning_method" not in st.session_state
    assert "time_series_model" not in st.session_state
    assert "problem_type" not in st.session_state
    assert "target_col" not in st.session_state


def test_clear_workflow_state_preserves_original_dataset_by_default():
    setup_state()
    st.session_state.update(
        {
            "df_original": "original-df",
            "df": "working-df",
            "summary": "summary",
            "model_name": "Random Forest",
            "feature_selection": "RFE",
            "tuning_method": "Grid Search",
        }
    )
    clear_workflow_state()
    assert st.session_state.get("df_original") == "original-df"
    assert "df" not in st.session_state
    assert "summary" not in st.session_state
    assert "model_name" not in st.session_state
    assert "feature_selection" not in st.session_state
    assert "tuning_method" not in st.session_state
