import pandas as pd
import numpy as np

from ai_agent.auto_engine import apply_auto_actions_snapshot


def make_df(with_nans=False):
    N = 30
    rng = pd.date_range("2020-01-01", periods=N, freq="D")
    df = pd.DataFrame({"date": rng, "value": np.arange(N)})
    if with_nans:
        df.loc[3:6, "value"] = np.nan
    return df


def make_tabular_df():
    N = 30
    return pd.DataFrame(
        {
            "feature_a": np.arange(N),
            "feature_b": np.arange(N) * 2,
            "year": np.arange(2000, 2000 + N),
            "month": np.tile(np.arange(1, 13), 3)[:N],
            "target": np.arange(N) + 1,
        }
    )


def test_apply_auto_actions_step1_basic():
    df = make_tabular_df()
    state = {}
    changes, activities = apply_auto_actions_snapshot(state, 1, df)
    assert isinstance(changes, dict)
    assert isinstance(activities, list)
    assert changes.get("eda_target_col") == "target"
    assert changes.get("eda_target_feature") in {"feature_a", "feature_b"}
    assert changes.get("eda_show_summary") is True
    assert changes.get("eda_show_corr") is True
    assert changes.get("eda_show_missing") is True
    assert changes.get("eda_show_target_dist") is True
    assert changes.get("eda_show_target_rel") is True
    assert any("target pair" in a.lower() for a in activities)


def test_apply_auto_actions_step2_cleaning_detected():
    df = make_df(with_nans=True)
    state = {}
    changes, activities = apply_auto_actions_snapshot(state, 2, df)
    # Expect cleaning decisions or preview keys
    assert isinstance(changes, dict)
    # At least one activity message should be present describing cleaning
    # match 'clean' or 'imput' to accept both 'impute' and 'imputation'
    assert any("clean" in a.lower() or "imput" in a.lower() for a in activities)


def test_no_step_advance_in_changes():
    df = make_df()
    state = {"step": 1}
    changes, activities = apply_auto_actions_snapshot(state, 1, df)
    # Engine should not change control flow 'step'
    assert "step" not in changes


def test_apply_auto_actions_step3_model_defaults():
    df = make_tabular_df()
    state = {}
    changes, activities = apply_auto_actions_snapshot(state, 3, df)
    assert "model_family" in changes
    assert "model_name" in changes
    assert changes.get("proceed_model") is True
    assert changes.get("feature_year") is False
    assert changes.get("feature_month") is False
    assert changes.get("tuning_method") == "Grid Search"


def test_apply_auto_actions_step3_time_series_defaults():
    df = make_df()
    state = {}
    changes, activities = apply_auto_actions_snapshot(state, 3, df)
    assert changes.get("problem_type") == "time_series"
    assert changes.get("time_series_model") == "ARIMA"
    assert changes.get("time_series_mode") == "Auto ARIMA"


def test_apply_auto_actions_step3_prefers_user_model_family():
    df = make_tabular_df()
    state = {"agent_preferences": {"preferred_model_family": "ML"}}
    changes, activities = apply_auto_actions_snapshot(state, 3, df)
    assert changes.get("problem_type") == "regression"
    assert changes.get("model_family") == "ML"
    assert any("Applied user preference for model family: ML." in a for a in activities)
