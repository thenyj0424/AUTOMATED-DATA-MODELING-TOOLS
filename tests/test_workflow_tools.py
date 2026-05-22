import pandas as pd

from ai_agent.workflow_tools import (
    build_report_payload,
    build_workflow_profile,
    perform_eda,
    recommend_eda_actions,
    recommend_model_setup,
)


def test_build_workflow_profile_detects_dataset_shape():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "value": [1.0, 2.0, 3.0],
            "category": ["a", "b", "a"],
        }
    )
    profile = build_workflow_profile(df)
    assert profile.rows == 3
    assert profile.cols == 3
    assert "value" in profile.numeric_cols
    assert "category" in profile.categorical_cols
    assert "time" in profile.datetime_cols
    assert profile.problem_type == "time_series"


def test_recommend_eda_actions_returns_actionable_sections():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [5, 6, 7, 8],
            "target": [0, 1, 0, 1],
        }
    )
    rec = recommend_eda_actions(df, target_col="target")
    assert rec["problem_type"] == "classification"
    assert "summary" in rec["recommended_sections"]
    assert "correlation" in rec["recommended_sections"]
    assert "target_relationships" in rec["recommended_sections"]


def test_recommend_eda_actions_prunes_heavy_visuals_for_wide_data():
    df = pd.DataFrame({f"col_{idx}": list(range(8)) for idx in range(10)})
    rec = recommend_eda_actions(df)
    assert "pairplot" not in rec["recommended_sections"]
    assert "correlation" in rec["recommended_sections"]


def test_perform_eda_includes_stationarity_for_time_series():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=20, freq="D"),
            "target": list(range(20)),
        }
    )
    result = perform_eda(df, target_col="target")
    assert result["problem_type"] == "time_series"
    assert "stationarity" in result
    assert "recommended_sections" in result
    assert "acf_pacf" in result["recommended_sections"]


def test_recommend_model_setup_prefers_supported_defaults():
    df = pd.DataFrame(
        {
            "feature_1": [1, 2, 3],
            "feature_2": [3, 2, 1],
            "target": [0, 1, 0],
        }
    )
    rec = recommend_model_setup(df, target_col="target")
    assert rec["problem_type"] == "classification"
    assert rec["model_name"] in {"Logistic Regression", "Decision Tree"}
    assert rec["model_family"] in {"Statistical", "ML"}


def test_build_report_payload_contains_export_ready_sections():
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    summary = build_workflow_profile(df, target_col="target")
    payload = build_report_payload(
        {
            "problem_type": "classification",
            "model_name": "Decision Tree",
            "metric_label": "Accuracy",
            "selected_features": ["x"],
            "baseline": {"metrics": {"Accuracy": 0.95}},
            "baseline_insights": {"feature_importances": [{"feature": "x", "value": 1.0}]},
        },
        {"baseline": {"metrics": {"Accuracy": 0.97}}},
        summary,
    )
    assert payload["problem_type"] == "classification"
    assert payload["model_name"] == "Decision Tree"
    assert payload["baseline_metrics"]["Accuracy"] == 0.95
    assert payload["tuned_metrics"]["Accuracy"] == 0.97
    assert payload["summary"]["rows"] == 2