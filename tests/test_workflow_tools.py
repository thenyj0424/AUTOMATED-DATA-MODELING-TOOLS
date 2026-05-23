import pandas as pd

from ai_agent.workflow_tools import (
    analyze_correlations,
    analyze_heteroscedasticity,
    analyze_multicollinearity,
    analyze_normality,
    build_report_payload,
    build_workflow_profile,
    collect_dataset_diagnostics,
    perform_eda,
    recommend_eda_actions,
    recommend_model_setup,
    run_statistical_tools_on_demand,
    should_run_statistical_tools,
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
    assert "correlation_analysis" in result
    assert "recommended_sections" in result
    assert "acf_pacf" in result["recommended_sections"]


def test_analyze_correlations_returns_sorted_top_pairs():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "z": [5, 4, 3, 2, 1],
        }
    )
    corr = analyze_correlations(df, top_k=2)
    assert corr["is_applicable"] is True
    assert len(corr["top_pairs"]) == 2
    assert corr["top_pairs"][0]["abs_corr"] >= corr["top_pairs"][1]["abs_corr"]


def test_collect_dataset_diagnostics_includes_tool_outputs():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=20, freq="D"),
            "target": list(range(20)),
            "feature": list(range(20, 0, -1)),
        }
    )
    diagnostics = collect_dataset_diagnostics(df, target_col="target")
    assert "eda" in diagnostics
    assert "stationarity" in diagnostics
    assert "correlation_analysis" in diagnostics
    assert "model_setup" in diagnostics


def test_statistical_tools_trigger_and_run_on_demand():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "x2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            "target": [1.0, 1.8, 3.2, 3.9, 5.1, 5.9, 7.3, 8.1, 8.8, 10.2, 11.1, 12.0],
        }
    )
    assert should_run_statistical_tools("run normality and VIF checks") is True
    assert should_run_statistical_tools("hello there") is False
    result = run_statistical_tools_on_demand(df, user_text="check stationarity and correlation", target_col="target")
    assert "stationarity_analysis" in result
    assert "correlation_analysis" in result
    assert "normality_analysis" in result
    assert "multicollinearity_analysis" in result
    assert "heteroscedasticity_analysis" in result


def test_statistical_tools_base_functions_return_structured_payloads():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "x2": [2, 1, 4, 3, 6, 5, 8, 7, 10, 9],
            "target": [1.1, 1.9, 3.2, 3.8, 5.2, 5.8, 7.1, 8.2, 8.9, 10.0],
        }
    )
    normality = analyze_normality(df)
    vif = analyze_multicollinearity(df)
    hetero = analyze_heteroscedasticity(df, target_col="target")
    assert "results" in normality
    assert "vif" in vif
    assert "pvalue" in hetero


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