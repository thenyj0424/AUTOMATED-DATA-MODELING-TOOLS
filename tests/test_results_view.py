from ai_agent.results_view import _build_time_series_frame, _build_time_series_insight_text, _pick_primary_metric


def test_pick_primary_metric_prefers_time_series_metrics():
    results = {
        "problem_type": "time_series",
        "time_series": {"metrics": {"rmse": 1.2345, "mae": 0.9876, "r2": 0.4567}},
    }
    metric_name, metric_value = _pick_primary_metric(results)
    assert metric_name == "RMSE"
    assert metric_value == 1.2345


def test_time_series_summary_mentions_real_values():
    results = {
        "problem_type": "time_series",
        "model_name": "Holt-Winters",
        "time_series": {
            "metrics": {"rmse": 1.2345, "mae": 0.9876, "r2": 0.4567},
            "model_details": {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
        },
    }
    text = _build_time_series_insight_text(results)
    assert "RMSE=1.2345" in text
    assert "MAE=0.9876" in text
    assert "trend=add" in text


def test_time_series_frame_includes_predictions_and_residuals():
    ts = {"y_true": [1.0, 2.0], "preds": [0.9, 2.1], "residuals": [0.1, -0.1]}
    frame = _build_time_series_frame(ts)
    assert list(frame.columns) == ["row", "actual", "predicted", "residual"]
    assert frame.iloc[0]["actual"] == 1.0
    assert frame.iloc[1]["predicted"] == 2.1