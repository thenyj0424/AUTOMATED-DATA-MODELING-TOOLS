from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from ai_agent.copilot_utils import add_agent_activity, add_chat_message, build_hybrid_step_hint, summarize_results_for_ai
from ai_agent.workflow_tools import build_report_payload


def _format_metric_value(value: Any) -> str:
	if value is None:
		return "—"
	if isinstance(value, (int, float)):
		return f"{value:.4f}" if abs(float(value)) < 1000 else f"{value:.2f}"
	return str(value)


def _pick_primary_metric(results: Dict[str, Any]) -> Tuple[str, Any]:
	metric_label = results.get("metric_label")
	baseline = results.get("baseline", {}) if isinstance(results, dict) else {}
	metrics = baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
	if results.get("problem_type") == "time_series":
		ts = results.get("time_series", {}) if isinstance(results, dict) else {}
		metrics = ts.get("metrics", {}) if isinstance(ts, dict) else {}
		for metric_name, metric_key in (("RMSE", "rmse"), ("MAE", "mae"), ("R2", "r2")):
			if metric_key in metrics:
				return metric_name, metrics.get(metric_key)
		if metrics:
			first_key = next(iter(metrics.keys()))
			return str(first_key).upper(), metrics.get(first_key)
		return "RMSE", None
	if metric_label and metric_label in metrics:
		return metric_label, metrics.get(metric_label)
	if metrics:
		first_key = next(iter(metrics.keys()))
		return first_key, metrics.get(first_key)
	return "Score", None


def _extract_top_features(insights: Dict[str, Any], limit: int = 3) -> List[Tuple[str, float]]:
	rows: List[Tuple[str, float]] = []
	for key in ("feature_importances", "coefficients"):
		entries = insights.get(key) or []
		for entry in entries:
			feature = entry.get("feature")
			value = entry.get("value")
			if feature is None or value is None:
				continue
			try:
				rows.append((str(feature), float(abs(value))))
			except Exception:
				continue
		if rows:
			break
	rows.sort(key=lambda item: item[1], reverse=True)
	return rows[:limit]


def _build_ai_insight_text(results: Dict[str, Any]) -> str:
	if results.get("problem_type") == "time_series":
		return _build_time_series_insight_text(results)
	metric_name, metric_value = _pick_primary_metric(results)
	insights = results.get("baseline_insights", {}) or {}
	top_features = _extract_top_features(insights, limit=3)
	metric_sentence = f"The baseline model scored {_format_metric_value(metric_value)} on {metric_name}."
	if top_features:
		feature_names = ", ".join(name for name, _ in top_features)
		feature_sentence = f"The top 3 features influencing the target are {feature_names}."
	else:
		feature_sentence = "The model insight view does not expose feature ranking for this run, so the strongest available signals are shown below."
	return f"{metric_sentence} {feature_sentence}"


def _build_time_series_metric_rows(ts: Dict[str, Any]) -> List[Tuple[str, Any]]:
	metrics = ts.get("metrics", {}) if isinstance(ts, dict) else {}
	rows: List[Tuple[str, Any]] = []
	for metric_name in ("rmse", "mae", "r2"):
		if metric_name in metrics:
			rows.append((metric_name.upper(), metrics.get(metric_name)))
	for key, value in metrics.items():
		if str(key).lower() not in {"rmse", "mae", "r2"}:
			rows.append((str(key).upper(), value))
	return rows


def _build_time_series_frame(ts: Dict[str, Any]) -> pd.DataFrame:
	y_true = list(ts.get("y_true", []) or [])
	preds = list(ts.get("preds", []) or [])
	residuals = list(ts.get("residuals", []) or [])
	row_count = min(len(y_true), len(preds))
	frame = pd.DataFrame(
		{
			"row": list(range(1, row_count + 1)),
			"actual": y_true[:row_count],
			"predicted": preds[:row_count],
		}
	)
	if residuals:
		frame["residual"] = residuals[:row_count]
	return frame


def _build_time_series_insight_text(results: Dict[str, Any]) -> str:
	ts = results.get("time_series", {}) if isinstance(results, dict) else {}
	metric_rows = _build_time_series_metric_rows(ts)
	model_details = ts.get("model_details", {}) if isinstance(ts, dict) else {}
	parts: List[str] = []
	if metric_rows:
		parts.append(
			"Forecast metrics: "
			+ ", ".join(f"{name}={_format_metric_value(value)}" for name, value in metric_rows[:3])
		)
	if model_details:
		compact_details = []
		for key in ("trend", "seasonal", "seasonal_periods", "order", "seasonal_order", "selection"):
			if key in model_details:
				compact_details.append(f"{key}={model_details[key]}")
		if compact_details:
			parts.append("Model setup: " + "; ".join(compact_details))
	if not parts:
		return "The time-series forecast completed successfully."
	return " ".join(parts)


def _render_insights(insights: Dict[str, Any]) -> None:
	if not insights:
		return
	with st.expander("Model Insights", expanded=True):
		if insights.get("coefficients"):
			st.write("Top coefficients")
			st.dataframe(pd.DataFrame(insights["coefficients"]), use_container_width=True)
		if insights.get("feature_importances"):
			st.write("Top feature importances")
			st.dataframe(pd.DataFrame(insights["feature_importances"]), use_container_width=True)
		if insights.get("support_vectors"):
			st.write(f"Support vectors: {insights['support_vectors']}")
		if insights.get("n_neighbors"):
			st.write(f"Neighbors: {insights['n_neighbors']}")


def _render_classification_artifacts(artifacts: Dict[str, Any], label: str) -> None:
	if not artifacts:
		return
	with st.expander(label):
		if artifacts.get("confusion_matrix"):
			fig, ax = plt.subplots(figsize=(4, 3))
			sns.heatmap(artifacts["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
			ax.set_xlabel("Predicted")
			ax.set_ylabel("Actual")
			st.pyplot(fig, use_container_width=True)
		roc = artifacts.get("roc_curve")
		if roc:
			fig, ax = plt.subplots(figsize=(4, 3))
			ax.plot(roc["fpr"], roc["tpr"], color="blue")
			ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
			ax.set_xlabel("FPR")
			ax.set_ylabel("TPR")
			st.pyplot(fig, use_container_width=True)
		pp = artifacts.get("proba_positive")
		if pp:
			fig, ax = plt.subplots(figsize=(4, 3))
			sns.histplot(pp, kde=True, ax=ax)
			ax.set_xlabel("Predicted probability")
			st.pyplot(fig, use_container_width=True)


def _render_regression_artifacts(artifacts: Dict[str, Any], show_predictions: bool, title: str) -> None:
	if not artifacts:
		return
	with st.expander(title):
		if show_predictions and artifacts.get("y_true"):
			fig, ax = plt.subplots(figsize=(4, 3))
			sns.scatterplot(x=artifacts["y_true"], y=artifacts["preds"], ax=ax)
			ax.set_xlabel("Actual")
			ax.set_ylabel("Predicted")
			st.pyplot(fig, use_container_width=True)
		if artifacts.get("residuals"):
			fig, axes = plt.subplots(1, 2, figsize=(6, 3))
			sns.scatterplot(x=artifacts["preds"], y=artifacts["residuals"], ax=axes[0])
			axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
			sns.histplot(artifacts["residuals"], kde=True, ax=axes[1])
			st.pyplot(fig, use_container_width=True)


def _render_selected_features(features: List[str], label: str = "Selected Features") -> None:
	with st.expander(label):
		st.dataframe(pd.DataFrame({"feature": features}), use_container_width=True)


def render_results_view(
	results: Optional[Dict[str, Any]],
	results_tuned: Optional[Dict[str, Any]],
	summary: Any,
	tuned_models_list: List[Dict[str, Any]],
	clear_tuned_state: Callable[[], None],
	render_system_message: Callable[[str, Any], None],
) -> None:
	if not results:
		st.info("Run analysis to see results.")
		st.stop()

	st.markdown(
		"""
		<style>
		.driver-card {
			border: 1px solid rgba(255,255,255,0.08);
			border-radius: 14px;
			padding: 1rem 1rem 0.9rem 1rem;
			background: rgba(255,255,255,0.03);
			height: 100%;
		}
		.driver-label {
			font-size: 0.85rem;
			font-weight: 700;
			opacity: 0.75;
			margin-bottom: 0.55rem;
		}
		.driver-name {
			font-size: 1.15rem;
			font-weight: 700;
			line-height: 1.2;
			word-break: break-word;
			overflow-wrap: anywhere;
			min-height: 2.6em;
		}
		.driver-score {
			margin-top: 0.4rem;
			font-size: 0.95rem;
			color: #4ade80;
			font-weight: 600;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.markdown("## Results")
	st.caption("A clear view of baseline performance, model drivers, and next-step insight.")

	metric_name, metric_value = _pick_primary_metric(results)
	primary_insight = _build_ai_insight_text(results)
	top_features = _extract_top_features(results.get("baseline_insights", {}) or {}, limit=3)
	st.session_state["report_payload"] = build_report_payload(results, results_tuned, summary)

	header_left, header_mid, header_right = st.columns([1.1, 1.1, 1.2])
	with header_left:
		st.metric("Primary metric", metric_name, _format_metric_value(metric_value))
	with header_mid:
		st.metric("Selected features", len(results.get("selected_features", []) or []))
	with header_right:
		st.markdown("**Top drivers**")
		if top_features:
			st.markdown("<div style='white-space: normal; line-height: 1.35;'>" + "<br>".join(f"• {name}" for name, _ in top_features) + "</div>", unsafe_allow_html=True)
		else:
			st.caption("Unavailable")

	insight_col, action_col = st.columns([1.6, 1])
	with insight_col:
		st.markdown("### AI Insight")
		st.info(primary_insight)
	with action_col:
		st.markdown("### Quick Actions")
		if st.button("Request AI Insight", key="request_ai_insight"):
			try:
				st.session_state["agent_last_outcome"] = summarize_results_for_ai(results)
				add_chat_message("user", "Please summarize key findings and actionable next steps for these results.")
				hint = build_hybrid_step_hint(4, None, summary)
				add_chat_message("assistant", hint)
				add_agent_activity("AI insight requested for Results page.")
				st.success("AI insight added to chat and activity panel.")
			except Exception as exc:
				st.warning(f"AI insight not available. Check LLM configuration or the results context. ({exc})")

	problem_type = results.get("problem_type")

	if problem_type == "time_series":
		ts = results.get("time_series", {})
		st.subheader("Time Series Results")
		st.caption(f"{results.get('model_name', '')} · {results.get('time_series_mode', '')}".strip(" ·"))
		metric_name, metric_value = _pick_primary_metric(results)
		header_cols = st.columns([1.0, 1.0, 1.2])
		with header_cols[0]:
			st.metric("Primary metric", metric_name, _format_metric_value(metric_value))
		with header_cols[1]:
			st.metric("Test points", len(ts.get("y_true", []) or []))
		with header_cols[2]:
			st.metric("Forecast points", len(ts.get("preds", []) or []))

		st.markdown("### Forecast summary")
		st.info(_build_time_series_insight_text(results))

		metric_rows = _build_time_series_metric_rows(ts)
		if metric_rows:
			st.markdown("### Metrics")
			metric_cols = st.columns(min(3, len(metric_rows)))
			for idx, (name, value) in enumerate(metric_rows[:3]):
				with metric_cols[idx]:
					st.metric(name, _format_metric_value(value))

		model_details = ts.get("model_details", {})
		if model_details:
			st.markdown("### Model Setup")
			setup_cols = st.columns(2)
			with setup_cols[0]:
				st.json(model_details)
			with setup_cols[1]:
				st.write("Selected model:")
				st.write(results.get("model_name", ""))

		forecast_frame = _build_time_series_frame(ts)
		if not forecast_frame.empty:
			st.markdown("### Forecast Table")
			st.dataframe(forecast_frame, use_container_width=True, hide_index=True)
			st.download_button(
				"Download forecast table (CSV)",
				forecast_frame.to_csv(index=False),
				file_name="time_series_forecast.csv",
			)

		if show_predictions and ts.get("y_true") and ts.get("preds"):
			st.markdown("### Actual vs Predicted")
			fig, ax = plt.subplots(figsize=(9, 4))
			ax.plot(ts["y_true"], label="Actual", linewidth=2)
			ax.plot(ts["preds"], label="Predicted", linewidth=2)
			ax.set_xlabel("Test observation")
			ax.set_ylabel("Target value")
			ax.legend()
			st.pyplot(fig, use_container_width=True)

		if ts.get("residuals"):
			st.markdown("### Residual Checks")
			fig, axes = plt.subplots(1, 2, figsize=(10, 4))
			if len(ts.get("preds", [])) == len(ts.get("residuals", [])):
				sns.scatterplot(x=ts["preds"], y=ts["residuals"], ax=axes[0])
			else:
				st.warning("Residual plot skipped because predictions and residuals have different lengths.")
			axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
			axes[0].set_xlabel("Predicted")
			axes[0].set_ylabel("Residual")
			sns.histplot(ts["residuals"], kde=True, ax=axes[1])
			axes[1].set_title("Residual Distribution")
			st.pyplot(fig, use_container_width=True)

		col_a, col_b = st.columns([1, 1])
		with col_a:
			if st.button("Back"):
				st.session_state["step"] = 3
				st.session_state["scroll_to_top"] = True
				st.rerun()
		with col_b:
			if st.button("Start Over", key="start_over_results"):
				st.session_state["step"] = 0
				st.session_state["scroll_to_top"] = True
				st.rerun()
		st.stop()

	left_col, right_col = st.columns(2, gap="large")

	with left_col:
		st.subheader("Baseline Performance")
		st.caption(f"{results.get('model_name', '')} · {results.get('metric_label', '')}")
		st.dataframe(pd.DataFrame(results["baseline"]["metrics"], index=[0]).T.reset_index(names="metric"), use_container_width=True, hide_index=True)

		baseline_artifacts = results.get("baseline_artifacts", {})
		baseline_insights = results.get("baseline_insights", {})
		_render_selected_features(results.get("selected_features", []))

		if problem_type == "classification":
			_render_classification_artifacts(baseline_artifacts, "Charts")
		elif problem_type == "regression":
			_render_regression_artifacts(baseline_artifacts, show_predictions, "Charts")

		st.markdown("### Why this model matters")
		_render_insights(baseline_insights)

	st.markdown("---")
	st.subheader("Model Drivers")
	if top_features:
		driver_cols = st.columns(len(top_features), gap="medium")
		for idx, (feature, score) in enumerate(top_features):
			with driver_cols[idx]:
				st.markdown(
					f"""
					<div class="driver-card">
						<div class="driver-label">Top {idx + 1}</div>
						<div class="driver-name">{feature}</div>
						<div class="driver-score">↑ {score:.4f}</div>
					</div>
					""",
					unsafe_allow_html=True,
				)
	else:
		st.info("Top feature ranking is not available for this model.")

	col_a, col_b = st.columns([1, 1])
	with col_a:
		if st.button("Back"):
			st.session_state["step"] = 3
			st.session_state["scroll_to_top"] = True
			st.rerun()
	with col_b:
		if st.button("Start Over"):
			st.session_state["step"] = 0
			st.session_state["scroll_to_top"] = True
			st.rerun()
