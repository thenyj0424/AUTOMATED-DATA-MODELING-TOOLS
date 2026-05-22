import os
import ast
import html
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_curve
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from ai_agent.config import (
	APP_TITLE,
	STATISTICAL_CLASSIFICATION_MODELS,
	STATISTICAL_REGRESSION_MODELS,
	ML_CLASSIFICATION_MODELS,
	ML_REGRESSION_MODELS,
	STEPWISE_DIRECTIONS,
	MODEL_BASED_OPTIONS,
	TUNING_METHODS,
	CLASSIFICATION_METRICS,
	REGRESSION_METRICS,
	DEFAULT_MAX_ROWS,
	DEFAULT_MAX_FEATURES,
	DEFAULT_CV_FOLDS,
	DEFAULT_TPE_TRIALS,
	MAX_ROWS_CAP,
	MAX_FEATURES_CAP,
	MAX_CV_CAP,
	MAX_TPE_TRIALS_CAP,
	ENABLE_LLM_SYSTEM_MESSAGES,
)
from ai_agent.data_utils import (
	read_csv,
	build_summary,
	describe_numeric,
	describe_categorical,
	split_columns,
	count_iqr_outliers,
	remove_iqr_outliers,
	impute_missing_values,
)
from ai_agent.eda import render_correlation, render_basic_plots
from ai_agent.eda import (
	render_missingness_heatmap,
	render_target_distribution,
	render_target_relationships,
	render_pairplot,
	render_outlier_summary,
)
from ai_agent.llm_client import (
	build_prompt,
	call_groq,
	groq_token_loaded,
	LLM_MODEL,
	DEFAULT_ROUTER_MODEL,
	DEFAULT_REASONER_MODEL,
	DEFAULT_REVIEWER_MODEL,
	build_system_message_prompt,
	explain_groq_error,
	classify_task_route,
	should_use_reviewer,
)
from ai_agent.modeling import (
	apply_target_missing,
	build_preprocessor,
	estimate_feature_names,
	build_selector,
	train_baseline,
	get_param_grid,
	train_time_series,
	evaluate_predictions,
	evaluate_roc_auc,
	get_selected_feature_names,
)
from ai_agent.tuning import run_tuning, is_optuna_available
from ai_agent.results_view import render_results_view
from ai_agent.copilot_utils import (
	build_copilot_context,
	summarize_results_for_ai,
	update_preferences_from_text,
	format_preferences_for_context,
	is_requirement_message,
)
from ai_agent.rag_store import build_workflow_rag_context

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


def init_state() -> None:
	if "step" not in st.session_state:
		st.session_state["step"] = 0
	if "df" not in st.session_state:
		st.session_state["df"] = None
	if "df_original" not in st.session_state:
		st.session_state["df_original"] = None
	if "df_cleaned" not in st.session_state:
		st.session_state["df_cleaned"] = None
	if "summary" not in st.session_state:
		st.session_state["summary"] = None
	if "summary_cleaned" not in st.session_state:
		st.session_state["summary_cleaned"] = None
	if "llm_text" not in st.session_state:
		st.session_state["llm_text"] = None
	if "results" not in st.session_state:
		st.session_state["results"] = None
	if "results_tuned" not in st.session_state:
		st.session_state["results_tuned"] = None
	if "tuned_models_list" not in st.session_state:
		st.session_state["tuned_models_list"] = []
	if "model_reco" not in st.session_state:
		st.session_state["model_reco"] = None
	if "cleaning_applied" not in st.session_state:
		st.session_state["cleaning_applied"] = False
	if "cleaning_confirmed" not in st.session_state:
		st.session_state["cleaning_confirmed"] = False
	if "scroll_to_top" not in st.session_state:
		st.session_state["scroll_to_top"] = False
	if "agent_auto_mode" not in st.session_state:
		st.session_state["agent_auto_mode"] = False
	if "agent_chat_messages" not in st.session_state:
		st.session_state["agent_chat_messages"] = []
	if "agent_activity" not in st.session_state:
		st.session_state["agent_activity"] = []
	if "agent_goal" not in st.session_state:
		st.session_state["agent_goal"] = ""
	if "agent_preferences" not in st.session_state:
		st.session_state["agent_preferences"] = {}
	if "agent_preference_notes" not in st.session_state:
		st.session_state["agent_preference_notes"] = []
	if "agent_preference_last_update" not in st.session_state:
		st.session_state["agent_preference_last_update"] = None
	if "agent_step_autorun" not in st.session_state:
		st.session_state["agent_step_autorun"] = {}
	if "analysis_ready" not in st.session_state:
		st.session_state["analysis_ready"] = False
	if "analysis_ready_message" not in st.session_state:
		st.session_state["analysis_ready_message"] = ""
	if "agent_undo_stack" not in st.session_state:
		st.session_state["agent_undo_stack"] = []
	if "agent_request_undo" not in st.session_state:
		st.session_state["agent_request_undo"] = False
	if "agent_request_revert_all" not in st.session_state:
		st.session_state["agent_request_revert_all"] = False
	if "agent_request_revert_index" not in st.session_state:
		st.session_state["agent_request_revert_index"] = None


def clear_workflow_state(keep_original: bool = True) -> None:
	keys_to_clear = [
		"df",
		"df_cleaned",
		"summary",
		"summary_cleaned",
		"llm_text",
		"results",
		"results_tuned",
		"tuned_models_list",
		"model_reco",
		"cleaning_applied",
		"cleaning_confirmed",
		"step1_state",
		"step2_state",
		"step3_state",
		"step4_state",
		"analysis_ready",
		"analysis_ready_message",
		"agent_preferences",
		"agent_preference_notes",
		"agent_preference_last_update",
	]
	if not keep_original:
		keys_to_clear.append("df_original")
		keys_to_clear.append("df_cleaned")
	for key in keys_to_clear:
		if key in st.session_state:
			del st.session_state[key]
	for key in list(st.session_state.keys()):
		if key.startswith("feature_") or key.startswith("tuned_"):
			del st.session_state[key]
	st.session_state["step"] = 0


def add_agent_activity(message: str, level: str = "info") -> None:
	activity = st.session_state.get("agent_activity", [])
	now_ts = time.time()
	activity.insert(
		0,
		{
			"time": datetime.now().strftime("%H:%M:%S"),
			"ts": now_ts,
			"level": level,
			"message": message,
		},
	)
	st.session_state["agent_activity"] = activity[:20]


def render_activity_toasts() -> None:
	now_ts = time.time()
	recent = []
	activity = st.session_state.get("agent_activity", [])
	for item in activity:
		ts = item.get("ts")
		# Backward compatibility: older activity entries may not have timestamps.
		if not isinstance(ts, (int, float)):
			ts = now_ts
			item["ts"] = ts
		if isinstance(ts, (int, float)) and (now_ts - ts) <= 6:
			recent.append(item)
		if len(recent) >= 3:
			break
	st.session_state["agent_activity"] = activity[:20]
	if not recent:
		st.components.v1.html(
			"""
			<script>
			const root = window.parent.document.getElementById('agent-toast-root');
			if (root) {
				root.remove();
			}
			</script>
			""",
			height=0,
		)
		return

	payload = [
		{
			"level": str(item.get("level", "info")),
			"time": str(item.get("time", "--:--")),
			"message": str(item.get("message", "")),
		}
		for item in reversed(recent)
	]
	st.components.v1.html(
		f"""
		<script>
		const data = {json.dumps(payload)};
		const doc = window.parent.document;
		let root = doc.getElementById('agent-toast-root');
		if (!root) {{
			root = doc.createElement('div');
			root.id = 'agent-toast-root';
			root.className = 'ui-toast-root';
			doc.body.appendChild(root);
		}}
		root.innerHTML = '';
		for (let i = 0; i < data.length; i += 1) {{
			const item = data[i] || {{}};
			const toast = doc.createElement('div');
			const level = (item.level || 'info').toLowerCase();
			toast.className = `ui-toast ui-toast-${{level}}`;
			toast.style.animationDelay = `${{i * 0.08}}s`;

			const timeTag = doc.createElement('span');
			timeTag.className = 'ui-toast-time';
			timeTag.textContent = item.time || '--:--';
			toast.appendChild(timeTag);

			const content = doc.createElement('span');
			content.textContent = item.message || '';
			toast.appendChild(content);

			root.appendChild(toast);
		}}
		</script>
		""",
		height=0,
	)


def push_undo_snapshot(key: str, old_value: Any) -> None:
	stack = st.session_state.get("agent_undo_stack", [])
	# store a tuple (key, exists_flag, old_value)
	exists = key in st.session_state
	stack.append((key, exists, old_value))
	st.session_state["agent_undo_stack"] = stack[-100:]


def undo_last_agent_action() -> Optional[str]:
	stack = st.session_state.get("agent_undo_stack", [])
	if not stack:
		return "No undo history available."
	key, existed, old_value = stack.pop()
	st.session_state["agent_undo_stack"] = stack
	if existed:
		st.session_state[key] = old_value
	else:
		if key in st.session_state:
			del st.session_state[key]
	return f"Reverted {key} to previous state." 


def revert_agent_action_at(index: int) -> Optional[str]:
	"""Revert a specific undo stack entry by index (0-based). Index must be valid.
	The undo stack is a list of tuples (key, existed, old_value).
	"""
	stack = st.session_state.get("agent_undo_stack", [])
	if index is None or index < 0 or index >= len(stack):
		return "Invalid undo index."
	key, existed, old_value = stack.pop(index)
	st.session_state["agent_undo_stack"] = stack
	if existed:
		st.session_state[key] = old_value
	else:
		if key in st.session_state:
			del st.session_state[key]
	return f"Reverted {key} to previous state." 


def add_chat_message(role: str, content: str) -> None:
	messages = st.session_state.get("agent_chat_messages", [])
	messages.append({"role": role, "content": content})
	st.session_state["agent_chat_messages"] = messages[-20:]


def build_rule_based_step_hint(step: int, df: Optional[pd.DataFrame], summary: Any) -> str:
	if step == 0:
		return "Upload your dataset, then click Continue to start AI-guided EDA."
	if step == 1 and summary is not None:
		missing_total = int(df.isna().sum().sum()) if df is not None else 0
		return (
			f"Dataset has {summary.rows} rows and {summary.cols} columns. "
			f"Missing values total: {missing_total}. Focus on missingness and target relationships first."
		)
	if step == 2 and df is not None:
		missing_total = int(df.isna().sum().sum())
		if missing_total > 0:
			return "Missing values remain. Median/mode imputation is a safe default before modeling."
		return "No missing values detected. You can proceed to modeling directly."
	if step == 3:
		return "Choose target and confirm features. AI can auto-fill setup, but you control transitions with Continue."
	if step == 4:
		return "Review metrics, residuals, and feature importance before final decisions."
	return "AI assistant is ready for the current step."


def build_hybrid_step_hint(step: int, df: Optional[pd.DataFrame], summary: Any) -> str:
	rule_hint = build_rule_based_step_hint(step, df, summary)
	if not groq_token_loaded():
		return rule_hint
	context = ""
	if summary is not None:
		context = (
			f"Rows: {summary.rows}, Cols: {summary.cols}, "
			f"Numeric: {len(summary.numeric_cols)}, Categorical: {len(summary.categorical_cols)}"
		)
	prompt = (
		"You are an AI copilot in a data app. Use this rule hint and context to provide one short actionable line. "
		"Do not exceed 30 words.\n"
		f"Step: {step}\n"
		f"Rule hint: {rule_hint}\n"
		f"Context: {context}"
	)
	text = call_groq(prompt, max_new_tokens=60, task_type="reasoning")
	if text and not text.startswith("ERROR:"):
		return text
	return rule_hint


def render_agent_shell(step: int, df: Optional[pd.DataFrame], summary: Any) -> None:
	with st.sidebar.expander("AI Copilot", expanded=True):
		mode_on = st.toggle(
			"AI Auto Mode",
			value=st.session_state.get("agent_auto_mode", False),
			key="agent_auto_mode_toggle",
		)
		if mode_on != st.session_state.get("agent_auto_mode", False):
			st.session_state["agent_auto_mode"] = mode_on
			add_agent_activity(
				f"Auto mode {'enabled' if mode_on else 'disabled'} (step changes still require Continue)."
			)

		rule_or_hybrid_hint = build_hybrid_step_hint(step, df, summary)
		st.caption(f"Hint: {rule_or_hybrid_hint}")

		if st.session_state.get("agent_auto_mode", False):
			st.info("Suggestions only. AI will not advance steps; use Continue when ready.")
			last_applied = st.session_state.get("agent_auto_last_applied", [])
			if last_applied:
				st.caption(f"Applied suggestions: {', '.join(last_applied[:5])}")

		conversation_slot = st.container()

		st.markdown("<div class='ui-chat-input'>", unsafe_allow_html=True)
		with st.form("chat_dock_form", clear_on_submit=True):
			st.text_area(
				"Message",
				value=st.session_state.get("chat_dock_input", ""),
				key="chat_dock_input",
				height=90,
				label_visibility="collapsed",
				placeholder="Tell the AI exactly what you want it to do...",
			)
			sent = st.form_submit_button("Send")
		st.markdown("</div>", unsafe_allow_html=True)
		if sent:
			user_text = st.session_state.get("chat_dock_input", "").strip()
			if not user_text:
				st.warning("Please enter a message before sending.")
			else:
				add_chat_message("user", user_text)
				st.session_state["agent_goal"] = user_text
				updated_preferences, preference_updates = update_preferences_from_text(
					user_text,
					st.session_state.get("agent_preferences"),
				)
				st.session_state["agent_preferences"] = updated_preferences
				if preference_updates:
					st.session_state["agent_preference_last_update"] = datetime.now().strftime("%H:%M:%S")
					notes = st.session_state.get("agent_preference_notes", [])
					notes.extend(preference_updates)
					st.session_state["agent_preference_notes"] = notes[-20:]
					for message in preference_updates:
						add_agent_activity(message)
				if groq_token_loaded():
					# If the message is a requirement, record it and acknowledge — do not call the LLM.
					if is_requirement_message(user_text):
						reqs = st.session_state.get("agent_requirements", [])
						reqs.insert(0, {"text": user_text, "time": datetime.now().isoformat()})
						st.session_state["agent_requirements"] = reqs[:50]
						add_chat_message("assistant", "Noted. Requirement recorded.")
						add_agent_activity("Requirement recorded via chat.")
						# Keep the user's goal recorded but do not prompt the LLM.
						st.session_state["agent_goal"] = user_text
					else:
						with st.spinner("AI thinking..."):
							context = build_copilot_context(step, df, summary)
							# Include a clarifying-instruction only for non-requirement messages.
							prompt = (
								"You are the app's AI agent. Follow the user's request exactly when possible. "
								"Be concise, action-oriented, and do not ignore the user's goal. "
								"Respect stored user preferences when proposing model choices unless the data constraints conflict, then explain briefly.\n"
								f"Current workflow context:\n{context}\n\n"
								f"User request to follow:\n{user_text}"
							)
							route_model = classify_task_route(user_text, context)
							reply = call_groq(prompt, max_new_tokens=180, task_type="chat", context=context, force_model=route_model) or ""
						if reply.startswith("ERROR:"):
							add_agent_activity(reply, level="error")
							reply = (
								f"{explain_groq_error(reply)} "
								f"{build_rule_based_step_hint(step, df, summary)}"
							)
						elif not reply.strip():
							reply = build_rule_based_step_hint(step, df, summary)
						add_chat_message("assistant", reply)
						add_agent_activity("AI replied to chat.")
				else:
					add_chat_message("assistant", "LLM not configured. Set GROQ_API_KEY to enable responses.")
					add_agent_activity("LLM not configured; no reply.", level="warn")

			with conversation_slot:
				st.markdown("<div class='ui-copilot-chat-title'>Conversation</div>", unsafe_allow_html=True)
				st.markdown("<div class='ui-chat-messages'>", unsafe_allow_html=True)
				for msg in st.session_state.get("agent_chat_messages", [])[-8:]:
					is_user = msg.get("role") == "user"
					row_class = "user" if is_user else "ai"
					text = html.escape(str(msg.get("content", ""))).replace("\n", "<br>")
					st.markdown(
						f"<div class='ui-chat-bubble-row {row_class}'><div class='ui-chat-bubble {row_class}'>{text}</div></div>",
						unsafe_allow_html=True,
					)
				if not st.session_state.get("agent_chat_messages"):
					st.caption("No messages yet.")
				st.markdown("</div>", unsafe_allow_html=True)
				st.components.v1.html(
					"""
					<script>
					const root = window.parent.document;
					const messages = root.querySelector('.ui-chat-messages');
					if (messages) {
						messages.scrollTop = messages.scrollHeight;
					}
					</script>
					""",
					height=0,
				)


def run_step_auto_actions(step: int) -> None:
	if not st.session_state.get("agent_auto_mode", False):
		return
	autorun_flags = st.session_state.get("agent_step_autorun", {})
	if autorun_flags.get(step):
		return

	try:
		from ai_agent.auto_engine import apply_auto_actions_snapshot

		# Build minimal snapshot and call engine
		snapshot = {k: st.session_state.get(k) for k in list(st.session_state.keys())}
		df = st.session_state.get("df")
		changes, activities = apply_auto_actions_snapshot(snapshot, step, df)

		current_goal = str(st.session_state.get("agent_goal", "") or "").strip()
		if groq_token_loaded():
			column_text = " ".join(map(str, list(df.columns)[:20])) if df is not None else ""
			summary_text = ""
			summary = st.session_state.get("summary")
			if summary is not None:
				summary_text = (
					f"rows={summary.rows} cols={summary.cols} "
					f"numeric={len(summary.numeric_cols)} categorical={len(summary.categorical_cols)} "
					f"datetime={len(summary.datetime_cols)}"
				)
			kb_context = build_workflow_rag_context(
				step,
				user_goal=current_goal,
				column_text=column_text,
				summary_text=summary_text,
			)
			auto_prompt = build_auto_mode_prompt(step, current_goal, snapshot, kb_context)
			auto_reply = call_groq(
				auto_prompt,
				max_new_tokens=220,
				task_type="reasoning",
				context=kb_context,
				force_model=DEFAULT_REASONER_MODEL,
			)
			ai_changes, ai_activities = parse_auto_mode_plan(auto_reply)
			for key, new_val in ai_changes.items():
				if key in {"step", "cleaning_confirmed", "analysis_ready"}:
					continue
				changes[key] = new_val
			activities.extend(ai_activities)
			if kb_context:
				activities.append("Auto mode used the KB to align the suggestion with the fixed workflow.")

		# Enforce strict gating: do not allow auto actions to change control-flow keys
		CONTROL_KEYS = {"step", "cleaning_confirmed", "analysis_ready"}
		applied = []
		ignored = []
		# Apply model_family first, then model_name, then remaining keys to avoid UI mismatch.
		ordered_keys = []
		if "model_family" in changes:
			ordered_keys.append("model_family")
		if "model_name" in changes:
			ordered_keys.append("model_name")
		for k in list(changes.keys()):
			if k not in ordered_keys:
				ordered_keys.append(k)

		for key in ordered_keys:
			if key in CONTROL_KEYS:
				ignored.append(key)
				continue
			new_val = changes[key]
			old_val = st.session_state.get(key) if key in st.session_state else None
			push_undo_snapshot(key, old_val)
			st.session_state[key] = new_val
			applied.append(key)

		# Record what was applied so UI can surface it; ignored keys are logged
		st.session_state["agent_auto_last_applied"] = applied
		for act in activities:
			add_agent_activity(act)
		if ignored:
			add_agent_activity(
				f"Auto actions attempted to change locked control keys (ignored): {', '.join(ignored)}",
				level="warning",
			)
	except Exception as exc:
		add_agent_activity(f"Auto actions failed: {exc}", level="error")

	autorun_flags[step] = True
	st.session_state["agent_step_autorun"] = autorun_flags


def build_auto_mode_prompt(
	step: int,
	user_goal: str,
	snapshot: Dict[str, Any],
	kb_context: str,
) -> str:
	current_keys = sorted([str(key) for key in snapshot.keys() if not str(key).startswith("agent_")])
	base_focus = {
		"step": step,
		"goal": user_goal or "(none)",
		"visible_keys": current_keys[:40],
	}
	return (
		"You are the auto-mode decision engine for a fixed data-modelling workflow. "
		"The workflow cannot change. Prioritize the user's request unless it conflicts with the dataset, "
		"the fixed workflow, or the uploaded data constraints. If the request is vague, use the KB guidance. "
		"Use the KB as the secondary source and keep recommendations compatible with the current step. "
		"Return JSON only using this schema: {\"changes\": {\"key\": value}, \"activities\": [\"short note\"], \"priority_source\": \"user|kb|dataset\"}. "
		"Only suggest supported session keys and never include control-flow keys.\n"
		f"Context: {json.dumps(base_focus, ensure_ascii=True)}\n"
		f"Knowledge base guidance:\n{kb_context or '(no kb context available)'}"
	)


def parse_auto_mode_plan(reply_text: Optional[str]) -> Tuple[Dict[str, Any], List[str]]:
	if not reply_text:
		return {}, []
	text = str(reply_text).strip()
	if text.startswith("```"):
		text = text.strip("`")
		if text.lower().startswith("json"):
			text = text[4:].strip()
	start = text.find("{")
	end = text.rfind("}")
	if start < 0 or end < 0 or end <= start:
		return {}, []
	try:
		payload = json.loads(text[start : end + 1])
	except Exception:
		return {}, []
	changes = payload.get("changes") or {}
	activities = payload.get("activities") or []
	if not isinstance(changes, dict):
		changes = {}
	if not isinstance(activities, list):
		activities = []
	clean_changes: Dict[str, Any] = {}
	for key, value in changes.items():
		if str(key).startswith("agent_"):
			continue
		clean_changes[str(key)] = value
	clean_activities = [str(item) for item in activities if str(item).strip()]
	return clean_changes, clean_activities


def clear_tuned_state() -> None:
	st.session_state["results_tuned"] = None
	st.session_state["tuned_models_list"] = []
	for key in list(st.session_state.keys()):
		if key.startswith("tuned_"):
			del st.session_state[key]


def has_missing_values(df: pd.DataFrame) -> bool:
	return bool(df.isna().any().any())


def render_system_message(step_name: str, summary: Optional[Any]) -> None:
	if not ENABLE_LLM_SYSTEM_MESSAGES:
		return
	if not groq_token_loaded():
		return
	context = ""
	if summary is not None:
		context = (
			f"Rows: {summary.rows}, Columns: {summary.cols}, "
			f"Numeric: {len(summary.numeric_cols)}, "
			f"Categorical: {len(summary.categorical_cols)}"
		)
	prompt = build_system_message_prompt(step_name, context)
	with st.spinner("Generating guidance..."):
		text = call_groq(prompt, max_new_tokens=80, task_type="reasoning", context=context)
	if text:
		st.info(text)


def build_time_series_preview_series(df: pd.DataFrame, time_col: str, target_col: str) -> pd.Series:
	preview_df = df[[time_col, target_col]].copy()
	preview_df[time_col] = pd.to_datetime(preview_df[time_col], errors="coerce")
	preview_df = preview_df.dropna(subset=[time_col, target_col]).sort_values(time_col)
	return preview_df[target_col].astype(float)


def render_acf_pacf_plots(series: pd.Series) -> None:
	series = series.dropna()
	if len(series) < 4:
		st.info("Need at least 4 non-missing observations to render ACF and PACF.")
		return
	lags = min(20, max(1, len(series) // 2 - 1))
	if lags < 1:
		st.info("Not enough observations for ACF and PACF plots.")
		return
	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	plot_acf(series, ax=axes[0], lags=lags)
	axes[0].set_title("ACF")
	plot_pacf(series, ax=axes[1], lags=lags, method="ywm")
	axes[1].set_title("PACF")
	st.pyplot(fig, use_container_width=True)


def inject_ui_styles() -> None:
	st.markdown(
		"""
		<style>
		.ui-sticky-progress {
			position: sticky;
			top: 0;
			z-index: 1000;
			padding: 0.7rem 0 0.65rem 0;
			background: transparent;
			backdrop-filter: none;
			border-bottom: none;
			margin-bottom: 0.8rem;
		}
		.ui-progress-row {
			display: flex;
			justify-content: center;
		}
		.ui-progress-track {
			display: inline-flex;
			align-items: center;
			gap: 0.15rem;
			max-width: 100%;
			overflow-x: auto;
			padding: 0.4rem 0.65rem;
			scrollbar-width: thin;
			border-radius: 999px;
			background: rgba(16, 22, 40, 0.72);
			border: 1px solid rgba(165, 186, 255, 0.18);
			box-shadow: 0 10px 28px rgba(5, 7, 16, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.06);
		}
		.ui-progress-item {
			display: inline-flex;
			align-items: center;
			gap: 0.35rem;
			flex: 0 0 auto;
		}
		.ui-progress-step {
			display: inline-flex;
			align-items: center;
			gap: 0.35rem;
			padding: 0.36rem 0.62rem;
			border-radius: 999px;
			font-size: 0.84rem;
			background: rgba(255, 255, 255, 0.07);
			color: rgba(230, 236, 255, 0.86);
			border: 1px solid rgba(175, 193, 255, 0.14);
		}
		.ui-progress-dot {
			width: 1.2rem;
			height: 1.2rem;
			border-radius: 50%;
			display: inline-flex;
			align-items: center;
			justify-content: center;
			font-size: 0.72rem;
			font-weight: 700;
			background: rgba(212, 224, 255, 0.2);
			color: rgba(238, 244, 255, 0.96);
		}
		.ui-progress-step.completed {
			background: rgba(90, 190, 255, 0.16);
			color: #c6efff;
			border-color: rgba(137, 216, 255, 0.35);
			font-weight: 600;
		}
		.ui-progress-step.completed .ui-progress-dot {
			background: #5ac9ff;
			color: #032133;
		}
		.ui-progress-step.active {
			background: linear-gradient(120deg, #2f6ed8 0%, #2aa6ff 100%);
			color: #ffffff;
			border-color: rgba(168, 225, 255, 0.56);
			font-weight: 600;
			box-shadow: 0 0 0 1px rgba(191, 233, 255, 0.35), 0 8px 20px rgba(37, 134, 227, 0.38);
		}
		.ui-progress-step.active .ui-progress-dot {
			background: #ffffff;
			color: #2259b8;
		}
		.ui-progress-connector {
			width: 1.45rem;
			height: 2px;
			background: rgba(193, 210, 255, 0.22);
			border-radius: 999px;
			flex: 0 0 auto;
		}
		.ui-progress-connector.done {
			background: linear-gradient(90deg, #56bfff 0%, #2f6ed8 100%);
		}
		.ui-chat-shell {
			display: flex;
			flex-direction: column;
			height: 100%;
		}
		.ui-workflow-actions {
			max-width: 30rem;
			margin: 0.2rem auto 0.8rem auto;
		}
		.ui-workflow-actions .ui-workflow-label {
			font-size: 0.82rem;
			letter-spacing: 0.02em;
			opacity: 0.8;
			text-align: center;
			margin-bottom: 0.3rem;
		}
		.ui-workflow-actions .ui-workflow-grid {
			display: flex;
			justify-content: center;
			align-items: center;
			gap: 0.35rem;
			width: 100%;
		}
		.ui-workflow-actions .ui-workflow-item {
			flex: 0 0 8.25rem;
			max-width: 8.25rem;
		}
		.ui-workflow-actions .stButton > button {
			width: 100%;
			min-height: 2.85rem;
			padding: 0.44rem 0.65rem;
			border-radius: 999px;
			font-weight: 600;
			font-size: 0.86rem;
			letter-spacing: 0.01em;
			border: 1px solid rgba(175, 193, 255, 0.16);
			background: rgba(255, 255, 255, 0.06);
			color: rgba(240, 246, 255, 0.98);
			box-shadow: 0 8px 18px rgba(5, 8, 18, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.04);
		}
		.ui-workflow-actions .stButton > button:hover {
			border-color: rgba(168, 225, 255, 0.42);
			background: linear-gradient(140deg, rgba(47, 110, 216, 0.86), rgba(42, 166, 255, 0.9));
			box-shadow: 0 0 0 1px rgba(191, 233, 255, 0.2), 0 10px 22px rgba(37, 134, 227, 0.28);
		}
		.ui-workflow-actions .stButton > button:focus {
			outline: none;
		}
		.ui-workflow-actions .ui-action-icon {
			display: inline-flex;
			align-items: center;
			justify-content: center;
			width: 1.45rem;
			height: 1.45rem;
			border-radius: 50%;
			margin-right: 0.45rem;
			font-size: 0.74rem;
			font-weight: 700;
			background: rgba(212, 224, 255, 0.2);
			color: rgba(238, 244, 255, 0.96);
			box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
		}
		.ui-workflow-actions .stButton > button:hover .ui-action-icon {
			background: #ffffff;
			color: #2259b8;
		}
		.ui-chat-messages {
			flex: 1;
			overflow-y: auto;
			max-height: 38vh;
			padding-right: 0.15rem;
			display: flex;
			flex-direction: column;
			gap: 0.4rem;
		}
		.ui-chat-bubble-row {
			display: flex;
			width: 100%;
		}
		.ui-chat-bubble-row.user {
			justify-content: flex-end;
		}
		.ui-chat-bubble-row.ai {
			justify-content: flex-start;
		}
		.ui-chat-bubble {
			max-width: 88%;
			padding: 0.45rem 0.62rem;
			border-radius: 0.85rem;
			font-size: 0.84rem;
			line-height: 1.35;
			white-space: pre-wrap;
			word-break: break-word;
			border: 1px solid transparent;
		}
		.ui-chat-bubble.user {
			background: linear-gradient(160deg, #2a72e6 0%, #3f9cff 100%);
			color: #ffffff;
			border-color: rgba(169, 217, 255, 0.6);
			border-bottom-right-radius: 0.2rem;
		}
		.ui-chat-bubble.ai {
			background: rgba(224, 232, 255, 0.12);
			color: rgba(236, 242, 255, 0.95);
			border-color: rgba(183, 198, 255, 0.24);
			border-bottom-left-radius: 0.2rem;
		}
		.ui-workflow-actions .stButton > button {
			width: 100%;
			min-height: 2.45rem;
			padding: 0.38rem 0.65rem;
			border-radius: 0.55rem;
			font-weight: 600;
			font-size: 0.84rem;
			letter-spacing: 0.01em;
			border: 1px solid rgba(175, 193, 255, 0.18);
			background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
			color: rgba(240, 246, 255, 0.98);
			box-shadow: 0 10px 20px rgba(5, 8, 18, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.05);
		}
		.ui-workflow-actions .stButton > button:hover {
			border-color: rgba(168, 225, 255, 0.42);
			background: linear-gradient(140deg, rgba(47, 110, 216, 0.88), rgba(42, 166, 255, 0.92));
			box-shadow: 0 0 0 1px rgba(191, 233, 255, 0.2), 0 10px 22px rgba(37, 134, 227, 0.28);
		}
		.ui-workflow-actions .stButton > button:focus {
			outline: none;
		}
		.ui-toast {
			background:
				radial-gradient(circle at 14% 18%, rgba(118, 233, 255, 0.38) 0%, rgba(118, 233, 255, 0) 40%),
				linear-gradient(140deg, #041a3d 0%, #0a2b61 52%, #113f83 100%);
			color: #ffffff;
			border: 1px solid rgba(188, 248, 255, 0.96);
			border-radius: 0.82rem;
			padding: 0.62rem 0.84rem;
			font-size: 0.94rem;
			font-weight: 600;
			line-height: 1.35;
			box-shadow: 0 0 0 1px rgba(222, 252, 255, 0.28), 0 16px 34px rgba(3, 12, 29, 0.72);
			text-shadow: 0 1px 0 rgba(1, 6, 20, 0.55);
			animation: toastInOut 5.2s ease forwards;
		}
		.ui-toast-time {
			display: inline-block;
			font-size: 0.76rem;
			opacity: 0.98;
			font-weight: 700;
			color: #e8fbff;
			margin-right: 0.45rem;
		}
		.ui-toast-error {
			background:
				radial-gradient(circle at 14% 18%, rgba(255, 213, 184, 0.44) 0%, rgba(255, 213, 184, 0) 38%),
				linear-gradient(140deg, #632127 0%, #8e2d32 55%, #b93f34 100%);
			border-color: rgba(255, 199, 176, 0.9);
			color: #fff8f5;
		}
		.ui-toast-warn {
			background:
				radial-gradient(circle at 14% 18%, rgba(255, 243, 182, 0.48) 0%, rgba(255, 243, 182, 0) 38%),
				linear-gradient(140deg, #5f3900 0%, #8a5200 52%, #b86b00 100%);
			border-color: rgba(255, 230, 146, 0.92);
			color: #fffdf3;
		}
		@keyframes toastInOut {
			0% { opacity: 0; transform: translateY(10px) scale(0.98); }
			10% { opacity: 1; transform: translateY(0) scale(1); }
			82% { opacity: 1; transform: translateY(0) scale(1); }
			100% { opacity: 0; transform: translateY(-8px) scale(0.99); }
		}
		.ui-copilot-chat-title {
			font-size: 0.82rem;
			text-transform: uppercase;
			letter-spacing: 0.03em;
			opacity: 0.78;
			margin-top: 0.2rem;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def render_sidebar_branding() -> None:
	st.sidebar.markdown("### Automated Data Modelling")


def render_stepper(steps: List[str], current_step: int) -> None:
	step_html = []
	for idx, label in enumerate(steps):
		if idx < current_step:
			state_class = " completed"
			badge = "✓"
		elif idx == current_step:
			state_class = " active"
			badge = str(idx + 1)
		else:
			state_class = ""
			badge = str(idx + 1)
		step_html.append(
			"<span class='ui-progress-item'>"
			f"<span class='ui-progress-step{state_class}'>"
			f"<span class='ui-progress-dot'>{badge}</span>{label}</span>"
			"</span>"
		)
		if idx < len(steps) - 1:
			connector_class = " done" if idx < current_step else ""
			step_html.append(f"<span class='ui-progress-connector{connector_class}'></span>")
	st.markdown(
		"<div class='ui-sticky-progress'><div class='ui-progress-row'><div class='ui-progress-track'>"
		+ "".join(step_html)
		+ "</div></div></div>",
		unsafe_allow_html=True,
	)


def render_workflow_actions() -> None:
	st.markdown("<div class='ui-workflow-actions'><div class='ui-workflow-label'>Workflow Actions</div>", unsafe_allow_html=True)
	with st.container():
		col_left, col_mid, col_right = st.columns([1, 1, 1], gap="small")
		with col_left:
			st.markdown("<div class='ui-workflow-item'>", unsafe_allow_html=True)
			if st.button("↶ Undo", key="main_undo_last", help="Undo last change"):
				msg = undo_last_agent_action()
				add_agent_activity(msg or "Undo performed.")
			st.markdown("</div>", unsafe_allow_html=True)
		with col_mid:
			st.markdown("<div class='ui-workflow-item'>", unsafe_allow_html=True)
			if st.button("⟲ Revert", key="main_revert_all", help="Revert all changes"):
				count = 0
				while st.session_state.get("agent_undo_stack"):
					msg = undo_last_agent_action()
					add_agent_activity(msg or "Reverted one action.")
					count += 1
					if count > 500:
						add_agent_activity("Revert halted: too many operations.", level="error")
						break
				add_agent_activity(f"Reverted {count} AI-driven actions.")
			st.markdown("</div>", unsafe_allow_html=True)
		with col_right:
			st.markdown("<div class='ui-workflow-item'>", unsafe_allow_html=True)
			if st.button("↺ Reset", key="main_start_over", help="Start over"):
				clear_workflow_state(keep_original=False)
				st.session_state["step"] = 0
				st.session_state["scroll_to_top"] = True
				st.rerun()
			st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)


def scroll_to_top_if_needed() -> None:
	if st.session_state.get("scroll_to_top"):
		st.components.v1.html("<script>window.scrollTo(0, 0);</script>", height=0)
		st.session_state["scroll_to_top"] = False


def restore_widget_state(bucket_key: str, keys: List[str]) -> None:
	saved = st.session_state.get(bucket_key, {})
	for key in keys:
		if key in saved and key not in st.session_state:
			st.session_state[key] = saved[key]


def save_widget_state(bucket_key: str, keys: List[str]) -> None:
	saved = st.session_state.get(bucket_key, {})
	for key in keys:
		if key in st.session_state:
			saved[key] = st.session_state[key]
	st.session_state[bucket_key] = saved


def synced_slider_number(
	label: str,
	key: str,
	min_value: int,
	max_value: int,
	value: int,
) -> int:
	if min_value >= max_value:
		fixed_value = max(min_value, max_value)
		st.number_input(
			f"{label} value",
			min_value=fixed_value,
			max_value=fixed_value,
			value=fixed_value,
			key=f"{key}_number",
			disabled=True,
			label_visibility="collapsed",
		)
		st.session_state[f"{key}_value"] = fixed_value
		return fixed_value
	state_key = f"{key}_value"
	clamped_value = max(min_value, min(value, max_value))
	if state_key not in st.session_state:
		st.session_state[state_key] = clamped_value
	else:
		st.session_state[state_key] = max(
			min_value,
			min(st.session_state[state_key], max_value),
		)
	slider_key = f"{key}_slider"
	number_key = f"{key}_number"
	if slider_key in st.session_state:
		st.session_state[slider_key] = max(
			min_value,
			min(st.session_state[slider_key], max_value),
		)
	if number_key in st.session_state:
		st.session_state[number_key] = max(
			min_value,
			min(int(st.session_state[number_key]), max_value),
		)

	def on_slider_change() -> None:
		value = max(
			min_value,
			min(st.session_state[slider_key], max_value),
		)
		st.session_state[state_key] = value
		st.session_state[number_key] = value

	def on_number_change() -> None:
		value = max(
			min_value,
			min(int(st.session_state[number_key]), max_value),
		)
		st.session_state[state_key] = value
		st.session_state[slider_key] = value
	col_a, col_b = st.columns([3, 1])
	with col_a:
		slider_value = st.slider(
			label,
			min_value=min_value,
			max_value=max_value,
			value=st.session_state[state_key],
			key=slider_key,
			on_change=on_slider_change,
		)
	with col_b:
		number_value = st.number_input(
			f"{label} value",
			min_value=min_value,
			max_value=max_value,
			value=st.session_state[state_key],
			key=number_key,
			on_change=on_number_change,
			label_visibility="collapsed",
		)
	if slider_value != st.session_state[state_key]:
		st.session_state[state_key] = slider_value
	if number_value != st.session_state[state_key]:
		st.session_state[state_key] = int(number_value)
	return st.session_state[state_key]


def cap_value(value: int, min_value: int, max_value: int) -> int:
	return max(min_value, min(value, max_value))


def parse_manual_value(value: Any) -> Any:
	if value is None:
		return None
	if isinstance(value, (int, float, bool)):
		return value
	if isinstance(value, str):
		stripped = value.strip()
		if not stripped:
			return None
		try:
			return ast.literal_eval(stripped)
		except Exception:
			return stripped
	return value


def render_manual_tuning_controls(
	problem_type: str,
	model_name: str,
	prefix: str,
) -> Dict[str, Any]:
	params: Dict[str, Any] = {}
	param_defs: List[Dict[str, Any]] = []
	if model_name in ["Logistic Regression", "SVM", "SVR"]:
		param_defs.append(
			{"name": "model__C", "label": "C", "min": 0.01, "max": 10.0, "value": 1.0}
		)
	if model_name in ["Ridge", "Lasso"]:
		param_defs.append(
			{"name": "model__alpha", "label": "alpha", "min": 0.0001, "max": 10.0, "value": 1.0}
		)
	if model_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
		param_defs.append(
			{"name": "model__max_depth", "label": "max_depth", "min": 1, "max": 30, "value": 5}
		)
	if model_name in ["Decision Tree", "Random Forest"]:
		param_defs.append(
			{
				"name": "model__min_samples_split",
				"label": "min_samples_split",
				"min": 2,
				"max": 20,
				"value": 2,
			}
		)
	if model_name in ["Random Forest", "Gradient Boosting"]:
		param_defs.append(
			{
				"name": "model__n_estimators",
				"label": "n_estimators",
				"min": 50,
				"max": 500,
				"value": 200,
			}
		)
	if model_name == "Gradient Boosting":
		param_defs.append(
			{
				"name": "model__learning_rate",
				"label": "learning_rate",
				"min": 0.01,
				"max": 0.3,
				"value": 0.1,
			}
		)
	if model_name in ["KNN"]:
		param_defs.append(
			{
				"name": "model__n_neighbors",
				"label": "n_neighbors",
				"min": 3,
				"max": 30,
				"value": 5,
			}
		)
	if model_name in ["Logistic Regression"]:
		param_defs.append(
			{
				"name": "model__max_iter",
				"label": "max_iter",
				"min": 100,
				"max": 2000,
				"value": 1000,
			}
		)

	if model_name in ["SVM", "SVR"]:
		kernel = st.selectbox(
			"Kernel",
			["rbf", "linear"],
			key=f"{prefix}manual_kernel",
		)
		params["model__kernel"] = kernel

	for definition in param_defs:
		name = definition["name"]
		label = definition["label"]
		min_val = definition["min"]
		max_val = definition["max"]
		value = definition["value"]
		key = f"{prefix}manual_{label}"
		if isinstance(value, float):
			selected = st.slider(
				label,
				min_value=float(min_val),
				max_value=float(max_val),
				value=float(value),
				step=0.01,
				key=key,
			)
		else:
			selected = st.slider(
				label,
				min_value=int(min_val),
				max_value=int(max_val),
				value=int(value),
				step=1,
				key=key,
			)
		params[name] = selected

	override_df = st.data_editor(
		pd.DataFrame(
			[
				{"param": "", "value": ""},
			],
			columns=["param", "value"],
		),
		use_container_width=True,
		key=f"{prefix}manual_overrides",
	)
	for _, row in override_df.iterrows():
		param_name = str(row.get("param", "")).strip()
		param_value = parse_manual_value(row.get("value"))
		if param_name and param_value is not None:
			params[param_name] = param_value

	if not params and problem_type in ["classification", "regression"]:
		st.info("No manual parameters selected.")
	return params


def sample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
	if len(df) <= max_rows:
		return df
	return df.sample(n=max_rows, random_state=42)


def build_llm_summary(df: pd.DataFrame, summary: Any) -> str:
	overview_text = (
		f"Rows: {summary.rows}\n"
		f"Columns: {summary.cols}\n"
		f"Numeric columns: {len(summary.numeric_cols)}\n"
		f"Categorical columns: {len(summary.categorical_cols)}\n"
		f"Datetime columns: {len(summary.datetime_cols)}\n"
	)
	missing_text = summary.missing_by_col.to_string(index=False)
	num_desc = describe_numeric(df, summary.numeric_cols)
	num_text = num_desc.head(10).to_string() if not num_desc.empty else "No numeric stats"
	prompt = build_prompt(overview_text, missing_text, num_text)
	return call_groq(prompt, task_type="reasoning", context=overview_text) or ""


def detect_unacceptable_features(
	problem_type: str,
	feature_cols: List[str],
	dtype_map: Dict[str, str],
) -> List[str]:
	if problem_type == "time_series":
		return []
	flags = []
	for col in feature_cols:
		name = col.lower()
		dtype = dtype_map.get(col, "")
		if any(token in name for token in ["date", "datetime", "time", "year", "month", "day", "year_built", "built_year"]):
			flags.append(col)
			continue
		if "datetime" in dtype or "datetimetz" in dtype:
			flags.append(col)
	return sorted(set(flags))


def extract_model_insights(pipeline: Any, feature_names: List[str]) -> Dict[str, Any]:
	model = pipeline.named_steps.get("model") if pipeline else None
	if model is None:
		return {}
	selected_names = get_selected_feature_names(pipeline, feature_names)
	insights: Dict[str, Any] = {}
	if hasattr(model, "coef_"):
		coef = np.array(model.coef_)
		if coef.ndim > 1:
			coef = np.mean(np.abs(coef), axis=0)
		else:
			coef = np.abs(coef)
		pairs = sorted(zip(selected_names, coef.tolist()), key=lambda x: abs(x[1]), reverse=True)
		insights["coefficients"] = [
			{"feature": name, "value": float(value)} for name, value in pairs[:20]
		]
	if hasattr(model, "feature_importances_"):
		importances = model.feature_importances_
		pairs = sorted(
			zip(selected_names, importances.tolist()), key=lambda x: x[1], reverse=True
		)
		insights["feature_importances"] = [
			{"feature": name, "value": float(value)} for name, value in pairs[:20]
		]
	if hasattr(model, "support_"):
		insights["support_vectors"] = int(len(model.support_))
	if hasattr(model, "n_neighbors"):
		insights["n_neighbors"] = int(model.n_neighbors)
	return insights


def build_classification_artifacts(
	model: Any,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	preds: np.ndarray,
) -> Dict[str, Any]:
	artifacts: Dict[str, Any] = {}
	cm = confusion_matrix(y_test, preds)
	artifacts["confusion_matrix"] = cm.tolist()
	artifacts["y_true"] = pd.Series(y_test).tolist()
	artifacts["preds"] = pd.Series(preds).tolist()
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(X_test)
		if proba.ndim == 2 and proba.shape[1] == 2:
			classes = pd.Series(y_test).dropna().unique().tolist()
			if len(classes) == 2:
				label_map = {classes[0]: 0, classes[1]: 1}
				y_binary = pd.Series(y_test).map(label_map)
				fpr, tpr, _ = roc_curve(y_binary, proba[:, 1])
				artifacts["roc_curve"] = {
					"fpr": fpr.tolist(),
					"tpr": tpr.tolist(),
				}
				artifacts["proba_positive"] = proba[:, 1].tolist()
				artifacts["residuals"] = (y_binary - proba[:, 1]).tolist()
	return artifacts


def build_regression_artifacts(
	y_test: pd.Series,
	preds: np.ndarray,
) -> Dict[str, Any]:
	residuals = y_test - preds
	return {
		"y_true": y_test.tolist(),
		"preds": preds.tolist(),
		"residuals": residuals.tolist(),
	}


def run_modeling_flow(settings: Dict[str, Any]) -> Dict[str, Any]:
	df = settings["df"]
	problem_type = settings["problem_type"]
	target_col = settings["target_col"]
	feature_cols = settings["feature_cols"]
	model_name = settings["model_name"]
	feature_selection = settings["feature_selection"]
	stepwise_direction = settings["stepwise_direction"]
	model_based = settings["model_based"]
	max_features = settings["max_features"]
	tuning_method = settings["tuning_method"]
	metric_label = settings["metric_label"]
	metric_scoring = settings["metric_scoring"]
	cv_folds = settings["cv_folds"]
	tpe_trials = settings["tpe_trials"]
	manual_params = settings.get("manual_params", {})

	work_df, error = apply_target_missing(
		df,
		target_col=target_col,
		problem_type=problem_type,
		target_missing=settings["target_missing"],
	)
	if error:
		return {"error": error}

	work_df = sample_rows(work_df, settings["max_rows"])
	X = work_df[feature_cols].copy()
	y = work_df[target_col]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	numeric_cols, categorical_cols = split_columns(X_train)
	preprocessor = build_preprocessor(numeric_cols, categorical_cols)
	feature_names = estimate_feature_names(preprocessor, X_train)

	selector = build_selector(
		feature_selection,
		problem_type,
		model_based,
		stepwise_direction,
		max_features,
		feature_names,
	)

	baseline = train_baseline(
		X_train,
		X_test,
		y_train,
		y_test,
		problem_type,
		model_name,
		selector,
		preprocessor,
		feature_names,
		metric_label,
	)
	baseline_preds = baseline["pipeline"].predict(X_test)
	if problem_type == "classification":
		baseline_artifacts = build_classification_artifacts(
			baseline["pipeline"],
			X_test,
			y_test,
			baseline_preds,
		)
	else:
		baseline_artifacts = build_regression_artifacts(y_test, baseline_preds)
	baseline_insights = extract_model_insights(baseline["pipeline"], feature_names)

	result = {
		"baseline": baseline,
		"baseline_artifacts": baseline_artifacts,
		"baseline_insights": baseline_insights,
		"tuned": None,
		"problem_type": problem_type,
		"metric_label": metric_label,
		"metric_scoring": metric_scoring,
		"model_name": model_name,
		"feature_selection": feature_selection,
		"selected_features": baseline["selected_features"],
	}

	if tuning_method == "Manual":
		if not manual_params:
			result["tuned"] = {"error": "Manual tuning requires parameters."}
			return result
		best_model = clone(baseline["pipeline"])
		best_model.set_params(**manual_params)
		best_model.fit(X_train, y_train)
		preds = best_model.predict(X_test)
		metrics = evaluate_predictions(y_test, preds, problem_type)

		roc_auc = None
		if problem_type == "classification" and metric_label == "ROC-AUC (ovr)":
			roc_auc = evaluate_roc_auc(best_model, X_test, y_test)
			if roc_auc is not None:
				metrics["roc_auc_ovr"] = roc_auc

		selected = get_selected_feature_names(best_model, feature_names)
		if problem_type == "classification":
			tuned_artifacts = build_classification_artifacts(
				best_model,
				X_test,
				y_test,
				preds,
			)
		else:
			tuned_artifacts = build_regression_artifacts(y_test, preds)
		tuned_insights = extract_model_insights(best_model, feature_names)
		result["tuned"] = {
			"metrics": metrics,
			"best_params": manual_params,
			"selected_features": selected,
			"artifacts": tuned_artifacts,
			"insights": tuned_insights,
		}
	elif tuning_method != "None":
		param_grid = get_param_grid(problem_type, model_name)
		try:
			search = run_tuning(
				pipeline=baseline["pipeline"],
				search_type=tuning_method,
				param_grid=param_grid,
				scoring=metric_scoring,
				cv=cv_folds,
				n_trials=tpe_trials,
			)
		except RuntimeError as exc:
			result["tuned"] = {"error": str(exc)}
			return result
		if search is None:
			result["tuned"] = {"error": "Tuning method is not available."}
			return result

		search.fit(X_train, y_train)
		best_model = search.best_estimator_
		preds = best_model.predict(X_test)
		metrics = evaluate_predictions(y_test, preds, problem_type)

		roc_auc = None
		if problem_type == "classification" and metric_label == "ROC-AUC (ovr)":
			roc_auc = evaluate_roc_auc(best_model, X_test, y_test)
			if roc_auc is not None:
				metrics["roc_auc_ovr"] = roc_auc

		selected = get_selected_feature_names(best_model, feature_names)
		if problem_type == "classification":
			tuned_artifacts = build_classification_artifacts(
				best_model,
				X_test,
				y_test,
				preds,
			)
		else:
			tuned_artifacts = build_regression_artifacts(y_test, preds)
		tuned_insights = extract_model_insights(best_model, feature_names)
		result["tuned"] = {
			"metrics": metrics,
			"best_params": search.best_params_,
			"selected_features": selected,
			"artifacts": tuned_artifacts,
			"insights": tuned_insights,
		}

	return result


def main() -> None:
	st.set_page_config(page_title=APP_TITLE, layout="wide")

	init_state()
	inject_ui_styles()
	render_sidebar_branding()
	steps = ["Upload", "EDA + LLM", "Data Cleaning", "Modeling Setup", "Results"]
	render_stepper(steps, st.session_state["step"])
	render_workflow_actions()

	# Render persistent agent shell (chat + activity panel)
	try:
		render_agent_shell(st.session_state.get("step", 0), st.session_state.get("df"), st.session_state.get("summary"))
	except Exception as exc:
		# Fail gracefully if LLM or render fails
		add_agent_activity(f"Agent shell render failed: {exc}", level="error")

	# Handle per-entry revert request
	req_index = st.session_state.get("agent_request_revert_index")
	if req_index is not None:
		msg = revert_agent_action_at(req_index)
		add_agent_activity(msg or f"Reverted entry {req_index}.")
		st.session_state["agent_request_revert_index"] = None
	scroll_to_top_if_needed()
	render_activity_toasts()

	if st.session_state["step"] == 0:
		uploaded = st.file_uploader("Upload CSV", type=["csv"])
		if uploaded:
			df = read_csv(uploaded)
			st.session_state["df_original"] = df.copy()
			st.session_state["df"] = df.copy()
			st.session_state["summary"] = build_summary(df)
			st.session_state["df_cleaned"] = None
			st.session_state["summary_cleaned"] = None
			st.session_state["cleaning_applied"] = False
			st.session_state["cleaning_confirmed"] = False

		df = st.session_state.get("df")
		summary = st.session_state.get("summary")
		if df is None or summary is None:
			st.info("Upload a CSV to begin.")
			st.stop()

		if st.button("Restore original dataset"):
			original = st.session_state.get("df_original")
			if original is not None:
				clear_workflow_state(keep_original=True)
				st.session_state["df_original"] = original
				st.session_state["df"] = original.copy()
				st.session_state["summary"] = build_summary(original)
				st.session_state["step"] = 0
				st.session_state["scroll_to_top"] = True
				st.rerun()

		st.subheader("Preview")
		st.dataframe(df.head(20), use_container_width=True)
		st.subheader("Dataset Overview")
		st.write(f"Rows: {summary.rows} | Columns: {summary.cols}")
		st.dataframe(summary.dtypes, use_container_width=True)
		st.dataframe(summary.missing_by_col, use_container_width=True)

		col_a, col_b = st.columns([1, 1])
		with col_b:
			if st.button("Continue"):
				st.session_state["step"] = 1
				st.session_state["scroll_to_top"] = True
				st.rerun()

		# Try autorun actions for current step if enabled (non-blocking)
		run_step_auto_actions(st.session_state.get("step", 0))

	elif st.session_state["step"] == 1:
		restore_widget_state(
			"step1_state",
			[
				"eda_show_summary",
				"eda_show_missing",
				"eda_show_corr",
				"eda_show_basic",
				"eda_show_target_dist",
				"eda_show_target_rel",
				"eda_show_pairplot",
				"eda_show_outliers",
				"eda_outlier_remove_cols",
				"eda_target_col",
				"eda_numeric_column",
				"eda_target_feature",
				"llm_text",
			],
		)
		df = st.session_state.get("df")
		summary = st.session_state.get("summary")
		if df is None or summary is None:
			st.info("Upload a CSV to begin.")
			st.stop()
		run_step_auto_actions(1)

		st.subheader("EDA Selection")
		col_a, col_b = st.columns(2)
		with col_a:
			st.checkbox("Summary statistics", key="eda_show_summary")
			st.checkbox("Correlation heatmap", key="eda_show_corr")
			st.checkbox("Basic plots", key="eda_show_basic")
			st.checkbox("Missingness", key="eda_show_missing")
		with col_b:
			st.checkbox("Target distribution", key="eda_show_target_dist")
			st.checkbox("Target vs feature", key="eda_show_target_rel")
			st.checkbox("Pairplot (sampled)", key="eda_show_pairplot")
			st.checkbox("Outlier summary", key="eda_show_outliers")

		if st.session_state.get("eda_show_summary"):
			st.subheader("Summary Statistics")
			num_desc = describe_numeric(df, summary.numeric_cols)
			if not num_desc.empty:
				st.dataframe(num_desc, use_container_width=True)
			else:
				st.info("No numeric columns for numeric summary.")
			cat_desc = describe_categorical(df, summary.categorical_cols)
			if cat_desc:
				for col, series in cat_desc.items():
					st.write(f"Top categories for {col}")
					st.dataframe(series.reset_index(), use_container_width=True)

		if st.session_state.get("eda_show_corr"):
			st.subheader("Correlation Heatmap")
			render_correlation(df, summary.numeric_cols)

		if st.session_state.get("eda_show_basic"):
			st.subheader("Basic Plots")
			render_basic_plots(df, summary.numeric_cols)

		if st.session_state.get("eda_show_missing"):
			st.subheader("Missingness Overview")
			render_missingness_heatmap(df)

		show_target_sections = (
			st.session_state.get("eda_show_target_dist")
			or st.session_state.get("eda_show_target_rel")
		)
		if show_target_sections:
			target_options = ["(none)"] + list(df.columns)
			current_target = st.session_state.get("eda_target_col")
			if current_target not in target_options:
				st.session_state["eda_target_col"] = target_options[1] if len(target_options) > 1 else "(none)"
			target_eda = st.selectbox(
				"Target column (for EDA)",
				target_options,
				key="eda_target_col",
			)
			target_eda = None if target_eda == "(none)" else target_eda
		else:
			target_eda = None

		if st.session_state.get("eda_show_target_dist"):
			st.subheader("Target Distribution")
			render_target_distribution(df, target_eda)

		if st.session_state.get("eda_show_target_rel"):
			st.subheader("Target vs Feature")
			render_target_relationships(
				df,
				target_eda,
				summary.numeric_cols,
				summary.categorical_cols,
			)

		if st.session_state.get("eda_show_pairplot"):
			st.subheader("Pairplot (Sampled)")
			render_pairplot(df, summary.numeric_cols)

		if st.session_state.get("eda_show_outliers"):
			st.subheader("Outlier Summary")
			render_outlier_summary(df, summary.numeric_cols)
			outlier_counts = count_iqr_outliers(df, summary.numeric_cols)
			if not outlier_counts.empty:
				st.multiselect(
					"Remove outlier columns",
					outlier_counts["column"].tolist(),
					key="eda_outlier_remove_cols",
				)
				if st.button("Apply outlier removal"):
					selected_cols = st.session_state.get("eda_outlier_remove_cols", [])
					updated_df, _ = remove_iqr_outliers(df, selected_cols)
					st.session_state["df"] = updated_df
					st.session_state["summary"] = build_summary(updated_df)
					st.session_state["df_cleaned"] = None
					st.session_state["summary_cleaned"] = None
					st.session_state["cleaning_applied"] = False
					st.session_state["cleaning_confirmed"] = False
					st.rerun()

		save_widget_state(
			"step1_state",
			[
				"eda_show_summary",
				"eda_show_missing",
				"eda_show_corr",
				"eda_show_basic",
				"eda_show_target_dist",
				"eda_show_target_rel",
				"eda_show_pairplot",
				"eda_show_outliers",
					"eda_outlier_remove_cols",
				"eda_target_col",
				"eda_numeric_column",
				"eda_target_feature",
			],
		)

		col_a, col_b = st.columns([1, 1])
		with col_a:
			if st.button("Back"):
				st.session_state["step"] = 0
				st.session_state["scroll_to_top"] = True
				st.rerun()
		with col_b:
			if st.button("Continue"):
				if df is not None and has_missing_values(df):
					st.session_state["step"] = 2
				else:
					st.session_state["step"] = 3
				st.session_state["scroll_to_top"] = True
				st.rerun()

	elif st.session_state["step"] == 2:
		restore_widget_state(
			"step2_state",
			[
				"missing_strategy",
				"numeric_impute_strategy",
				"impute_categorical",
				"cleaning_applied",
				"cleaning_confirmed",
			],
		)
		df_original = st.session_state.get("df_original")
		df = st.session_state.get("df")
		summary = st.session_state.get("summary")
		if df_original is None or df is None or summary is None:
			st.info("Upload a CSV to begin.")
			st.stop()
		run_step_auto_actions(2)

		st.subheader("Data Cleaning")
		st.caption("Handle missing values only. Outliers are managed in EDA.")
		render_system_message("Data Cleaning", summary)

		st.subheader("Missing Value Imputation")
		missing_table = df_original.isna().sum().reset_index()
		missing_table.columns = ["column", "missing_count"]
		st.dataframe(missing_table, use_container_width=True)
		missing_strategy = st.radio(
			"Missing value action",
			["drop", "impute"],
			horizontal=True,
			key="missing_strategy",
		)
		numeric_strategy = "median"
		impute_categorical = True
		if missing_strategy == "impute":
			numeric_strategy = st.selectbox(
				"Impute numeric columns",
				["mean", "median"],
				key="numeric_impute_strategy",
			)
			impute_categorical = st.checkbox(
				"Impute categorical columns with mode",
				value=True,
				key="impute_categorical",
			)

		if st.button("Apply cleaning"):
			work_df = df_original.copy()
			if missing_strategy == "drop":
				work_df = work_df.dropna()
			else:
				work_df = impute_missing_values(
					work_df,
					numeric_strategy=numeric_strategy,
					impute_categorical=impute_categorical,
				)
			st.session_state["df_cleaned"] = work_df
			st.session_state["summary_cleaned"] = build_summary(work_df)
			st.session_state["cleaning_applied"] = True
			st.session_state["cleaning_confirmed"] = False
			st.success("Cleaning applied.")

		cleaned_df = st.session_state.get("df_cleaned")
		cleaned_summary = st.session_state.get("summary_cleaned")
		if cleaned_df is not None and cleaned_summary is not None:
			st.subheader("Before vs After")
			col_left, col_right = st.columns(2)
			with col_left:
				st.caption("Before")
				st.write(f"Rows: {summary.rows} | Columns: {summary.cols}")
				st.dataframe(summary.missing_by_col, use_container_width=True)
				st.dataframe(df_original.head(10), use_container_width=True)
			with col_right:
				st.caption("After")
				st.write(f"Rows: {cleaned_summary.rows} | Columns: {cleaned_summary.cols}")
				st.dataframe(cleaned_summary.missing_by_col, use_container_width=True)
				st.dataframe(cleaned_df.head(10), use_container_width=True)

		confirm_clean = st.checkbox(
			"Confirm cleaned data and proceed to modeling",
			key="cleaning_confirmed_display",
		)
		if st.button("Proceed to modeling"):
			candidate_df = cleaned_df if cleaned_df is not None else df_original
			candidate_summary = cleaned_summary if cleaned_summary is not None else summary
			if candidate_df.isna().any().any():
				st.error("Resolve missing values before proceeding to modeling.")
			else:
				st.session_state["df"] = candidate_df
				st.session_state["summary"] = candidate_summary
				st.session_state["cleaning_confirmed"] = True
				st.session_state["step"] = 3
				st.session_state["scroll_to_top"] = True
				st.rerun()

		col_a, col_b = st.columns([1, 1])
		with col_a:
			if st.button("Back"):
				st.session_state["step"] = 1
				st.session_state["scroll_to_top"] = True
				st.rerun()

		save_widget_state(
			"step2_state",
			[
				"missing_strategy",
				"numeric_impute_strategy",
				"impute_categorical",
				"cleaning_applied",
				"cleaning_confirmed",
				"cleaning_confirmed_display",
			],
		)
		st.stop()

	elif st.session_state["step"] == 3:
		step3_keys = [
			"target_col",
			"target_missing",
			"problem_type",
			"use_modeling",
			"model_reco",
			"time_col",
			"time_series_model",
			"time_series_mode",
			"arima_p",
			"arima_d",
			"arima_q",
			"sarima_P",
			"sarima_D",
			"sarima_Q",
			"sarima_m",
			"hw_trend",
			"hw_seasonal_periods",
			"model_family",
			"model_name",
			"metric_label",
			"proceed_model",
			"max_rows_value",
			"max_rows_slider",
			"max_rows_number",
			"selected_features",
			"confirm_features",
			"feature_warning",
			"feature_warning_signature",
			"feature_selection",
			"stepwise_direction",
			"model_based",
			"max_features_value",
			"max_features_slider",
			"max_features_number",
			"tuning_method",
			"cv_folds_value",
			"cv_folds_slider",
			"cv_folds_number",
			"tpe_trials_value",
			"tpe_trials_slider",
			"tpe_trials_number",
			"manual_params",
			"show_predictions",
		]
		restore_widget_state("step3_state", step3_keys)
		if "requirement_text" in st.session_state:
			del st.session_state["requirement_text"]
		step3_state_bucket = st.session_state.get("step3_state")
		if isinstance(step3_state_bucket, dict) and "requirement_text" in step3_state_bucket:
			step3_state_bucket.pop("requirement_text", None)
			st.session_state["step3_state"] = step3_state_bucket
		df = st.session_state.get("df")
		summary = st.session_state.get("summary")
		if df is None or summary is None:
			st.info("Upload a CSV to begin.")
			st.stop()
		if df.isna().any().any() and not st.session_state.get("cleaning_confirmed"):
			st.info("Complete data cleaning before modeling.")
			st.stop()
		run_step_auto_actions(3)
		render_system_message("Modeling Setup", summary)

		st.subheader("Problem Setup")
		target_col = st.selectbox("Target column", df.columns, key="target_col")
		target_missing = st.selectbox(
			"Target missing values",
			["drop", "error (stop if missing)"],
			key="target_missing",
		)
		missing_mode = "drop" if target_missing == "drop" else "error"

		problem_options = ["classification", "regression"]
		if summary.datetime_cols:
			problem_options.append("time_series")
		problem_type = st.selectbox("Problem type", problem_options, key="problem_type")

		preferred_model_family = st.session_state.get("agent_preferences", {}).get("preferred_model_family")
		if preferred_model_family in ["Statistical", "ML"] and "model_family" not in st.session_state:
			st.session_state["model_family"] = preferred_model_family

		use_modeling = st.radio(
			"Do you want to build a model?",
			["Yes", "No (EDA only)"],
			horizontal=True,
			key="use_modeling",
		)
		if use_modeling != "Yes":
			st.info(
				"You selected EDA-only. Review the summary and plots above, then return "
				"here when you are ready to build a model."
			)
			save_widget_state("step3_state", step3_keys)
			st.stop()

		if problem_type == "time_series":
			st.info("Datetime columns detected. Time series analysis is available.")
			time_col = st.selectbox("Time column", summary.datetime_cols, key="time_col")
			model_name = st.selectbox(
				"Model",
				["ARIMA", "SARIMA", "Holt-Winters"],
				key="time_series_model",
			)

			params: Dict[str, Any] = {}
			if model_name in ["ARIMA", "SARIMA"]:
				time_series_mode = st.radio(
					"ARIMA approach",
					["Auto ARIMA", "Manual"],
					horizontal=True,
					key="time_series_mode",
				)
				if model_name == "SARIMA":
					seasonal_periods = st.number_input(
						"Seasonal period (m)",
						min_value=2,
						max_value=24,
						value=12,
						key="sarima_m",
					)
					params["seasonal_periods"] = int(seasonal_periods)

				if time_series_mode == "Manual":
					preview_series = build_time_series_preview_series(df, time_col, target_col)
					st.subheader("ACF and PACF")
					render_acf_pacf_plots(preview_series)
					col_a, col_b, col_c = st.columns(3)
					with col_a:
						p = st.number_input("p", min_value=0, max_value=5, value=1, key="arima_p")
					with col_b:
						d = st.number_input("d", min_value=0, max_value=2, value=1, key="arima_d")
					with col_c:
						q = st.number_input("q", min_value=0, max_value=5, value=1, key="arima_q")
					params["order"] = (int(p), int(d), int(q))
					params["mode"] = "Manual"
					if model_name == "SARIMA":
						col_a, col_b, col_c = st.columns(3)
						with col_a:
							P = st.number_input("P", min_value=0, max_value=3, value=1, key="sarima_P")
						with col_b:
							D = st.number_input("D", min_value=0, max_value=2, value=1, key="sarima_D")
						with col_c:
							Q = st.number_input("Q", min_value=0, max_value=3, value=1, key="sarima_Q")
						params["seasonal_order"] = (int(P), int(D), int(Q), int(params["seasonal_periods"]))
				else:
					st.info("Auto ARIMA will search for the best parameters using AIC.")
					params["mode"] = "Auto ARIMA"
			else:
				col_a, col_b = st.columns(2)
				with col_a:
					trend = st.selectbox("Trend", ["add", "mul", None], index=0, key="hw_trend")
				with col_b:
					seasonal_periods = st.number_input(
						"Seasonal periods",
						min_value=2,
						max_value=24,
						value=12,
						key="hw_seasonal_periods",
					)
				params["trend"] = trend
				params["seasonal"] = "mul"
				params["seasonal_periods"] = int(seasonal_periods)

			if st.button("Run time series analysis"):
				with st.spinner("Running time series analysis..."):
					results = train_time_series(
						df,
						time_col,
						target_col,
						model_name,
						missing_mode,
						params=params,
					)
				if results.get("error"):
					st.error(results["error"])
				else:
					st.session_state["results"] = {
						"problem_type": "time_series",
						"model_name": model_name,
						"time_series_mode": params.get("mode", "Manual"),
						"time_series": results
					}
					st.session_state["step"] = 4
					st.session_state["scroll_to_top"] = True
					st.rerun()

			save_widget_state("step3_state", step3_keys)
			st.stop()

		# Apply explicit requested model/family from requirement text before rendering controls.
		req_texts = []
		if st.session_state.get("agent_goal"):
			req_texts.append(str(st.session_state.get("agent_goal")))
		for req in st.session_state.get("agent_requirements", []) or []:
			if isinstance(req, dict):
				req_texts.append(str(req.get("text", "")))
			else:
				req_texts.append(str(req))

		def _extract_requested_model(text: str):
			req_blob = str(text or "").lower().replace("-", " ").replace("_", " ")
			req_compact = req_blob.replace(" ", "")
			tokens = req_blob.split()
			if "decision tree" in req_blob or "decisiontree" in req_compact or "dtree" in req_compact or ((any(tok.startswith("deci") or tok in {"decison", "descision", "dcsn"} for tok in tokens)) and ("tree" in tokens)):
				return "Decision Tree", "ML"
			if "random forest" in req_blob or "randomforest" in req_compact or "rf" in tokens or ((any(tok.startswith("rand") or tok in {"ranom", "rendom"} for tok in tokens)) and ("forest" in tokens)):
				return "Random Forest", "ML"
			if "gradient boosting" in req_blob or "gradientboosting" in req_compact or "xgboost" in req_compact:
				return "Gradient Boosting", "ML"
			if "knn" in req_blob or "k nearest" in req_blob or "nearest neighbor" in req_blob:
				return "KNN", "ML"
			if " svm" in f" {req_blob}" or "support vector" in req_blob:
				return "SVM", "ML"
			if "svr" in req_blob:
				return "SVR", "ML"
			if "logistic" in req_blob:
				return "Logistic Regression", "Statistical"
			if "linear regression" in req_blob or "linearregression" in req_compact:
				return "Linear Regression", "Statistical"
			if "ridge" in req_blob:
				return "Ridge", "Statistical"
			if "lasso" in req_blob:
				return "Lasso", "Statistical"
			return None, None

		requested_model = None
		requested_family = None
		for text in req_texts:
			requested_model, requested_family = _extract_requested_model(text)
			if requested_model:
				break

		if requested_model:
			st.session_state["model_family"] = requested_family
			st.session_state["model_name"] = requested_model

		# Guard against invalid persisted family values (for example "decision_tree")
		# that can break Streamlit selectbox serialization.
		current_family = st.session_state.get("model_family")
		valid_families = ["Statistical", "ML"]
		if current_family and current_family not in valid_families:
			family_text = str(current_family).strip().lower().replace("-", "_").replace(" ", "_")
			if family_text in {"ml", "machine_learning", "machinelearning"}:
				st.session_state["model_family"] = "ML"
			elif family_text in {"statistical", "stats", "statistics"}:
				st.session_state["model_family"] = "Statistical"
			else:
				# Infer safest family from current model name if possible.
				current_model = str(st.session_state.get("model_name") or "").strip()
				if current_model in (ML_CLASSIFICATION_MODELS + ML_REGRESSION_MODELS):
					st.session_state["model_family"] = "ML"
				elif current_model in (STATISTICAL_CLASSIFICATION_MODELS + STATISTICAL_REGRESSION_MODELS):
					st.session_state["model_family"] = "Statistical"
				else:
					st.session_state["model_family"] = "ML"
			add_agent_activity(
				f"Invalid model family '{current_family}' normalized to '{st.session_state.get('model_family')}'.",
				level="warn",
			)

		model_family = st.selectbox(
			"Model family",
			["Statistical", "ML"],
			key="model_family",
		)
		preferences_context = format_preferences_for_context(st.session_state.get("agent_preferences"))
		if preferences_context:
			st.caption(f"Preference memory: {preferences_context}")
		if problem_type == "classification":
			model_options = (
				STATISTICAL_CLASSIFICATION_MODELS
				if model_family == "Statistical"
				else ML_CLASSIFICATION_MODELS
			)
			# Ensure stored model_name is present in the current options; otherwise pick a safe default.
			current_model = st.session_state.get("model_name")
			if current_model and current_model not in model_options:
				st.session_state["agent_activity"] = st.session_state.get("agent_activity", [])[:50]
				st.session_state["agent_activity"].insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "level": "warn", "message": f"Requested model '{current_model}' not valid for family '{model_family}'; defaulting to '{model_options[0]}'"})
				st.session_state["model_name"] = model_options[0]
			model_name = st.selectbox("Model", model_options, key="model_name")
			metric_label = st.selectbox(
				"Metric to optimize",
				list(CLASSIFICATION_METRICS.keys()),
				key="metric_label",
			)
			metric_scoring = CLASSIFICATION_METRICS[metric_label]
		else:
			model_options = (
				STATISTICAL_REGRESSION_MODELS
				if model_family == "Statistical"
				else ML_REGRESSION_MODELS
			)
			current_model = st.session_state.get("model_name")
			if current_model and current_model not in model_options:
				st.session_state["agent_activity"] = st.session_state.get("agent_activity", [])[:50]
				st.session_state["agent_activity"].insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "level": "warn", "message": f"Requested model '{current_model}' not valid for family '{model_family}'; defaulting to '{model_options[0]}'"})
				st.session_state["model_name"] = model_options[0]
			model_name = st.selectbox("Model", model_options, key="model_name")
			metric_label = st.selectbox(
				"Metric to optimize",
				list(REGRESSION_METRICS.keys()),
				key="metric_label",
			)
			metric_scoring = REGRESSION_METRICS[metric_label]

		proceed = st.checkbox("Proceed with this model", key="proceed_model")
		if not proceed:
			st.info("Pick a model, then confirm to see advanced options.")
			save_widget_state("step3_state", step3_keys)
			st.stop()

		st.subheader("Data Limits")
		min_rows = min(1000, len(df))
		max_rows_default = cap_value(DEFAULT_MAX_ROWS, min_rows, MAX_ROWS_CAP)
		max_rows_cap = min(MAX_ROWS_CAP, len(df))
		max_rows = synced_slider_number(
			"Max rows to use",
			key="max_rows",
			min_value=min_rows,
			max_value=max_rows_cap,
			value=min(max_rows_default, max_rows_cap),
		)

		feature_options = [c for c in df.columns if c != target_col]
		if len(feature_options) > MAX_FEATURES_CAP:
			feature_options = feature_options[:MAX_FEATURES_CAP]
			st.info("Too many columns detected; showing the first columns only.")

		feature_cols = [
			col for col in feature_options if st.session_state.get(f"feature_{col}", True)
		]
		feature_signature = f"{problem_type}:{','.join(feature_cols)}"
		if st.session_state.get("feature_warning_signature") != feature_signature:
			st.session_state["feature_warning_signature"] = feature_signature
			st.session_state["feature_warning"] = None
		dtype_map = df[feature_cols].dtypes.astype(str).to_dict() if feature_cols else {}
		deterministic_flags = detect_unacceptable_features(problem_type, feature_cols, dtype_map)
		if groq_token_loaded() and not deterministic_flags and not st.session_state.get("feature_warning"):
			feature_text = ", ".join([f"{col} ({dtype_map[col]})" for col in feature_cols])
			prompt = (
				"You are a data scientist. Provide a single-sentence warning about any "
				"features that should not be used for this modeling task. "
				"Explicitly flag date/time-derived fields or partial dates (year, month, day, "
				"year_built, built_year, yyyy, mm, dd) for regression/classification unless "
				"the user is doing time-series analysis. "
				"If all features are usable, say 'No unacceptable feature detected'. "
				f"Problem type: {problem_type}. "
				f"Features: {feature_text}."
			)
			with st.spinner("Checking feature suitability..."):
				selected_model = DEFAULT_REVIEWER_MODEL if should_use_reviewer(prompt) else DEFAULT_REASONER_MODEL
				st.session_state["feature_warning"] = call_groq(prompt, task_type="review", force_model=selected_model)
		if deterministic_flags:
			st.warning(
				"Unacceptable features detected (likely date/time-derived): "
				+ ", ".join(deterministic_flags)
			)
		elif st.session_state.get("feature_warning"):
			st.warning(st.session_state["feature_warning"])
		elif groq_token_loaded():
			st.warning("No unacceptable feature detected")
		else:
			st.info("Set GROQ_API_KEY to enable feature suitability warnings.")

		st.subheader("Feature Columns")
		stored_features = st.session_state.get("selected_features")
		if stored_features:
			stored_set = set(stored_features)
		else:
			stored_set = set(feature_options)
		with st.expander("Select features", expanded=True):
			for col in feature_options:
				key = f"feature_{col}"
				if key not in st.session_state:
					st.session_state[key] = col in stored_set
				st.checkbox(col, key=key)

		feature_cols = [
			col for col in feature_options if st.session_state.get(f"feature_{col}", False)
		]
		st.session_state["selected_features"] = feature_cols
		if not feature_cols:
			st.warning("Select at least one feature column.")
			auto_max_features = 1
		else:
			numeric_cols, categorical_cols = split_columns(df[feature_cols])
			preprocessor = build_preprocessor(numeric_cols, categorical_cols)
			feature_names = estimate_feature_names(preprocessor, df[feature_cols])
			auto_max_features = min(MAX_FEATURES_CAP, len(feature_names))

		confirm_features = st.checkbox(
			"Confirm feature selection",
			key="confirm_features",
		)
		if not confirm_features:
			st.info("Confirm feature selection to choose max features.")
			save_widget_state("step3_state", step3_keys)
			st.stop()
		else:
			st.subheader("Feature Selection")
			feature_methods = (
				["None", "Stepwise"]
				if model_family == "Statistical"
				else ["None", "RFE", "SelectKBest", "Model-based"]
			)
			current_feature_selection = st.session_state.get("feature_selection")
			if current_feature_selection not in feature_methods:
				st.session_state["feature_selection"] = feature_methods[0]
			feature_selection = st.selectbox("Method", feature_methods, key="feature_selection")
			stepwise_direction = "forward"
			model_based = "L1/Lasso"

			if feature_selection == "Stepwise":
				stepwise_direction = st.selectbox(
					"Stepwise direction",
					STEPWISE_DIRECTIONS,
					key="stepwise_direction",
				)
				if problem_type == "classification" and model_name != "Logistic Regression":
					st.warning("Stepwise selection is only supported for logistic regression.")
					feature_selection = "None"
				elif problem_type == "regression" and model_name != "Linear Regression":
					st.warning("Stepwise selection is only supported for linear regression.")
					feature_selection = "None"
			if feature_selection == "Model-based":
				model_based = st.selectbox(
					"Model-based selector",
					MODEL_BASED_OPTIONS,
					key="model_based",
				)

			max_allowed = auto_max_features
			if feature_selection != "None":
				max_allowed = max(1, auto_max_features - 1)
				if max_allowed < auto_max_features:
					st.info("Feature selection requires fewer than total features.")

			st.caption(f"Detected max features after preprocessing: {auto_max_features}")
			max_features = synced_slider_number(
				"Max selected features",
				key="max_features",
				min_value=1,
				max_value=max(1, max_allowed),
				value=max(1, max_allowed),
			)

			current_tuning = st.session_state.get("tuning_method")
			if model_family != "Statistical":
				tuning_options = TUNING_METHODS
				if "TPE (Optuna)" in tuning_options and not is_optuna_available():
					tuning_options = [opt for opt in TUNING_METHODS if opt != "TPE (Optuna)"]
				if current_tuning not in tuning_options:
					st.session_state["tuning_method"] = "Grid Search" if "Grid Search" in tuning_options else tuning_options[0]

		if model_family == "Statistical":
			st.subheader("Hyperparameter Tuning")
			st.info("Hyperparameter tuning is disabled for statistical models.")
			st.session_state["tuning_method"] = "None"
			tuning_method = "None"
			cv_folds = cap_value(DEFAULT_CV_FOLDS, 2, MAX_CV_CAP)
			tpe_trials = DEFAULT_TPE_TRIALS
		else:
			st.subheader("Hyperparameter Tuning")
			tuning_options = TUNING_METHODS
			if "TPE (Optuna)" in tuning_options and not is_optuna_available():
				tuning_options = [opt for opt in TUNING_METHODS if opt != "TPE (Optuna)"]
				st.info("TPE tuning requires Optuna. Install with: pip install optuna")

			tuning_method = st.selectbox("Tuning method", tuning_options, key="tuning_method")
			cv_folds = synced_slider_number(
				"Cross-validation folds",
				key="cv_folds",
				min_value=2,
				max_value=MAX_CV_CAP,
				value=cap_value(DEFAULT_CV_FOLDS, 2, MAX_CV_CAP),
			)
			tpe_trials = DEFAULT_TPE_TRIALS
			if tuning_method == "TPE (Optuna)":
				tpe_trials = synced_slider_number(
					"TPE trials",
					key="tpe_trials",
					min_value=10,
					max_value=MAX_TPE_TRIALS_CAP,
					value=cap_value(DEFAULT_TPE_TRIALS, 10, MAX_TPE_TRIALS_CAP),
				)

		manual_params: Dict[str, Any] = {}
		if tuning_method == "Manual":
			manual_params = render_manual_tuning_controls(problem_type, model_name, prefix="")

		if st.button("Run analysis"):
			if not feature_cols:
				st.error("Pick at least one feature column to continue.")
				st.stop()
			with st.spinner("Running analysis..."):
				results = run_modeling_flow(
					{
						"df": df,
						"problem_type": problem_type,
						"target_col": target_col,
						"feature_cols": feature_cols,
						"model_name": model_name,
						"feature_selection": feature_selection,
						"stepwise_direction": stepwise_direction,
						"model_based": model_based,
						"max_features": max_features,
						"tuning_method": tuning_method,
						"metric_label": metric_label,
						"metric_scoring": metric_scoring,
						"cv_folds": cv_folds,
						"tpe_trials": tpe_trials,
						"manual_params": manual_params,
						"max_rows": max_rows,
						"target_missing": missing_mode,
					}
				)
				if results.get("error"):
					st.error(results["error"])
				else:
					st.session_state["agent_last_outcome"] = summarize_results_for_ai(results)
					st.session_state["results"] = results
					st.session_state["step"] = 4
					st.session_state["scroll_to_top"] = True
					st.rerun()

		col_a, col_b = st.columns([1, 1])
		with col_a:
			if st.button("Back"):
				st.session_state["step"] = 3
				st.session_state["scroll_to_top"] = True
				st.rerun()

			save_widget_state("step3_state", step3_keys)

	elif st.session_state["step"] == 4:
		render_results_view(
			results=st.session_state.get("results"),
			results_tuned=st.session_state.get("results_tuned"),
			summary=st.session_state.get("summary"),
			tuned_models_list=st.session_state.get("tuned_models_list", []),
			clear_tuned_state=clear_tuned_state,
			render_system_message=render_system_message,
		)
		st.stop()
		results = st.session_state.get("results")
		results_tuned = st.session_state.get("results_tuned")
		if not results:
			st.info("Run analysis to see results.")
			st.stop()
		render_system_message("Results", st.session_state.get("summary"))

		show_predictions = st.checkbox("Show prediction vs actual", key="show_predictions", value=True)
		problem_type = results.get("problem_type")
		if problem_type == "time_series":
			ts = results.get("time_series", {})
			st.subheader("Time Series Results")
			st.write(f"Model: {results.get('model_name')}")
			if ts.get("model_details"):
				st.subheader("Model Details")
				st.json(ts["model_details"])
			if ts.get("metrics"):
				st.subheader("Metrics")
				st.json(ts["metrics"])
			if show_predictions and ts.get("y_true") and ts.get("preds"):
				st.subheader("Actual vs Predicted (Test)")
				fig, ax = plt.subplots(figsize=(8, 4))
				ax.plot(ts["y_true"], label="Actual")
				ax.plot(ts["preds"], label="Predicted")
				ax.legend()
				st.pyplot(fig, use_container_width=True)
				pred_df = pd.DataFrame({"actual": ts["y_true"], "predicted": ts["preds"]})
				st.download_button(
					"Download predictions (CSV)",
					pred_df.to_csv(index=False),
					file_name="time_series_predictions.csv",
				)
			if ts.get("residuals"):
				st.subheader("Residual Checks")
				fig, axes = plt.subplots(1, 2, figsize=(10, 4))
				sns.scatterplot(x=ts["preds"], y=ts["residuals"], ax=axes[0])
				axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
				axes[0].set_xlabel("Predicted")
				axes[0].set_ylabel("Residual")
				sns.histplot(ts["residuals"], kde=True, ax=axes[1])
				axes[1].set_title("Residual Distribution")
				st.pyplot(fig, use_container_width=True)
			st.stop()

		st.subheader("Baseline Results")
		st.json(results["baseline"]["metrics"])
		baseline_artifacts = results.get("baseline_artifacts", {})
		baseline_insights = results.get("baseline_insights", {})

		if problem_type == "regression" and baseline_artifacts:
			if show_predictions:
				st.subheader("Actual vs Predicted")
				fig, ax = plt.subplots(figsize=(6, 4))
				sns.scatterplot(
					x=baseline_artifacts["y_true"],
					y=baseline_artifacts["preds"],
					ax=ax,
				)
				ax.set_xlabel("Actual")
				ax.set_ylabel("Predicted")
				st.pyplot(fig, use_container_width=True)
				pred_df = pd.DataFrame(
					{"actual": baseline_artifacts["y_true"], "predicted": baseline_artifacts["preds"]}
				)
				st.download_button(
					"Download predictions (CSV)",
					pred_df.to_csv(index=False),
					file_name="regression_predictions.csv",
				)

			st.subheader("Residuals")
			fig, axes = plt.subplots(1, 2, figsize=(10, 4))
			sns.scatterplot(
				x=baseline_artifacts["preds"],
				y=baseline_artifacts["residuals"],
				ax=axes[0],
			)
			axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
			axes[0].set_xlabel("Predicted")
			axes[0].set_ylabel("Residual")
			sns.histplot(baseline_artifacts["residuals"], kde=True, ax=axes[1])
			axes[1].set_title("Residual Distribution")
			st.pyplot(fig, use_container_width=True)

		if problem_type == "classification" and baseline_artifacts:
			st.subheader("Confusion Matrix")
			fig, ax = plt.subplots(figsize=(4, 4))
			sns.heatmap(
				baseline_artifacts["confusion_matrix"],
				annot=True,
				fmt="d",
				cmap="Blues",
				ax=ax,
			)
			ax.set_xlabel("Predicted")
			ax.set_ylabel("Actual")
			st.pyplot(fig, use_container_width=False)

			roc_curve_data = baseline_artifacts.get("roc_curve")
			if roc_curve_data:
				st.subheader("ROC Curve")
				fig, ax = plt.subplots(figsize=(5, 4))
				ax.plot(roc_curve_data["fpr"], roc_curve_data["tpr"], color="blue")
				ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
				ax.set_xlabel("False Positive Rate")
				ax.set_ylabel("True Positive Rate")
				st.pyplot(fig, use_container_width=False)
			else:
				st.info("ROC curve is available for binary classification with probabilities.")

			proba_positive = baseline_artifacts.get("proba_positive")
			if proba_positive:
				st.subheader("Positive Class Probability")
				fig, ax = plt.subplots(figsize=(6, 3))
				sns.histplot(proba_positive, kde=True, ax=ax)
				ax.set_xlabel("Predicted probability")
				st.pyplot(fig, use_container_width=True)
				if results.get("model_name") == "Logistic Regression":
					st.subheader("Residual Checks")
					fig, axes = plt.subplots(1, 2, figsize=(10, 4))
					sns.scatterplot(
						x=proba_positive,
						y=baseline_artifacts.get("residuals", []),
						ax=axes[0],
					)
					axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
					axes[0].set_xlabel("Predicted probability")
					axes[0].set_ylabel("Residual")
					sns.histplot(baseline_artifacts.get("residuals", []), kde=True, ax=axes[1])
					axes[1].set_title("Residual Distribution")
					st.pyplot(fig, use_container_width=True)

		if baseline_insights:
			st.subheader("Model Insights")
			if baseline_insights.get("coefficients"):
				st.write("Top coefficients")
				st.dataframe(pd.DataFrame(baseline_insights["coefficients"]))
			if baseline_insights.get("feature_importances"):
				st.write("Top feature importances")
				st.dataframe(pd.DataFrame(baseline_insights["feature_importances"]))
			if baseline_insights.get("support_vectors"):
				st.write(f"Support vectors: {baseline_insights['support_vectors']}")
			if baseline_insights.get("n_neighbors"):
				st.write(f"Neighbors: {baseline_insights['n_neighbors']}")
		st.subheader("Selected Features")
		st.dataframe(pd.DataFrame({"feature": results["selected_features"]}))

		# Get tuned models from list
		tuned_models_list = st.session_state.get("tuned_models_list", [])
		first_tuned_model = tuned_models_list[0] if len(tuned_models_list) > 0 else None
		second_tuned_model = tuned_models_list[1] if len(tuned_models_list) > 1 else None

		# MAIN COMPARISON LAYOUT - First vs Second Tuned Model
		st.markdown("---")
		st.subheader("Tuned Models Comparison")
		
		left_col, right_col = st.columns(2)
		
		with left_col:
			if first_tuned_model:
				st.markdown("### 🟦 First Tuned Model")
				first_metrics = first_tuned_model.get("metrics", {})
				st.json(first_metrics)
				
				if first_tuned_model.get("best_params"):
					st.markdown("#### Best Parameters")
					st.json(first_tuned_model["best_params"])
				
				first_insights = first_tuned_model.get("insights", {})
				if first_insights:
					st.markdown("#### Model Insights")
					if first_insights.get("coefficients"):
						st.write("**Top Coefficients**")
						st.dataframe(pd.DataFrame(first_insights["coefficients"]), use_container_width=True)
					if first_insights.get("feature_importances"):
						st.write("**Top Feature Importances**")
						st.dataframe(pd.DataFrame(first_insights["feature_importances"]), use_container_width=True)
					if first_insights.get("support_vectors"):
						st.write(f"**Support Vectors:** {first_insights['support_vectors']}")
					if first_insights.get("n_neighbors"):
						st.write(f"**Neighbors:** {first_insights['n_neighbors']}")
				
				st.markdown("#### Selected Features")
				if first_tuned_model.get("selected_features"):
					st.dataframe(pd.DataFrame({"feature": first_tuned_model["selected_features"]}), use_container_width=True)
				else:
					st.write("Same as baseline")
			else:
				st.markdown("### 🟦 First Tuned Model")
				st.info("👉 Scroll down to generate your first tuned model")
		
		with right_col:
			if second_tuned_model:
				st.markdown("### 🟩 Second Tuned Model")
				second_metrics = second_tuned_model.get("metrics", {})
				st.json(second_metrics)
				
				if second_tuned_model.get("best_params"):
					st.markdown("#### Best Parameters")
					st.json(second_tuned_model["best_params"])
				
				second_insights = second_tuned_model.get("insights", {})
				if second_insights:
					st.markdown("#### Model Insights")
					if second_insights.get("coefficients"):
						st.write("**Top Coefficients**")
						st.dataframe(pd.DataFrame(second_insights["coefficients"]), use_container_width=True)
					if second_insights.get("feature_importances"):
						st.write("**Top Feature Importances**")
						st.dataframe(pd.DataFrame(second_insights["feature_importances"]), use_container_width=True)
					if second_insights.get("support_vectors"):
						st.write(f"**Support Vectors:** {second_insights['support_vectors']}")
					if second_insights.get("n_neighbors"):
						st.write(f"**Neighbors:** {second_insights['n_neighbors']}")
				
				st.markdown("#### Selected Features")
				if second_tuned_model.get("selected_features"):
					st.dataframe(pd.DataFrame({"feature": second_tuned_model["selected_features"]}), use_container_width=True)
				else:
					st.write("Same as baseline")
			else:
				st.markdown("### 🟩 Second Tuned Model")
				if first_tuned_model:
					st.info("👉 Scroll down to generate a second tuned model for comparison")
				else:
					st.info("Generate first tuned model to unlock second model comparison")
		
		# Metrics Comparison Table (First vs Second)
		if first_tuned_model and second_tuned_model:
			st.markdown("---")
			st.markdown("### Metrics Comparison (First → Second)")
			first_metrics = first_tuned_model.get("metrics", {})
			second_metrics = second_tuned_model.get("metrics", {})
			compare_rows = []
			metric_keys = sorted(set(first_metrics.keys()) | set(second_metrics.keys()))
			for key in metric_keys:
				first_val = first_metrics.get(key)
				second_val = second_metrics.get(key)
				improvement = None
				if first_val is not None and second_val is not None:
					try:
						improvement = float(second_val) - float(first_val)
					except:
						improvement = None
				compare_rows.append(
					{
						"Metric": key,
						"First Tuned": first_val,
						"Second Tuned": second_val,
						"Improvement": improvement,
					}
				)
			if compare_rows:
				comparison_df = pd.DataFrame(compare_rows)
				st.dataframe(comparison_df, use_container_width=True)
		
		# Handle errors from tuning
		if second_tuned_model and second_tuned_model.get("error"):
			st.error(second_tuned_model["error"])
		
		# Continue with visualizations
		display_model = second_tuned_model or first_tuned_model
		tuned_artifacts = display_model.get("artifacts", {}) if display_model else {}
		if problem_type == "regression" and tuned_artifacts:
			if show_predictions:
				st.subheader("Self-Tuned Actual vs Predicted")
				fig, ax = plt.subplots(figsize=(6, 4))
				sns.scatterplot(
					x=tuned_artifacts["y_true"],
					y=tuned_artifacts["preds"],
					ax=ax,
				)
				ax.set_xlabel("Actual")
				ax.set_ylabel("Predicted")
				st.pyplot(fig, use_container_width=True)
				pred_df = pd.DataFrame(
					{"actual": tuned_artifacts["y_true"], "predicted": tuned_artifacts["preds"]}
				)
				st.download_button(
					"Download self-tuned predictions (CSV)",
					pred_df.to_csv(index=False),
					file_name="regression_predictions_self_tuned.csv",
				)

			st.subheader("Self-Tuned Residuals")
			fig, axes = plt.subplots(1, 2, figsize=(10, 4))
			sns.scatterplot(
				x=tuned_artifacts["preds"],
				y=tuned_artifacts["residuals"],
				ax=axes[0],
			)
			axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
			axes[0].set_xlabel("Predicted")
			axes[0].set_ylabel("Residual")
			sns.histplot(tuned_artifacts["residuals"], kde=True, ax=axes[1])
			axes[1].set_title("Residual Distribution")
			st.pyplot(fig, use_container_width=True)

		if problem_type == "classification" and tuned_artifacts:
			st.subheader("Self-Tuned Confusion Matrix")
			fig, ax = plt.subplots(figsize=(4, 4))
			sns.heatmap(
				tuned_artifacts["confusion_matrix"],
				annot=True,
				fmt="d",
				cmap="Blues",
				ax=ax,
			)
			ax.set_xlabel("Predicted")
			ax.set_ylabel("Actual")
			st.pyplot(fig, use_container_width=False)

			roc_curve_data = tuned_artifacts.get("roc_curve")
			if roc_curve_data:
				st.subheader("Self-Tuned ROC Curve")
				fig, ax = plt.subplots(figsize=(5, 4))
				ax.plot(roc_curve_data["fpr"], roc_curve_data["tpr"], color="blue")
				ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
				ax.set_xlabel("False Positive Rate")
				ax.set_ylabel("True Positive Rate")
				st.pyplot(fig, use_container_width=False)

			proba_positive = tuned_artifacts.get("proba_positive")
			if proba_positive:
				st.subheader("Self-Tuned Positive Class Probability")
				fig, ax = plt.subplots(figsize=(6, 3))
				sns.histplot(proba_positive, kde=True, ax=ax)
				ax.set_xlabel("Predicted probability")
				st.pyplot(fig, use_container_width=True)
				if results.get("model_name") == "Logistic Regression":
					st.subheader("Self-Tuned Residual Checks")
					fig, axes = plt.subplots(1, 2, figsize=(10, 4))
					sns.scatterplot(
						x=proba_positive,
						y=tuned_artifacts.get("residuals", []),
						ax=axes[0],
					)
					axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
					axes[0].set_xlabel("Predicted probability")
					axes[0].set_ylabel("Residual")
					sns.histplot(tuned_artifacts.get("residuals", []), kde=True, ax=axes[1])
					axes[1].set_title("Residual Distribution")
					st.pyplot(fig, use_container_width=True)

			st.subheader("Tuned Results")
			st.json(results_tuned["baseline"]["metrics"])
			tuned_artifacts = results_tuned.get("baseline_artifacts", {})
			tuned_insights = results_tuned.get("baseline_insights", {})

			if problem_type == "regression" and tuned_artifacts:
				if show_predictions:
					st.subheader("Tuned: Actual vs Predicted")
					fig, ax = plt.subplots(figsize=(6, 4))
					sns.scatterplot(
						x=tuned_artifacts["y_true"],
						y=tuned_artifacts["preds"],
						ax=ax,
					)
					ax.set_xlabel("Actual")
					ax.set_ylabel("Predicted")
					st.pyplot(fig, use_container_width=True)
					pred_df = pd.DataFrame(
						{"actual": tuned_artifacts["y_true"], "predicted": tuned_artifacts["preds"]}
					)
					st.download_button(
						"Download tuned predictions (CSV)",
						pred_df.to_csv(index=False),
						file_name="tuned_regression_predictions.csv",
					)

				st.subheader("Tuned: Residuals")
				fig, axes = plt.subplots(1, 2, figsize=(10, 4))
				sns.scatterplot(
					x=tuned_artifacts["preds"],
					y=tuned_artifacts["residuals"],
					ax=axes[0],
				)
				axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
				axes[0].set_xlabel("Predicted")
				axes[0].set_ylabel("Residual")
				sns.histplot(tuned_artifacts["residuals"], kde=True, ax=axes[1])
				axes[1].set_title("Residual Distribution")
				st.pyplot(fig, use_container_width=True)

			if problem_type == "classification" and tuned_artifacts:
				st.subheader("Tuned: Confusion Matrix")
				fig, ax = plt.subplots(figsize=(4, 4))
				sns.heatmap(
					tuned_artifacts["confusion_matrix"],
					annot=True,
					fmt="d",
					cmap="Blues",
					ax=ax,
				)
				ax.set_xlabel("Predicted")
				ax.set_ylabel("Actual")
				st.pyplot(fig, use_container_width=False)

				roc_curve_data = tuned_artifacts.get("roc_curve")
				if roc_curve_data:
					st.subheader("Tuned: ROC Curve")
					fig, ax = plt.subplots(figsize=(5, 4))
					ax.plot(roc_curve_data["fpr"], roc_curve_data["tpr"], color="blue")
					ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
					ax.set_xlabel("False Positive Rate")
					ax.set_ylabel("True Positive Rate")
					st.pyplot(fig, use_container_width=False)

		col_a, col_b = st.columns([1, 1])
		with col_a:
			if st.button("Back to setup"):
				clear_tuned_state()
				st.session_state["step"] = 3
				st.session_state["scroll_to_top"] = True
				st.rerun()
		with col_b:
			if st.button("Start Over"):
				clear_workflow_state(keep_original=False)
				st.session_state["step"] = 0
				st.session_state["scroll_to_top"] = True
				st.rerun()


if __name__ == "__main__":
	main()

