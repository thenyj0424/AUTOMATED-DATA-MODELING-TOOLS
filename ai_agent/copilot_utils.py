from __future__ import annotations

from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ai_agent.llm_client import call_groq, groq_token_loaded
from ai_agent.rag_store import build_workflow_rag_context


def is_requirement_message(user_text: str) -> bool:
	"""Heuristic to detect requirement-style messages.

	Treat messages as requirements when the user labels them (e.g. "Requirement:")
	or when the text is declarative (no question mark) and reasonably long.
	"""
	if not user_text:
		return False
	txt = user_text.strip()
	lower = txt.lower()
	if lower.startswith(("requirement:", "requirement -", "req:", "req -")):
		return True
	if "requirement" in lower:
		return True
	# Declarative heuristic: no question mark and at least 5 words
	if "?" not in txt and len(txt.split()) >= 5:
		return True
	return False


def _last_match_index(text: str, phrases: List[str]) -> int:
	best = -1
	for phrase in phrases:
		idx = text.rfind(phrase)
		if idx > best:
			best = idx
	return best


def update_preferences_from_text(
	user_text: str,
	current_preferences: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
	prefs: Dict[str, Any] = dict(current_preferences or {})
	updates: List[str] = []
	lower = user_text.lower()

	ml_idx = _last_match_index(
		lower,
		[
			"prefer ml",
			"use ml",
			"machine learning",
			"ml model",
		],
	)
	stat_idx = _last_match_index(
		lower,
		[
			"prefer statistical",
			"statistical model",
			"classical model",
		],
	)
	if ml_idx >= 0 or stat_idx >= 0:
		new_family = "ML" if ml_idx > stat_idx else "Statistical"
		if prefs.get("preferred_model_family") != new_family:
			prefs["preferred_model_family"] = new_family
			updates.append(f"Preference saved: model family = {new_family}")

	if any(token in lower for token in ["interpretability", "interpretable", "explainable"]):
		if prefs.get("priority") != "interpretability":
			prefs["priority"] = "interpretability"
			updates.append("Preference saved: prioritize interpretability")
	elif any(token in lower for token in ["accuracy", "best performance", "highest score"]):
		if prefs.get("priority") != "accuracy":
			prefs["priority"] = "accuracy"
			updates.append("Preference saved: prioritize accuracy")
	elif any(token in lower for token in ["speed", "faster", "quick", "latency"]):
		if prefs.get("priority") != "speed":
			prefs["priority"] = "speed"
			updates.append("Preference saved: prioritize speed")

	return prefs, updates


def format_preferences_for_context(preferences: Optional[Dict[str, Any]]) -> str:
	prefs = preferences or {}
	parts: List[str] = []
	if prefs.get("preferred_model_family"):
		parts.append(f"Model family preference: {prefs['preferred_model_family']}")
	if prefs.get("priority"):
		parts.append(f"Optimization priority: {prefs['priority']}")
	return " | ".join(parts)


def add_agent_activity(message: str, level: str = "info") -> None:
	activity = st.session_state.get("agent_activity", [])
	activity.insert(
		0,
		{
			"time": datetime.now().strftime("%H:%M:%S"),
			"ts": time.time(),
			"level": level,
			"message": message,
		},
	)
	st.session_state["agent_activity"] = activity[:20]


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


def build_copilot_context(step: int, df: Optional[pd.DataFrame], summary: Any) -> str:
	parts = [f"Step: {step}"]
	# System policy: treat requirement-style chat messages as record-only.
	parts.append(
		"System policy: When the user sends a requirement (a declarative instruction),"
		" acknowledge with a short 'Noted' and record it. Do not ask clarifying"
		" questions unless the user explicitly requests a recommendation or asks a question."
	)
	current_goal = st.session_state.get("agent_goal")
	if current_goal:
		parts.append(f"Current user request: {current_goal}")
	preference_text = format_preferences_for_context(st.session_state.get("agent_preferences"))
	if preference_text:
		parts.append(f"Stored user preferences: {preference_text}")
	if summary is not None:
		parts.append(
			f"Dataset: rows={summary.rows}, cols={summary.cols}, numeric={len(summary.numeric_cols)}, categorical={len(summary.categorical_cols)}, datetime={len(summary.datetime_cols)}"
		)
		missing_total = int(summary.missing_by_col["missing_count"].sum()) if hasattr(summary, "missing_by_col") and "missing_count" in summary.missing_by_col else None
		if missing_total is not None:
			parts.append(f"Missing values total: {missing_total}")
	if df is not None:
		parts.append(f"Columns: {', '.join(map(str, list(df.columns)[:20]))}")
	current_results = st.session_state.get("results")
	if current_results:
		result_summary = summarize_results_for_ai(current_results)
		if result_summary:
			parts.append(f"Latest results: {result_summary}")
	last_outcome = st.session_state.get("agent_last_outcome")
	if last_outcome:
		parts.append(f"Previous outcome: {last_outcome}")
	last_activity = st.session_state.get("agent_activity", [])[:3]
	if last_activity:
		activity_bits = [item.get("message", "") for item in last_activity if item.get("message")]
		if activity_bits:
			parts.append(f"Recent AI actions: {' | '.join(activity_bits)}")
	column_text = " ".join(map(str, list(df.columns)[:20])) if df is not None else ""
	summary_text = ""
	if summary is not None:
		summary_text = f"rows={summary.rows} cols={summary.cols} numeric={len(summary.numeric_cols)} categorical={len(summary.categorical_cols)} datetime={len(summary.datetime_cols)}"
	knowledge_context = build_workflow_rag_context(
		step,
		user_goal=str(current_goal or ""),
		column_text=column_text,
		summary_text=summary_text,
	)
	if knowledge_context:
		parts.append(f"Knowledge base guidance:\n{knowledge_context}")
	return "\n".join(parts)


def build_hybrid_step_hint(step: int, df: Optional[pd.DataFrame], summary: Any) -> str:
	rule_hint = build_rule_based_step_hint(step, df, summary)
	if not groq_token_loaded():
		return rule_hint
	prompt = (
		"You are an AI copilot in a data app. Use the rule hint and context to provide one short actionable line. "
		"Do not exceed 30 words. Focus on the current step and prior outcome if present.\n"
		f"Rule hint: {rule_hint}\n"
		f"Context:\n{build_copilot_context(step, df, summary)}"
	)
	text = call_groq(prompt, max_new_tokens=60, task_type="reasoning")
	if text and not text.startswith("ERROR:"):
		return text
	return rule_hint


def summarize_results_for_ai(results: Optional[Dict[str, Any]]) -> str:
	if not results:
		return ""
	problem_type = results.get("problem_type", "")
	model_name = results.get("model_name", "")
	parts = [f"Problem type: {problem_type}", f"Model: {model_name}"]
	baseline = results.get("baseline", {})
	metrics = baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
	if metrics:
		metric_bits = []
		for key in list(metrics.keys())[:4]:
			metric_bits.append(f"{key}={metrics[key]}")
		parts.append("Baseline metrics: " + ", ".join(metric_bits))
	selected_features = results.get("selected_features") or []
	if selected_features:
		parts.append(f"Selected features: {', '.join(map(str, selected_features[:8]))}")
	if problem_type == "time_series":
		ts = results.get("time_series", {})
		if ts.get("metrics"):
			parts.append(f"Time series metrics: {ts['metrics']}")
	return " | ".join(parts)
