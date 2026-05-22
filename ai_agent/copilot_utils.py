from __future__ import annotations

import json
import re
from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ai_agent.llm_client import call_groq, groq_token_loaded
from ai_agent.rag_store import build_workflow_rag_context
from ai_agent.workflow_tools import perform_eda


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


def sanitize_user_facing_text(text: str) -> str:
	cleaned = str(text or "")
	replacements = [
		(r"\bthe users wants\b", "the request is"),
		(r"\bthe user wants\b", "the request is"),
		(r"\buser wants\b", "the request is"),
		(r"\bthe user would like\b", "the request is"),
		(r"\bthe user is asking\b", "the request is"),
		(r"\bas the user wants\b", "as requested"),
		(r"\bself interpretation\b", "neutral summary"),
	]
	for pattern, replacement in replacements:
		cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\s+", " ", cleaned).strip()
	return cleaned


def strip_internal_plan_payload(text: str) -> str:
	cleaned = str(text or "").strip()
	if "{" in cleaned and "}" in cleaned:
		start = cleaned.find("{")
		end = cleaned.rfind("}")
		if end > start:
			candidate = cleaned[start : end + 1]
			try:
				payload = json.loads(candidate)
			except Exception:
				payload = None
			if isinstance(payload, dict) and any(key in payload for key in ("changes", "activities", "priority_source")):
				cleaned = (cleaned[:start] + " " + cleaned[end + 1 :]).strip()
	cleaned = re.sub(r"\s+", " ", cleaned).strip()
	return cleaned


def condense_user_facing_text(text: str, max_sentences: int = 3, max_words: int = 80) -> str:
	cleaned = str(text or "").strip()
	if not cleaned:
		return cleaned
	sentences = re.split(r"(?<=[.!?])\s+", cleaned)
	if len(sentences) > max_sentences:
		cleaned = " ".join(sentences[:max_sentences]).strip()
	words = cleaned.split()
	if len(words) > max_words:
		cleaned = " ".join(words[:max_words]).rstrip(" ,;:") + "..."
	return re.sub(r"\s+", " ", cleaned).strip()


def is_meta_help_request(user_text: str) -> bool:
	lower = str(user_text or "").strip().lower()
	if not lower:
		return False
	phrases = [
		"explain my request",
		"explain the request",
		"what did i ask",
		"summarize my request",
		"summarise my request",
		"clarify my request",
		"what am i asking",
		"what am i requesting",
	]
	return any(phrase in lower for phrase in phrases)


def build_meta_help_reply(user_text: str, current_goal: str = "") -> str:
	goal = sanitize_user_facing_text(current_goal or "")
	if goal:
		return (
			f"Your request is: {goal}. "
			"I can also turn it into a short checklist if you want."
		)
	return (
		"Your request is to explain the current instruction in plain language. "
		"I can summarize it in one sentence or turn it into a short checklist."
	)


def apply_requirement_to_state(req_text: str) -> List[str]:
	"""Parse a recorded requirement text and apply relevant session_state keys.

	Returns a list of human-readable activity notes that were applied.
	"""
	notes: List[str] = []
	def _normalize_request_text_local(text: str) -> str:
		return str(text or "").lower().replace("-", " ").replace("_", " ")

	txt = _normalize_request_text_local(req_text)

	# Time-series hints
	if any(token in txt for token in ["time series", "timeseries", "forecast", "arima", "holt", "winters", "seasonal"]):
		st.session_state["problem_type"] = "time_series"
		notes.append("Applied requirement: problem_type = time_series")
		if "holt" in txt or "winters" in txt:
			st.session_state["time_series_model"] = "Holt-Winters"
			notes.append("Applied requirement: time_series_model = Holt-Winters")
		if "seasonal 12" in txt or "period 12" in txt or "m=12" in txt:
			st.session_state["hw_seasonal_periods"] = 12
			notes.append("Applied requirement: hw_seasonal_periods = 12")

	# Model family preferences as flexible hints — do not force a concrete family
	if any(token in txt for token in ["ml", "machine learning", "random forest", "decision tree", "xgboost", "knn", "svm", "svr"]):
		st.session_state.setdefault("model_selection_hint", {})
		st.session_state["model_selection_hint"].update({"family": "ML", "flexible": True})
		notes.append("Applied requirement: prefer ML models (hint set)")

	if any(token in txt for token in ["statistical", "linear regression", "logistic", "ridge", "lasso"]):
		st.session_state.setdefault("model_selection_hint", {})
		st.session_state["model_selection_hint"].update({"family": "Statistical", "flexible": True})
		notes.append("Applied requirement: prefer Statistical models (hint set)")

	# Explicit model names
	if "decision tree" in txt or "dtree" in txt or ("decision" in txt and "tree" in txt):
		st.session_state["model_name"] = "Decision Tree"
		st.session_state.setdefault("model_family", "ML")
		notes.append("Applied requirement: model_name = Decision Tree")
	if "random forest" in txt or "randomforest" in txt or " rf " in f" {txt} ":
		st.session_state["model_name"] = "Random Forest"
		st.session_state.setdefault("model_family", "ML")
		notes.append("Applied requirement: model_name = Random Forest")
	if "knn" in txt or "k nearest" in txt:
		st.session_state["model_name"] = "KNN"
		st.session_state.setdefault("model_family", "ML")
		notes.append("Applied requirement: model_name = KNN")
	if "logistic" in txt:
		st.session_state["model_name"] = "Logistic Regression"
		st.session_state.setdefault("model_family", "Statistical")
		notes.append("Applied requirement: model_name = Logistic Regression")
	if "linear regression" in txt:
		st.session_state["model_name"] = "Linear Regression"
		st.session_state.setdefault("model_family", "Statistical")
		notes.append("Applied requirement: model_name = Linear Regression")

	# If user asked to 'use any ML model', set a preference hint but do not hard-choose the model
	if any(phrase in txt for phrase in ["use any ml model", "use any ml", "use an ml model"]):
		st.session_state.setdefault("model_selection_hint", {})
		st.session_state["model_selection_hint"].update({"family": "ML", "flexible": True})
		notes.append("Applied requirement: prefer ML models (no hard model selected)")

	# Performance / tradeoff hints
	if any(token in txt for token in ["speed", "faster", "quick", "latency", "fast", "lightweight"]):
		st.session_state.setdefault("model_selection_hint", {})
		st.session_state["model_selection_hint"].update({"speed": "high", "flexible": True})
		notes.append("Applied requirement: prioritize speed / lightweight models (hint set)")
	if any(token in txt for token in ["accurate", "accuracy", "precision", "best accuracy", "maximize accuracy"]):
		st.session_state.setdefault("model_selection_hint", {})
		st.session_state["model_selection_hint"].update({"accuracy": "high", "flexible": True})
		notes.append("Applied requirement: prioritize accuracy (hint set)")
	if any(token in txt for token in ["interpretable", "explainable", "transparent"]):
		st.session_state.setdefault("model_selection_hint", {})
		st.session_state["model_selection_hint"].update({"interpretability": True, "flexible": True})
		notes.append("Applied requirement: prefer interpretable / explainable models (hint set)")

	# Persist a simple flag so other code paths can notice a recent requirement application
	st.session_state["agent_requirement_last_applied"] = datetime.now().isoformat()
	return notes


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
	lower = (user_text or "").lower()

	# Model family hints
	if any(tok in lower for tok in ["ml", "machine learning"]):
		prefs["preferred_model_family"] = "ML"
		updates.append("Preference saved: prefer ML models")

	if any(tok in lower for tok in ["statistical", "linear regression", "logistic", "ridge", "lasso"]):
		prefs["preferred_model_family"] = "Statistical"
		updates.append("Preference saved: prefer Statistical models")

	# Problem type hints
	if any(tok in lower for tok in ["time series", "timeseries", "forecast", "arima", "holt", "winters", "seasonal"]):
		prefs["problem_type"] = "time_series"
		updates.append("Preference saved: treat as time_series problem")

	# Optimization priorities/tradeoffs
	if any(tok in lower for tok in ["speed", "faster", "quick", "latency", "fast"]):
		prefs["priority"] = "speed"
		updates.append("Preference saved: prioritize speed")
	elif any(tok in lower for tok in ["accurate", "accuracy", "precision", "best accuracy", "maximize accuracy"]):
		prefs["priority"] = "accuracy"
		updates.append("Preference saved: prioritize accuracy")
	elif any(tok in lower for tok in ["interpretable", "explainable", "transparent"]):
		prefs["priority"] = "interpretability"
		updates.append("Preference saved: prioritize interpretability")

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
		"System policy: Auto AI records requirement-style chat messages and selects actions for the workflow."
		" Acknowledge requirements with a short 'Noted' and record them. Do not mention which model is speaking."
		" Avoid self-interpretations such as 'the user wants'. Do not echo internal planning JSON or tool payloads."
		" Use neutral phrasing, answer with the result first, keep replies short, and ask clarifying questions only when the user explicitly requests a recommendation or asks a question."
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


def get_active_target_column() -> Optional[str]:
	for key in ("eda_target_col", "target_col"):
		value = st.session_state.get(key)
		if value and str(value).strip() and str(value) != "(none)":
			return str(value)
	return None


def is_dataset_question(user_text: str, step: int, df: Optional[pd.DataFrame], summary: Any) -> bool:
	if not user_text or df is None or summary is None:
		return False
	if is_requirement_message(user_text):
		return False
	lower = user_text.strip().lower()
	dataset_tokens = [
		"dataset",
		"data",
		"csv",
		"rows",
		"row",
		"columns",
		"column",
		"missing",
		"null",
		"na",
		"target",
		"feature",
		"correlation",
		"outlier",
		"distribution",
		"stationarity",
		"time series",
		"forecast",
		"trend",
		"model",
		"predict",
	]
	if any(token in lower for token in dataset_tokens):
		return True
	if "?" in lower and any(
		token in lower
		for token in [
			"what should i do",
			"which model",
			"best model",
			"what about",
			"can you",
			"should i",
			"does the dataset",
			"tell me about",
			"how many",
		]
	):
		return True
	if step >= 1 and any(
		token in lower
		for token in ["summary", "missing", "feature", "target", "distribution", "stationarity", "trend"]
	):
		return True
	return False


def build_dataset_analysis_context(
	step: int,
	df: pd.DataFrame,
	summary: Any,
	user_text: str,
) -> Dict[str, Any]:
	target_col = get_active_target_column()
	analysis = perform_eda(df, target_col=target_col)
	compact = {
		"step": step,
		"user_question": user_text,
		"target_col": target_col,
		"profile": analysis.get("profile", {}),
		"problem_type": analysis.get("problem_type"),
		"recommended_sections": analysis.get("recommended_sections", []),
		"missingness": analysis.get("missingness", {}),
		"stationarity": analysis.get("stationarity", {}),
	}
	return {
		"target_col": target_col,
		"analysis": analysis,
		"prompt_context": json.dumps(compact, ensure_ascii=True, default=str),
	}


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
