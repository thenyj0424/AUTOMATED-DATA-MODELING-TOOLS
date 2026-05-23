from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_agent.llm_client import retrieve_knowledge, format_knowledge_context, KNOWLEDGE_BASE_DIR


def knowledge_base_path() -> Path:
	return KNOWLEDGE_BASE_DIR


def retrieve_rag_context(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
	return retrieve_knowledge(query, top_k=top_k)


def build_rag_context_text(query: str, top_k: int = 3) -> str:
	return format_knowledge_context(query, top_k=top_k)


def build_workflow_rag_query(step: int, user_goal: str = "", column_text: str = "", summary_text: str = "") -> str:
	parts = ["System Adaptation Protocol", "Human Interaction Protocol", "greeting", "requirement handling", "irrelevant request refusal"]
	if step == 1:
		parts.extend(["EDA Selection Logic", "system adaptation usage", "missingness", "target relationships", "outliers"])
	elif step == 2:
		parts.extend(["Data Cleaning Techniques", "system adaptation usage", "missing values", "imputation", "outlier review"])
	elif step == 3:
		parts.extend(["Data Modelling Selection", "system adaptation usage", "classification", "regression", "time series", "model family", "model name"])
	elif step == 4:
		parts.extend(["Model Evaluations", "baseline", "tuned", "metrics", "residuals"])
	if user_goal:
		parts.append(user_goal)
	if column_text:
		parts.append(column_text)
	if summary_text:
		parts.append(summary_text)
	return " ".join(parts)


def _default_top_k(step: int) -> int:
	if step == 3:
		return 3
	return 2


def build_workflow_rag_context(step: int, user_goal: str = "", column_text: str = "", summary_text: str = "", top_k: Optional[int] = None) -> str:
	query = build_workflow_rag_query(step, user_goal=user_goal, column_text=column_text, summary_text=summary_text)
	use_top_k = top_k if top_k is not None else _default_top_k(step)
	# Keep retrieval bounded for free-tier efficiency.
	if use_top_k < 1:
		use_top_k = 1
	if use_top_k > 4:
		use_top_k = 4
	return format_knowledge_context(query, top_k=use_top_k)