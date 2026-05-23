import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DEFAULT_ROUTER_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip().strip('"').strip("'")
DEFAULT_REASONER_MODEL = os.getenv("GROQ_MODEL2", "meta-llama/llama-4-scout-17b-16e-instruct").strip().strip('"').strip("'")
DEFAULT_REVIEWER_MODEL = os.getenv("GROQ_MODEL3", "llama-3.3-70b-versatile").strip().strip('"').strip("'")
DEFAULT_ROUTER_MODEL_BACKUP = os.getenv("GROQ_MODEL_BACKUP", DEFAULT_ROUTER_MODEL).strip().strip('"').strip("'")
DEFAULT_REASONER_MODEL_BACKUP = os.getenv("GROQ_MODEL_BACKUP2", DEFAULT_REASONER_MODEL).strip().strip('"').strip("'")
DEFAULT_REVIEWER_MODEL_BACKUP = os.getenv("GROQ_MODEL_BACKUP3", DEFAULT_REVIEWER_MODEL).strip().strip('"').strip("'")
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
DEFAULT_GROQ_API_KEY_BACKUP = os.getenv("GROQ_API_KEY_BACKUP", "").strip().strip('"').strip("'")

LLM_MODEL = DEFAULT_ROUTER_MODEL

ROUTER_MODEL = "router"
REASONER_MODEL = "reasoner"
REVIEWER_MODEL = "reviewer"

RAG_MODES = {"router", "reasoner", "reviewer", "default"}

MODEL_FALLBACKS = {
	DEFAULT_ROUTER_MODEL: [
		DEFAULT_ROUTER_MODEL,
		DEFAULT_REASONER_MODEL_BACKUP,
		DEFAULT_REVIEWER_MODEL_BACKUP,
		DEFAULT_ROUTER_MODEL_BACKUP,
	],
	DEFAULT_REASONER_MODEL: [
		DEFAULT_REASONER_MODEL,
		DEFAULT_REVIEWER_MODEL_BACKUP,
		DEFAULT_ROUTER_MODEL_BACKUP,
		DEFAULT_REASONER_MODEL_BACKUP,
	],
	DEFAULT_REVIEWER_MODEL: [
		DEFAULT_REVIEWER_MODEL,
		DEFAULT_ROUTER_MODEL_BACKUP,
		DEFAULT_REASONER_MODEL_BACKUP,
		DEFAULT_REVIEWER_MODEL_BACKUP,
	],
}


def _normalize_env_value(value: Optional[str]) -> str:
	return str(value or "").strip().strip('"').strip("'")


def _model_chain_for(model_name: str) -> List[str]:
	chain = MODEL_FALLBACKS.get(model_name, [model_name])
	if model_name not in chain:
		chain = [model_name] + list(chain)
	seen: set[str] = set()
	ordered: List[str] = []
	for candidate in chain:
		if not candidate or candidate in seen:
			continue
		seen.add(candidate)
		ordered.append(candidate)
	return ordered


def _token_chain() -> List[str]:
	primary_token = os.getenv("GROQ_API_KEY", DEFAULT_GROQ_API_KEY).strip().strip('"').strip("'")
	backup_token = os.getenv("GROQ_API_KEY_BACKUP", DEFAULT_GROQ_API_KEY_BACKUP).strip().strip('"').strip("'")
	return [token for token in [primary_token, backup_token] if token]


def get_model_name(mode: str = "default") -> str:
	if mode == REVIEWER_MODEL:
		return DEFAULT_REVIEWER_MODEL
	if mode == REASONER_MODEL:
		return DEFAULT_REASONER_MODEL
	return DEFAULT_ROUTER_MODEL


def _normalize_task_type(task_type: Optional[str]) -> str:
	value = (task_type or "").strip().lower()
	if value in {"intent", "routing", "classify", "classification", "chat", "conversation", "structured_extraction"}:
		return ROUTER_MODEL
	if value in {"reason", "reasoning", "plan", "orchestrate", "orchestration", "rag_answer", "qa", "pipeline"}:
		return REASONER_MODEL
	if value in {"review", "expert_review", "validate", "validation", "debug", "high_risk"}:
		return REVIEWER_MODEL
	return "default"


def route_model(task_type: Optional[str] = None, prompt: str = "", context: str = "") -> Tuple[str, str]:
	normalized = _normalize_task_type(task_type)
	if normalized == ROUTER_MODEL:
		return DEFAULT_ROUTER_MODEL, ROUTER_MODEL
	if normalized == REASONER_MODEL:
		return DEFAULT_REASONER_MODEL, REASONER_MODEL
	if normalized == REVIEWER_MODEL:
		return DEFAULT_REVIEWER_MODEL, REVIEWER_MODEL

	text = f"{prompt}\n{context}".lower()
	if any(token in text for token in ["intent", "classify", "route", "extract", "conversation", "chat"]):
		return DEFAULT_ROUTER_MODEL, ROUTER_MODEL
	if any(token in text for token in ["review", "validate", "uncertain", "risk", "edge case", "critical"]):
		return DEFAULT_REVIEWER_MODEL, REVIEWER_MODEL
	return DEFAULT_REASONER_MODEL, REASONER_MODEL


def _clean_text(value: str) -> str:
	text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
	return text


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
	text = content or ""
	if not text.lstrip().startswith("---"):
		return {}, text
	parts = text.split("---", 2)
	if len(parts) < 3:
		return {}, text
	meta_block = parts[1]
	body = parts[2].lstrip("\n")
	meta: Dict[str, Any] = {}
	for raw_line in meta_block.splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or ":" not in line:
			continue
		key, value = line.split(":", 1)
		key = key.strip().lower()
		value = value.strip()
		if not value:
			meta[key] = ""
			continue
		if value.startswith("[") and value.endswith("]"):
			items = [item.strip().strip('"').strip("'") for item in value[1:-1].split(",") if item.strip()]
			meta[key] = items
		elif key == "priority":
			try:
				meta[key] = int(value)
			except Exception:
				meta[key] = 0
		else:
			meta[key] = value.strip('"').strip("'")
	return meta, body


def _read_knowledge_chunks() -> List[Dict[str, str]]:
	chunks_dir = KNOWLEDGE_BASE_DIR / "chunks"
	if not chunks_dir.exists():
		return []
	chunks: List[Dict[str, str]] = []
	for path in sorted(chunks_dir.glob("*.md")):
		try:
			content = path.read_text(encoding="utf-8")
		except Exception:
			continue
		meta, body = _parse_frontmatter(content)
		title = str(meta.get("title") or path.stem.replace("_", " ").strip())
		tags = meta.get("tags") or []
		if isinstance(tags, str):
			tags = [tags]
		chunks.append(
			{
				"title": title,
				"path": str(path),
				"tags": [str(tag).lower() for tag in tags if str(tag).strip()],
				"summary": str(meta.get("summary", "")),
				"priority": int(meta.get("priority", 0) or 0),
				"content": body,
			}
		)
	return chunks


def retrieve_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
	query_text = _clean_text(query)
	if not query_text:
		return []
	query_tokens = {token for token in query_text.split() if len(token) > 2}
	results: List[Dict[str, Any]] = []
	for chunk in _read_knowledge_chunks():
		content = chunk.get("content", "")
		content_text = _clean_text(content)
		title_text = _clean_text(chunk.get("title", ""))
		tag_text = " ".join(chunk.get("tags", []))
		summary_text = _clean_text(chunk.get("summary", ""))
		priority = int(chunk.get("priority", 0) or 0)
		score = 0
		if query_text in title_text or query_text in summary_text or query_text in content_text:
			score += 5
		for token in query_tokens:
			if token in title_text:
				score += 3
			if token in tag_text:
				score += 2
			if token in summary_text:
				score += 2
			if token in content_text:
				score += 1
		score += max(0, priority)
		if score <= 0:
			continue
		results.append(
			{
				"title": chunk.get("title", ""),
				"path": chunk.get("path", ""),
				"tags": chunk.get("tags", []),
				"priority": priority,
				"score": score,
				"snippet": content.strip()[:500],
			}
		)
	results.sort(key=lambda item: (-int(item.get("score", 0)), str(item.get("title", ""))))
	return results[:top_k]


def format_knowledge_context(query: str, top_k: int = 3) -> str:
	results = retrieve_knowledge(query, top_k=top_k)
	if not results:
		return ""
	parts = []
	for item in results:
		title = item.get("title", "knowledge")
		snippet = item.get("snippet", "")
		tags = item.get("tags", [])
		tag_text = f" tags={', '.join(tags)}" if tags else ""
		parts.append(f"[{title}{tag_text}] {snippet}")
	return "\n\n".join(parts)


def build_prompt(overview_text: str, missing_text: str, num_text: str) -> str:
	return (
		"You are a data scientist. Be brief and direct.\n"
		"Summarize the dataset and suggest modeling directions.\n\n"
		f"Overview:\n{overview_text}\n"
		f"Missing values per column:\n{missing_text}\n\n"
		f"Numeric summary (top rows):\n{num_text}\n\n"
		"Provide: 1) summary (2-3 sentences), 2) possible targets, 3) modeling suggestions.\n"
		"Keep under 120 words."
	)


def build_system_message_prompt(step_name: str, context: str) -> str:
	return (
		"You are a helpful assistant guiding a user through a data workflow. "
		"Provide 2 short, actionable tips for the current step. Keep under 40 words.\n"
		f"Step: {step_name}\n"
		f"Context: {context}\n"
	)


def call_groq(
	prompt: str,
	max_new_tokens: int = 160,
	task_type: Optional[str] = None,
	context: str = "",
	force_model: Optional[str] = None,
) -> Optional[str]:
	token_chain = _token_chain()
	if not token_chain:
		return None

	try:
		model_name, routed_mode = route_model(task_type=task_type, prompt=prompt, context=context)
		if force_model:
			model_name = get_model_name(force_model) if force_model in RAG_MODES else force_model
		fallback_chain = _model_chain_for(model_name)
		last_error: Optional[Exception] = None
		for index, candidate_model in enumerate(fallback_chain):
			candidate_token = token_chain[min(index, len(token_chain) - 1)]
			if not candidate_model or not candidate_token:
				continue
			try:
				client = ChatGroq(
					api_key=candidate_token,
					model=candidate_model,
					temperature=0.2,
					max_tokens=max_new_tokens,
				)
				response = client.invoke([HumanMessage(content=prompt)])
				if response and response.content:
					return response.content.strip()
			except Exception as exc:
				last_error = exc
				# Continue to the next fallback model/key pair for transient failures such as
				# rate limits, quota limits, timeouts, or model-not-found errors.
				continue
		if last_error is not None:
			return f"ERROR: Groq request failed for model {model_name} - {last_error}"
		return ""
	except Exception as exc:
		return f"ERROR: Groq request failed - {exc}"


def build_model_routing_prompt(user_text: str, context: str = "") -> str:
	return (
		"Classify the request into one of these labels only: router, reasoner, reviewer. "
		"Use router for intent classification, structured extraction, simple chat handling, and routing decisions. "
		"Use reasoner for main planning, RAG synthesis, and full modelling orchestration. "
		"Use reviewer for uncertain, high-risk, or edge-case validation. Return only one label.\n"
		f"User request: {user_text}\n"
		f"Context: {context}"
	)


def classify_task_route(user_text: str, context: str = "") -> str:
	prompt = build_model_routing_prompt(user_text, context)
	text = call_groq(prompt, max_new_tokens=32, task_type="intent", context=context)
	label = (text or "").strip().lower()
	if "reviewer" in label:
		return REVIEWER_MODEL
	if "reasoner" in label:
		return REASONER_MODEL
	return ROUTER_MODEL


def should_use_reviewer(user_text: str, context: str = "") -> bool:
	combined = f"{user_text}\n{context}".lower()
	return any(token in combined for token in ["uncertain", "edge case", "high risk", "validate", "critical", "debug"])


def explain_groq_error(error_text: str) -> str:
	text = (error_text or "").lower()
	if "401" in text or "unauthorized" in text or "invalid api key" in text:
		return "Groq authentication failed (401). Check whether GROQ_API_KEY is valid and active."
	if "429" in text or "rate" in text or "quota" in text:
		return "Groq rate limit or quota reached (429). Try again later or use a different key."
	if "404" in text or ("model" in text and "not found" in text):
		return (
			"Configured Groq model was not found. Check whether the selected model is enabled for your Groq account. "
			"The app will try the other configured models as fallback."
		)
	if "timeout" in text or "connection" in text or "network" in text:
		return "Network or timeout error while contacting Groq."
	return "Groq request failed. Check key, model, and network settings."


def groq_token_loaded() -> bool:
	return bool(_token_chain())
