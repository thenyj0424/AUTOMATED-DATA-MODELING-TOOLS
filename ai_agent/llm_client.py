import os
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


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


def call_groq(prompt: str, max_new_tokens: int = 160) -> Optional[str]:
	token = os.getenv("GROQ_API_KEY")
	if not token:
		return None

	try:
		client = ChatGroq(
			api_key=token,
			model=LLM_MODEL,
			temperature=0.2,
			max_tokens=max_new_tokens,
		)
		response = client.invoke([HumanMessage(content=prompt)])
		return response.content.strip() if response and response.content else ""
	except Exception as exc:
		return f"ERROR: Groq request failed - {exc}"


def explain_groq_error(error_text: str) -> str:
	text = (error_text or "").lower()
	if "401" in text or "unauthorized" in text or "invalid api key" in text:
		return "Groq authentication failed (401). Check whether GROQ_API_KEY is valid and active."
	if "429" in text or "rate" in text or "quota" in text:
		return "Groq rate limit or quota reached (429). Try again later or use a different key."
	if "404" in text or "model" in text and "not found" in text:
		return "Configured Groq model was not found. Check GROQ_MODEL in .env."
	if "timeout" in text or "connection" in text or "network" in text:
		return "Network or timeout error while contacting Groq."
	return "Groq request failed. Check key, model, and network settings."


def groq_token_loaded() -> bool:
	return bool(os.getenv("GROQ_API_KEY"))
