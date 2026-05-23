#!/usr/bin/env python
"""Standalone Groq smoke test for the configured router, reasoner, and reviewer models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

if str(BASE_DIR) not in sys.path:
	sys.path.insert(0, str(BASE_DIR))

from ai_agent.llm_client import (
	DEFAULT_REASONER_MODEL,
	DEFAULT_REASONER_MODEL_BACKUP,
	DEFAULT_REVIEWER_MODEL,
	DEFAULT_REVIEWER_MODEL_BACKUP,
	DEFAULT_ROUTER_MODEL,
	DEFAULT_ROUTER_MODEL_BACKUP,
	call_groq,
)


DEFAULT_PROMPTS = {
	"intent": "Reply with a single short sentence that says the router is working.",
	"reasoning": "Reply with a single short sentence that says the reasoner is working.",
	"review": "Reply with a single short sentence that says the reviewer is working.",
}


def _resolve_model_label(model_name: str) -> str:
	if model_name == DEFAULT_ROUTER_MODEL:
		return "router"
	if model_name == DEFAULT_ROUTER_MODEL_BACKUP:
		return "router-backup"
	if model_name == DEFAULT_REASONER_MODEL:
		return "reasoner"
	if model_name == DEFAULT_REASONER_MODEL_BACKUP:
		return "reasoner-backup"
	if model_name == DEFAULT_REVIEWER_MODEL:
		return "reviewer"
	if model_name == DEFAULT_REVIEWER_MODEL_BACKUP:
		return "reviewer-backup"
	return "custom"


def _masked_key(value: str) -> str:
	if not value:
		return "missing"
	if len(value) <= 8:
		return "present"
	return f"{value[:4]}...{value[-4:]}"


def _print_config_summary() -> None:
	models = {
		"router": (os.getenv("GROQ_MODEL", "").strip(), os.getenv("GROQ_MODEL_BACKUP", "").strip()),
		"reasoner": (os.getenv("GROQ_MODEL2", "").strip(), os.getenv("GROQ_MODEL_BACKUP2", "").strip()),
		"reviewer": (os.getenv("GROQ_MODEL3", "").strip(), os.getenv("GROQ_MODEL_BACKUP3", "").strip()),
	}
	primary_key = os.getenv("GROQ_API_KEY", "").strip()
	backup_key = os.getenv("GROQ_API_KEY_BACKUP", "").strip()

	print("Groq configuration summary:")
	for label, (primary_model, backup_model) in models.items():
		print(f"- {label}: primary={primary_model or 'missing'} | backup={backup_model or 'missing'}")
		if primary_model and backup_model and primary_model == backup_model:
			print("  warning: primary and backup model are identical")
	print(f"- primary key: {_masked_key(primary_key)}")
	print(f"- backup key: {_masked_key(backup_key)}")
	if not primary_key and not backup_key:
		print("  warning: no Groq key found in .env")


def _build_probe_plan(check_all: bool, task_type: str, model: str) -> list[tuple[str, str, str]]:
	if check_all:
		return [
			("router", DEFAULT_ROUTER_MODEL, "Reply with a single short sentence that says the router is working."),
			("router-backup", DEFAULT_ROUTER_MODEL_BACKUP, "Reply with a single short sentence that says the router backup is working."),
			("reasoner", DEFAULT_REASONER_MODEL, "Reply with a single short sentence that says the reasoner is working."),
			("reasoner-backup", DEFAULT_REASONER_MODEL_BACKUP, "Reply with a single short sentence that says the reasoner backup is working."),
			("reviewer", DEFAULT_REVIEWER_MODEL, "Reply with a single short sentence that says the reviewer is working."),
			("reviewer-backup", DEFAULT_REVIEWER_MODEL_BACKUP, "Reply with a single short sentence that says the reviewer backup is working."),
		]
	selected_model = model.strip() or {
		"intent": DEFAULT_ROUTER_MODEL,
		"reasoning": DEFAULT_REASONER_MODEL,
		"review": DEFAULT_REVIEWER_MODEL,
	}[task_type]
	selected_prompt = DEFAULT_PROMPTS[task_type]
	return [(task_type, selected_model, selected_prompt)]


def main() -> int:
	parser = argparse.ArgumentParser(description="Run a quick Groq smoke test.")
	parser.add_argument(
		"--task-type",
		choices=["intent", "reasoning", "review"],
		default="reasoning",
		help="Which route to test when no explicit model is provided.",
	)
	parser.add_argument(
		"--all",
		action="store_true",
		help="Check all six configured model slots in one run.",
	)
	parser.add_argument(
		"--model",
		default="",
		help="Force a specific Groq model name instead of the routed model.",
	)
	parser.add_argument(
		"--prompt",
		default="",
		help="Prompt to send to Groq. If omitted, a route-specific smoke prompt is used.",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=64,
		help="Maximum number of tokens to request from Groq.",
	)
	args = parser.parse_args()

	_print_config_summary()
	probe_plan = _build_probe_plan(args.all, args.task_type, args.model)
	if args.all:
		print("Running six-model check...")
	else:
		print(f"Task type: {args.task_type}")

	any_failures = False
	any_configured = False
	for label, model_name, prompt in probe_plan:
		forced_model = model_name
		message = args.prompt.strip() or prompt
		print("")
		print(f"[{label}] model: {model_name} ({_resolve_model_label(model_name)})")
		result = call_groq(
			message,
			max_new_tokens=args.max_new_tokens,
			task_type=args.task_type,
			force_model=forced_model,
		)
		if result is None:
			print("  not configured")
			continue
		any_configured = True
		if result.startswith("ERROR:"):
			any_failures = True
			print(f"  failed: {result}")
		else:
			print(f"  ok: {result}")

	if not any_configured:
		print("")
		print("Groq is not configured. Set GROQ_API_KEY or GROQ_API_KEY_BACKUP in .env.")
		return 2
	return 1 if any_failures else 0


if __name__ == "__main__":
	raise SystemExit(main())
