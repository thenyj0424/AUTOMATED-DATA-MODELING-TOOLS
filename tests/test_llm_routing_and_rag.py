from ai_agent.llm_client import (
	DEFAULT_REASONER_MODEL,
	DEFAULT_REVIEWER_MODEL,
	DEFAULT_ROUTER_MODEL,
	call_groq,
	route_model,
	should_use_reviewer,
	retrieve_knowledge,
	format_knowledge_context,
)
from ai_agent.rag_store import build_workflow_rag_query, build_workflow_rag_context
from main import build_auto_mode_prompt


def test_route_model_uses_router_for_chat_like_tasks():
	model_name, mode = route_model(task_type="chat", prompt="Hello", context="")
	assert model_name == DEFAULT_ROUTER_MODEL
	assert mode == "router"


def test_route_model_uses_reasoner_for_planning():
	model_name, mode = route_model(task_type="reasoning", prompt="Design a pipeline", context="")
	assert model_name == DEFAULT_REASONER_MODEL
	assert mode == "reasoner"


def test_route_model_uses_reviewer_for_validation():
	model_name, mode = route_model(task_type="review", prompt="Validate edge cases", context="")
	assert model_name == DEFAULT_REVIEWER_MODEL
	assert mode == "reviewer"


def test_should_use_reviewer_flags_high_risk_text():
	assert should_use_reviewer("Please validate this high risk pipeline", "") is True


def test_retrieve_knowledge_returns_seed_chunk():
	results = retrieve_knowledge("routing model selection", top_k=2)
	assert results
	assert any("modeling_routing" in item["path"] or "routing" in item["title"].lower() for item in results)


def test_format_knowledge_context_contains_seed_text():
	text = format_knowledge_context("rag testing", top_k=2)
	assert "RAG Testing Note" in text or "Testing Note" in text


def test_retrieve_knowledge_prioritizes_reserved_chunks():
	results = retrieve_knowledge("evaluation metrics validation", top_k=3)
	assert results
	assert results[0]["title"] == "Model Evaluations"
	assert results[0]["score"] >= results[-1]["score"]


def test_workflow_query_targets_protocol_and_step_topic():
	step1_query = build_workflow_rag_query(1, user_goal="show plots")
	step2_query = build_workflow_rag_query(2, user_goal="clean missing values")
	step4_query = build_workflow_rag_query(4, user_goal="compare metrics")
	assert "System Adaptation Protocol" in step1_query
	assert "EDA Selection Logic" in step1_query
	assert "Data Cleaning Techniques" in step2_query
	assert "Model Evaluations" in step4_query


def test_auto_mode_prompt_prioritizes_user_request_and_kb():
	prompt = build_auto_mode_prompt(
		1,
		"show only missingness",
		{"df": None, "step": 1},
		"[EDA Selection Logic] example guidance",
	)
	assert "Prioritize the user's request" in prompt
	assert "Knowledge base guidance" in prompt
	assert "show only missingness" in prompt


def test_call_groq_falls_back_to_reviewer_on_rate_limit(monkeypatch):
	from ai_agent import llm_client

	class DummyResponse:
		def __init__(self, content):
			self.content = content

	class DummyGroq:
		def __init__(self, api_key, model, temperature, max_tokens):
			self.model = model
		def invoke(self, messages):
			if self.model == DEFAULT_REASONER_MODEL:
				raise RuntimeError("429 rate limit exceeded")
			if self.model == DEFAULT_REVIEWER_MODEL:
				return DummyResponse("backup ok")
			raise RuntimeError("model unavailable")

	monkeypatch.setenv("GROQ_API_KEY", "test-key")
	monkeypatch.setattr(llm_client, "ChatGroq", DummyGroq)
	result = call_groq("test prompt", task_type="reasoning", force_model=DEFAULT_REASONER_MODEL)
	assert result == "backup ok"


def test_build_workflow_rag_context_uses_step_aware_top_k(monkeypatch):
	from ai_agent import rag_store

	calls = []

	def _fake_format(query, top_k=3):
		calls.append(top_k)
		return "context"

	monkeypatch.setattr(rag_store, "format_knowledge_context", _fake_format)

	build_workflow_rag_context(1, user_goal="eda")
	build_workflow_rag_context(3, user_goal="model")

	assert calls[0] == 2
	assert calls[1] == 3