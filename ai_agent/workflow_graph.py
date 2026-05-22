from __future__ import annotations

import json
from typing import Any, Dict, Optional, TypedDict

import pandas as pd

from ai_agent.workflow_tools import (
    build_report_payload,
    build_workflow_profile,
    perform_eda,
    recommend_model_setup,
)

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency during first rollout
    END = None
    START = None
    StateGraph = None


class WorkflowState(TypedDict, total=False):
    step: int
    df: pd.DataFrame
    summary: Any
    target_col: Optional[str]
    results: Dict[str, Any]
    results_tuned: Dict[str, Any]
    workflow_profile: Dict[str, Any]
    eda_recommendation: Dict[str, Any]
    model_recommendation: Dict[str, Any]
    report_payload: Dict[str, Any]


def build_workflow_graph() -> Optional[Any]:
    if StateGraph is None:
        return None

    graph: Any = StateGraph(WorkflowState)

    def _profile_node(state: WorkflowState) -> Dict[str, Any]:
        df = state.get("df")
        if df is None:
            return {}
        profile = build_workflow_profile(df, target_col=state.get("target_col"))
        return {"workflow_profile": json.loads(json.dumps(profile.__dict__, default=str))}

    def _eda_node(state: WorkflowState) -> Dict[str, Any]:
        df = state.get("df")
        if df is None:
            return {}
        return {"eda_recommendation": perform_eda(df, target_col=state.get("target_col"))}

    def _model_node(state: WorkflowState) -> Dict[str, Any]:
        df = state.get("df")
        if df is None:
            return {}
        return {"model_recommendation": recommend_model_setup(df, target_col=state.get("target_col"))}

    def _report_node(state: WorkflowState) -> Dict[str, Any]:
        return {
            "report_payload": build_report_payload(
                state.get("results"),
                state.get("results_tuned"),
                state.get("summary"),
            )
        }

    graph.add_node("profile", _profile_node)
    graph.add_node("eda", _eda_node)
    graph.add_node("model", _model_node)
    graph.add_node("report", _report_node)
    graph.set_entry_point("profile")
    graph.add_edge("profile", "eda")
    graph.add_edge("eda", "model")
    graph.add_edge("model", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_workflow_graph(state: WorkflowState) -> WorkflowState:
    compiled = build_workflow_graph()
    if compiled is None:
        fallback = dict(state)
        df = fallback.get("df")
        if isinstance(df, pd.DataFrame):
            fallback["workflow_profile"] = json.loads(json.dumps(build_workflow_profile(df, target_col=fallback.get("target_col")).__dict__, default=str))
            fallback["eda_recommendation"] = perform_eda(df, target_col=fallback.get("target_col"))
            fallback["model_recommendation"] = recommend_model_setup(df, target_col=fallback.get("target_col"))
        fallback["report_payload"] = build_report_payload(
            fallback.get("results"),
            fallback.get("results_tuned"),
            fallback.get("summary"),
        )
        return fallback
    result = compiled.invoke(state)
    return result if isinstance(result, dict) else dict(state)