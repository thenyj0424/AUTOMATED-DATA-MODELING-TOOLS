from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool
from statsmodels.tsa.stattools import adfuller

from ai_agent.data_utils import build_summary


@dataclass
class WorkflowProfile:
    rows: int
    cols: int
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]
    missing_top: List[Dict[str, Any]]
    problem_type: str
    target_col: Optional[str]


def build_workflow_profile(df: pd.DataFrame, target_col: Optional[str] = None) -> WorkflowProfile:
    summary = build_summary(df)
    problem_type = "time_series" if summary.datetime_cols else ("classification" if summary.categorical_cols else "regression")
    missing_top = (
        summary.missing_by_col.sort_values("missing_count", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )
    return WorkflowProfile(
        rows=summary.rows,
        cols=summary.cols,
        numeric_cols=summary.numeric_cols,
        categorical_cols=summary.categorical_cols,
        datetime_cols=summary.datetime_cols,
        missing_top=missing_top,
        problem_type=problem_type,
        target_col=target_col,
    )


def _infer_target_problem_type(df: pd.DataFrame, target_col: Optional[str], default_problem_type: str) -> str:
    if default_problem_type == "time_series":
        return "time_series"
    if not target_col or target_col not in df.columns:
        return default_problem_type
    series = df[target_col]
    if pd.api.types.is_datetime64_any_dtype(series):
        return "time_series"
    if pd.api.types.is_numeric_dtype(series):
        unique_count = int(series.dropna().nunique())
        if unique_count <= 10:
            return "classification"
        return "regression"
    return "classification"


def summarize_missingness(df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
    missing = (
        df.isna().sum().reset_index(name="missing_count")
        .rename(columns={"index": "column"})
        .sort_values("missing_count", ascending=False)
    )
    total_rows = max(1, len(df))
    top_rows = missing.head(top_k).to_dict(orient="records")
    return {
        "top_missing": top_rows,
        "columns_with_missing": int((missing["missing_count"] > 0).sum()),
        "missing_ratio_by_column": [
            {"column": row["column"], "missing_ratio": round(row["missing_count"] / total_rows, 4)}
            for row in top_rows
        ],
    }


def analyze_stationarity(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    if not target_col or target_col not in df.columns:
        return {
            "is_applicable": False,
            "reason": "No valid target column supplied.",
            "is_stationary": None,
        }
    series = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if len(series) < 12:
        return {
            "is_applicable": False,
            "reason": "Not enough numeric observations for a stationarity test.",
            "is_stationary": None,
        }
    try:
        statistic, pvalue, _, _, critical_values, _ = adfuller(series, autolag="AIC")
        is_stationary = bool(pvalue <= 0.05)
        return {
            "is_applicable": True,
            "test": "ADF",
            "statistic": round(float(statistic), 6),
            "pvalue": round(float(pvalue), 6),
            "critical_values": {k: round(float(v), 6) for k, v in critical_values.items()},
            "is_stationary": is_stationary,
            "recommendation": "No differencing needed" if is_stationary else "Consider differencing or trend removal",
        }
    except Exception as exc:
        return {
            "is_applicable": False,
            "reason": f"Stationarity test failed: {exc}",
            "is_stationary": None,
        }


def recommend_eda_actions(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    profile = build_workflow_profile(df, target_col=target_col)
    problem_type = _infer_target_problem_type(df, target_col, profile.problem_type)
    visible_sections: List[str] = ["summary", "missingness"]
    if profile.numeric_cols:
        if len(profile.numeric_cols) >= 2:
            visible_sections.append("correlation")
        visible_sections.append("basic_plots")
        visible_sections.append("outlier_summary")
    if profile.categorical_cols:
        visible_sections.append("target_distribution")
    if target_col:
        visible_sections.append("target_relationships")
    if 2 <= len(profile.numeric_cols) <= 6 and df.shape[1] <= 20 and len(df) <= 1000:
        visible_sections.append("pairplot")
    if problem_type == "time_series":
        visible_sections.extend(["time_series_line", "acf_pacf", "stationarity"])
    visible_sections = list(dict.fromkeys(visible_sections))
    return {
        "problem_type": problem_type,
        "target_col": target_col,
        "recommended_sections": visible_sections,
        "missing_top": profile.missing_top,
        "stationarity": analyze_stationarity(df, target_col=target_col),
        "missingness": summarize_missingness(df),
    }


def perform_eda(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    profile = build_workflow_profile(df, target_col=target_col)
    eda_plan = recommend_eda_actions(df, target_col=target_col)
    return {
        "profile": {
            "rows": profile.rows,
            "cols": profile.cols,
            "numeric_cols": profile.numeric_cols,
            "categorical_cols": profile.categorical_cols,
            "datetime_cols": profile.datetime_cols,
            "problem_type": profile.problem_type,
            "target_col": profile.target_col,
        },
        "missingness": summarize_missingness(df),
        "stationarity": eda_plan["stationarity"],
        "recommended_sections": eda_plan["recommended_sections"],
        "problem_type": eda_plan["problem_type"],
        "target_col": target_col,
    }


def recommend_model_setup(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    profile = build_workflow_profile(df, target_col=target_col)
    problem_type = _infer_target_problem_type(df, target_col, profile.problem_type)
    if problem_type == "time_series":
        return {
            "problem_type": "time_series",
            "model_family": "Statistical",
            "model_name": "ARIMA",
            "reason": "Datetime columns suggest a time-series workflow.",
        }

    if problem_type == "classification":
        model_family = "Statistical" if len(profile.categorical_cols) <= 2 else "ML"
        model_name = "Logistic Regression" if model_family == "Statistical" else "Decision Tree"
    else:
        model_family = "Statistical" if len(profile.numeric_cols) <= 5 else "ML"
        model_name = "Linear Regression" if model_family == "Statistical" else "Random Forest"

    return {
        "problem_type": problem_type,
        "model_family": model_family,
        "model_name": model_name,
        "reason": "Selected from dataset shape, target type, and supported model families.",
    }


def build_report_payload(
    results: Optional[Dict[str, Any]],
    results_tuned: Optional[Dict[str, Any]],
    summary: Any,
) -> Dict[str, Any]:
    results = results or {}
    results_tuned = results_tuned or {}
    baseline = results.get("baseline", {}) if isinstance(results, dict) else {}
    metrics = baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
    tuned_baseline = results_tuned.get("baseline", {}) if isinstance(results_tuned, dict) else {}
    tuned_metrics = tuned_baseline.get("metrics", {}) if isinstance(tuned_baseline, dict) else {}
    insights = results.get("baseline_insights", {}) if isinstance(results, dict) else {}
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "problem_type": results.get("problem_type"),
        "model_name": results.get("model_name"),
        "metric_label": results.get("metric_label"),
        "baseline_metrics": metrics,
        "tuned_metrics": tuned_metrics,
        "selected_features": results.get("selected_features", []),
        "top_drivers": insights.get("feature_importances", []) or insights.get("coefficients", []),
        "summary": {
            "rows": getattr(summary, "rows", None),
            "cols": getattr(summary, "cols", None),
            "numeric_cols": getattr(summary, "numeric_cols", []),
            "categorical_cols": getattr(summary, "categorical_cols", []),
            "datetime_cols": getattr(summary, "datetime_cols", []),
        },
    }
@tool("workflow_profile")
def workflow_profile_tool(profile_json: str) -> str:
    """Return a workflow profile payload in compact JSON."""
    return profile_json




@tool("perform_eda")
def perform_eda_tool(profile_json: str) -> str:
    """Return an EDA action plan in compact JSON."""
    return profile_json


@tool("stationarity_analysis")
def stationarity_analysis_tool(profile_json: str) -> str:
    """Return a stationarity analysis payload in compact JSON."""
    return profile_json
@tool("eda_recommendation")
def eda_recommendation_tool(profile_json: str) -> str:
    """Return EDA recommendations in compact JSON."""
    return profile_json


@tool("model_setup_recommendation")
def model_setup_recommendation_tool(profile_json: str) -> str:
    """Return model setup recommendations in compact JSON."""
    return profile_json


@tool("report_payload")
def report_payload_tool(report_json: str) -> str:
    """Return a prepared report payload in compact JSON."""
    return report_json