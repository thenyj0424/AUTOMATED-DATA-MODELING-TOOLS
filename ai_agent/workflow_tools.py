from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy.stats import jarque_bera, shapiro, spearmanr
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


def analyze_correlations(df: pd.DataFrame, top_k: int = 8) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        return {
            "is_applicable": False,
            "reason": "Need at least two numeric columns for correlation analysis.",
            "top_pairs": [],
            "numeric_cols": numeric_cols,
        }
    corr = df[numeric_cols].corr(numeric_only=True).abs()
    pairs: List[Dict[str, Any]] = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            pairs.append({"feature_a": col_a, "feature_b": col_b, "abs_corr": round(float(value), 6)})
    pairs.sort(key=lambda item: item["abs_corr"], reverse=True)
    return {
        "is_applicable": True,
        "numeric_cols": numeric_cols,
        "top_pairs": pairs[:max(1, top_k)],
        "strong_pairs": [item for item in pairs if item["abs_corr"] >= 0.7][:max(1, top_k)],
    }


def analyze_normality(df: pd.DataFrame, max_columns: int = 8) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()[:max(1, max_columns)]
    if not numeric_cols:
        return {
            "is_applicable": False,
            "reason": "No numeric columns available for normality testing.",
            "results": [],
        }
    results: List[Dict[str, Any]] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 8:
            results.append({
                "column": col,
                "is_applicable": False,
                "reason": "Not enough observations for normality tests.",
            })
            continue
        sample = series.sample(n=min(5000, len(series)), random_state=42) if len(series) > 5000 else series
        try:
            shapiro_stat, shapiro_p = shapiro(sample)
            jb_stat, jb_p = jarque_bera(sample)
            results.append(
                {
                    "column": col,
                    "is_applicable": True,
                    "shapiro_pvalue": round(float(shapiro_p), 6),
                    "jarque_bera_pvalue": round(float(jb_p), 6),
                    "looks_normal": bool(shapiro_p > 0.05 and jb_p > 0.05),
                }
            )
        except Exception as exc:
            results.append({
                "column": col,
                "is_applicable": False,
                "reason": f"Normality test failed: {exc}",
            })
    return {
        "is_applicable": True,
        "tested_columns": numeric_cols,
        "results": results,
    }


def analyze_multicollinearity(df: pd.DataFrame, max_features: int = 12) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()[:max(2, max_features)]
    if len(numeric_cols) < 2:
        return {
            "is_applicable": False,
            "reason": "Need at least two numeric features for VIF analysis.",
            "vif": [],
        }
    frame = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < max(8, len(numeric_cols) + 1):
        return {
            "is_applicable": False,
            "reason": "Not enough complete rows for VIF analysis.",
            "vif": [],
        }
    try:
        vif_rows: List[Dict[str, Any]] = []
        values = frame.values.astype(float)
        for idx, col in enumerate(frame.columns):
            y = values[:, idx]
            X_other = np.delete(values, idx, axis=1)
            if X_other.shape[1] == 0:
                continue
            X_design = np.column_stack([np.ones(len(X_other)), X_other])
            coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            y_hat = X_design @ coeffs
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot <= 0:
                vif_value = float("inf")
            else:
                r2 = max(0.0, min(0.999999, 1.0 - (ss_res / ss_tot)))
                vif_value = 1.0 / (1.0 - r2)
            vif_rows.append({"feature": col, "vif": round(float(vif_value), 6), "high_vif": bool(vif_value >= 10.0)})
        vif_rows.sort(key=lambda item: item["vif"], reverse=True)
        return {
            "is_applicable": True,
            "vif": vif_rows,
            "high_vif_features": [row["feature"] for row in vif_rows if row["high_vif"]],
        }
    except Exception as exc:
        return {
            "is_applicable": False,
            "reason": f"VIF analysis failed: {exc}",
            "vif": [],
        }


def analyze_heteroscedasticity(df: pd.DataFrame, target_col: Optional[str] = None, max_features: int = 8) -> Dict[str, Any]:
    if not target_col or target_col not in df.columns:
        return {
            "is_applicable": False,
            "reason": "No valid numeric target supplied for heteroscedasticity testing.",
            "pvalue": None,
        }
    target_series = pd.to_numeric(df[target_col], errors="coerce")
    if not pd.api.types.is_numeric_dtype(target_series):
        return {
            "is_applicable": False,
            "reason": "Target must be numeric for heteroscedasticity testing.",
            "pvalue": None,
        }
    candidate_features = [
        col for col in df.select_dtypes(include=["number"]).columns.tolist()
        if col != target_col
    ][:max(1, max_features)]
    if not candidate_features:
        return {
            "is_applicable": False,
            "reason": "Need at least one numeric predictor for heteroscedasticity testing.",
            "pvalue": None,
        }
    frame = pd.DataFrame({"target": target_series, **{c: pd.to_numeric(df[c], errors="coerce") for c in candidate_features}}).dropna()
    if len(frame) < max(20, len(candidate_features) + 5):
        return {
            "is_applicable": False,
            "reason": "Not enough complete rows for heteroscedasticity testing.",
            "pvalue": None,
        }
    try:
        y = frame["target"].to_numpy(dtype=float)
        X = frame[candidate_features].to_numpy(dtype=float)
        X_design = np.column_stack([np.ones(len(X)), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        y_hat = X_design @ coeffs
        residual_abs = np.abs(y - y_hat)
        statistic, pvalue = spearmanr(y_hat, residual_abs)
        pvalue = float(pvalue) if pvalue is not None and not np.isnan(pvalue) else 1.0
        return {
            "is_applicable": True,
            "test": "Spearman(abs_residual, fitted)",
            "pvalue": round(pvalue, 6),
            "is_heteroscedastic": bool(pvalue <= 0.05),
            "recommendation": "Use robust errors or transform features" if pvalue <= 0.05 else "No strong heteroscedasticity signal",
        }
    except Exception as exc:
        return {
            "is_applicable": False,
            "reason": f"Heteroscedasticity test failed: {exc}",
            "pvalue": None,
        }


def should_run_statistical_tools(user_text: str) -> bool:
    text = str(user_text or "").lower()
    if not text:
        return False
    tokens = [
        "stationarity",
        "adf",
        "correlation",
        "normality",
        "shapiro",
        "jarque",
        "vif",
        "multicollinearity",
        "collinearity",
        "heteroscedastic",
        "breusch",
        "statistical test",
        "hypothesis test",
    ]
    return any(token in text for token in tokens)


def run_statistical_tools_on_demand(df: pd.DataFrame, user_text: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    if not should_run_statistical_tools(user_text):
        return {}
    return {
        "stationarity_analysis": analyze_stationarity(df, target_col=target_col),
        "correlation_analysis": analyze_correlations(df),
        "normality_analysis": analyze_normality(df),
        "multicollinearity_analysis": analyze_multicollinearity(df),
        "heteroscedasticity_analysis": analyze_heteroscedasticity(df, target_col=target_col),
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
        "correlation_analysis": analyze_correlations(df),
        "recommended_sections": eda_plan["recommended_sections"],
        "problem_type": eda_plan["problem_type"],
        "target_col": target_col,
    }


def collect_dataset_diagnostics(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    return {
        "eda": perform_eda(df, target_col=target_col),
        "stationarity": analyze_stationarity(df, target_col=target_col),
        "correlation_analysis": analyze_correlations(df),
        "model_setup": recommend_model_setup(df, target_col=target_col),
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


@tool("correlation_analysis")
def correlation_analysis_tool(profile_json: str) -> str:
    """Return a correlation analysis payload in compact JSON."""
    return profile_json


@tool("normality_analysis")
def normality_analysis_tool(profile_json: str) -> str:
    """Return a normality analysis payload in compact JSON."""
    return profile_json


@tool("multicollinearity_analysis")
def multicollinearity_analysis_tool(profile_json: str) -> str:
    """Return a multicollinearity (VIF) analysis payload in compact JSON."""
    return profile_json


@tool("heteroscedasticity_analysis")
def heteroscedasticity_analysis_tool(profile_json: str) -> str:
    """Return a heteroscedasticity analysis payload in compact JSON."""
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