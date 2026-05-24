from typing import Dict, Any, Tuple, List, Optional
import json
import pandas as pd
from ai_agent.data_utils import impute_missing_values, build_summary, count_iqr_outliers, remove_iqr_outliers
from ai_agent.llm_client import call_groq, groq_token_loaded, DEFAULT_REASONER_MODEL
from ai_agent.rag_store import build_workflow_rag_context
from ai_agent.copilot_utils import infer_time_series_configuration
from ai_agent.config import (
    STATISTICAL_CLASSIFICATION_MODELS,
    STATISTICAL_REGRESSION_MODELS,
    ML_CLASSIFICATION_MODELS,
    ML_REGRESSION_MODELS,
)


def _infer_tabular_problem_type(df: pd.DataFrame, state: Dict[str, Any]) -> str:
    if state.get("problem_type") in {"classification", "regression", "time_series"}:
        return state["problem_type"]
    summary = build_summary(df)
    if summary.datetime_cols:
        return "time_series"
    return "classification" if len(summary.categorical_cols) > 0 else "regression"


def _infer_eda_target(df: pd.DataFrame, state: Dict[str, Any], summary: Any) -> str:
    if state.get("eda_target_col") in list(df.columns):
        return state["eda_target_col"]
    if summary is None:
        return df.columns[-1]
    for col in getattr(summary, "categorical_cols", []):
        if not _is_useless_feature(col, df.dtypes.astype(str).get(col, "")):
            return col
    if summary.numeric_cols:
        return summary.numeric_cols[-1]
    return df.columns[-1]


def _pick_eda_target_pair(df: pd.DataFrame, target_col: str, summary: Any) -> Optional[str]:
    if target_col not in df.columns:
        return None

    dtype_map = df.dtypes.astype(str).to_dict()
    target_series = df[target_col]
    feature_cols = [c for c in df.columns if c != target_col]

    def usable(col: str) -> bool:
        return not _is_useless_feature(col, dtype_map.get(col, ""))

    if pd.api.types.is_numeric_dtype(target_series):
        numeric_candidates = [c for c in getattr(summary, "numeric_cols", []) if c in feature_cols and usable(c)]
        if numeric_candidates:
            corr_frame = df[[target_col] + numeric_candidates].apply(pd.to_numeric, errors="coerce")
            corr = corr_frame.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore").abs()
            if not corr.empty:
                return corr.sort_values(ascending=False).index[0]

    if getattr(summary, "numeric_cols", None):
        numeric_candidates = [c for c in summary.numeric_cols if c in feature_cols and usable(c)]
        if numeric_candidates and not pd.api.types.is_numeric_dtype(target_series):
            grouped = df[[target_col] + numeric_candidates].dropna()
            if not grouped.empty:
                best_col = None
                best_score = -1.0
                for col in numeric_candidates:
                    means = grouped.groupby(target_col)[col].mean()
                    if len(means) < 2:
                        continue
                    score = float(means.std())
                    if score > best_score:
                        best_score = score
                        best_col = col
                if best_col is not None:
                    return best_col

    for col in feature_cols:
        if usable(col):
            return col
    return feature_cols[0] if feature_cols else None


def _is_useless_feature(col: str, dtype_name: str) -> bool:
    name = str(col).lower()
    dtype_name = str(dtype_name).lower()
    tokens = ["date", "datetime", "time", "year", "month", "day", "yr", "mo", "dt", "built_year", "year_built"]
    if "datetime" in dtype_name or "datetimetz" in dtype_name:
        return True
    return any(token in name for token in tokens)


def _recommend_outlier_action(df: pd.DataFrame, outlier_counts: pd.DataFrame) -> Tuple[str, List[str], str]:
    if outlier_counts.empty:
        return "Keep outliers", [], "No outliers detected."

    rows = max(int(len(df)), 1)
    top_columns = outlier_counts.head(3)["column"].astype(str).tolist()
    remove_cols: List[str] = []
    measurement_tokens = ["cm", "mm", "width", "height", "length", "age", "weight", "price", "income", "salary", "amount", "score", "count"]

    for _, row in outlier_counts.iterrows():
        column = str(row["column"])
        count = int(row["outlier_count"])
        ratio = count / rows
        lower_name = column.lower()
        measurement_like = any(token in lower_name for token in measurement_tokens)
        if ratio >= 0.15 or (measurement_like and (ratio >= 0.05 or count >= 5)):
            remove_cols.append(column)

    if remove_cols:
        reason = f"Recommended removal for {', '.join(remove_cols)} because the outliers are concentrated in measurement-like columns."
        return "Remove selected outlier columns", remove_cols, reason

    reason = f"Recommended keeping {', '.join(top_columns)} because the outliers look sparse and may be valid variation."
    return "Keep outliers", top_columns, reason


def _normalize_request_text(text: str) -> str:
    return str(text or "").lower().replace("-", " ").replace("_", " ")


def _extract_explicit_model_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    txt = _normalize_request_text(text)
    compact = txt.replace(" ", "")
    tokens = txt.split()

    def has_decision_tree() -> bool:
        if "decision tree" in txt or "decisiontree" in compact or "dtree" in compact:
            return True
        decision_like = any(tok.startswith("deci") or tok in {"decison", "descision", "dcsn"} for tok in tokens)
        return decision_like and "tree" in tokens

    def has_random_forest() -> bool:
        if "random forest" in txt or "randomforest" in compact or "rf" in tokens:
            return True
        random_like = any(tok.startswith("rand") or tok in {"ranom", "rendom"} for tok in tokens)
        return random_like and "forest" in tokens

    # ML family
    if has_decision_tree():
        return "Decision Tree", "ML"
    if has_random_forest():
        return "Random Forest", "ML"
    if "gradient boosting" in txt or "gradientboosting" in compact or "xgboost" in compact:
        return "Gradient Boosting", "ML"
    if "knn" in txt or "k nearest" in txt or "nearest neighbor" in txt or "nearestneighbour" in compact:
        return "KNN", "ML"
    if " svm" in f" {txt}" or "support vector" in txt:
        return "SVM", "ML"
    if "svr" in txt:
        return "SVR", "ML"

    # Statistical family
    if "logistic" in txt:
        return "Logistic Regression", "Statistical"
    if "linear regression" in txt or "linearregression" in compact:
        return "Linear Regression", "Statistical"
    if "ridge" in txt:
        return "Ridge", "Statistical"
    if "lasso" in txt:
        return "Lasso", "Statistical"

    return None, None


def _extract_explicit_model(req_texts: List[str]) -> Tuple[Optional[str], Optional[str]]:
    # Latest request wins: evaluate in order and return the first recognized model.
    for text in req_texts:
        model, family = _extract_explicit_model_from_text(text)
        if model:
            return model, family
    return None, None


def apply_auto_actions_snapshot(state: Dict[str, Any], step: int, df: pd.DataFrame) -> Tuple[Dict[str, Any], List[str]]:
    """Compute a dict of session keys -> new values to apply as non-destructive auto actions.
    Returns (changes, activity_messages). Does NOT mutate the input state.
    """
    changes: Dict[str, Any] = {}
    activities: List[str] = []

    # If an LLM is available, ask the reasoner model for a suggested plan first.
    # The LLM-driven plan is treated as authoritative; rule-based defaults fill only missing keys.
    if groq_token_loaded():
        try:
            column_text = " ".join(map(str, list(df.columns)[:20])) if df is not None else ""
            summary_text = ""
            summary = None
            if df is not None:
                summary = build_summary(df)
                summary_text = (
                    f"rows={summary.rows} cols={summary.cols} "
                    f"numeric={len(summary.numeric_cols)} categorical={len(summary.categorical_cols)} "
                    f"datetime={len(summary.datetime_cols)}"
                )
            kb_context = build_workflow_rag_context(
                step, user_goal=str(state.get("agent_goal", "") or ""), column_text=column_text, summary_text=summary_text
            )
            # Build a concise auto-mode prompt for the reasoner
            base_focus = {"step": step, "goal": str(state.get("agent_goal", "") or "(none)"), "visible_keys": sorted([str(k) for k in state.keys()])[:40]}
            auto_prompt = (
                "You are the auto-mode decision engine for a fixed data-modelling workflow. "
                "Prioritize the user's request unless it conflicts with the dataset or workflow. "
                "Return JSON only using this schema: {\"changes\": {\"key\": value}, \"activities\": [\"short note\"], \"priority_source\": \"user|kb|dataset\"}. "
                "Only suggest supported session keys and never include control-flow keys.\n"
                f"Context: {json.dumps(base_focus, ensure_ascii=True)}\n"
                f"Knowledge base guidance:\n{kb_context or '(no kb context available)'}"
            )
            reply = call_groq(auto_prompt, max_new_tokens=220, task_type="reasoning", context=kb_context, force_model=DEFAULT_REASONER_MODEL)
            # If reasoner model fails or returns an error/empty, retry without forcing the reasoner
            if not reply or (isinstance(reply, str) and reply.startswith("ERROR:")):
                retry = call_groq(auto_prompt, max_new_tokens=220, task_type="reasoning", context=kb_context, force_model=None)
                if retry and not (isinstance(retry, str) and retry.startswith("ERROR:")):
                    reply = retry
            # parse JSON payload from reply
            def _parse_ai_plan(text: Optional[str]) -> Tuple[Dict[str, Any], List[str]]:
                if not text:
                    return {}, []
                t = str(text).strip()
                if t.startswith("```"):
                    t = t.strip("`")
                    if t.lower().startswith("json"):
                        t = t[4:].strip()
                s = t.find("{")
                e = t.rfind("}")
                if s < 0 or e < 0 or e <= s:
                    return {}, []
                try:
                    payload = json.loads(t[s : e + 1])
                except Exception:
                    return {}, []
                ch = payload.get("changes") or {}
                acts = payload.get("activities") or []
                if not isinstance(ch, dict):
                    ch = {}
                if not isinstance(acts, list):
                    acts = []
                clean_changes: Dict[str, Any] = {}
                for k, v in ch.items():
                    ks = str(k)
                    if ks.startswith("agent_"):
                        continue
                    if ks in {"step", "cleaning_confirmed", "analysis_ready"}:
                        continue
                    clean_changes[ks] = v
                clean_acts = [str(a) for a in acts if str(a).strip()]
                return clean_changes, clean_acts

            ai_changes, ai_activities = _parse_ai_plan(reply)
            # Explicit user requirements override the AI plan if they mention a concrete model.
            req_texts = []
            if state.get("agent_goal"):
                req_texts.append(str(state.get("agent_goal")))
            for r in state.get("agent_requirements", []) or []:
                try:
                    if isinstance(r, dict):
                        req_texts.append(str(r.get("text", "")))
                    else:
                        req_texts.append(str(r))
                except Exception:
                    continue
            explicit_model: Optional[str] = None
            explicit_family: Optional[str] = None
            explicit_time_series_changes: Dict[str, Any] = {}
            for text in req_texts:
                ts_changes = infer_time_series_configuration(text)
                if ts_changes:
                    explicit_time_series_changes = ts_changes
                    break
            has_time_series_request = bool(explicit_time_series_changes)
            explicit_model, explicit_family = _extract_explicit_model(req_texts)

            if explicit_model:
                ai_changes["model_name"] = explicit_model
                ai_changes["model_family"] = explicit_family or ai_changes.get("model_family", "ML")
                ai_changes.setdefault("proceed_model", True)
                ai_activities.append(f"Applied explicit user requirement: {explicit_model}.")
            if explicit_time_series_changes:
                ai_changes.update(explicit_time_series_changes)
                ai_changes.setdefault("proceed_model", True)
                ts_model = explicit_time_series_changes.get("time_series_model")
                ts_trend = explicit_time_series_changes.get("hw_trend")
                ts_seasonal = explicit_time_series_changes.get("hw_seasonal")
                ts_parts = [f"Applied time-series configuration: {ts_model or 'default'}"]
                if ts_trend:
                    ts_parts.append(f"trend={ts_trend}")
                if ts_seasonal:
                    ts_parts.append(f"seasonal={ts_seasonal}")
                if explicit_time_series_changes.get("hw_seasonal_periods"):
                    ts_parts.append(f"seasonal_periods={explicit_time_series_changes['hw_seasonal_periods']}")
                if explicit_time_series_changes.get("time_series_mode"):
                    ts_parts.append(f"mode={explicit_time_series_changes['time_series_mode']}")
                ai_activities.append("; ".join(ts_parts))

            # If no AI changes produced, enforce explicit requirements locally as fallback
            if not ai_changes:
                # fallback enforcement for common model requests
                if explicit_model:
                    ai_changes = {"model_family": explicit_family or "ML", "model_name": explicit_model, "proceed_model": True}
                    ai_activities = [f"Applied user requirement fallback: {explicit_model} model."]
                elif has_time_series_request:
                    ai_changes = {"problem_type": "time_series", "proceed_model": True}
                    ai_changes.update(explicit_time_series_changes)
                    ai_activities = ["Applied user requirement fallback: time-series configuration."]
            if ai_changes:
                activities.extend(ai_activities)
                activities.append("Auto mode used the KB and user requirements for AI-driven suggestions.")
                # Start with AI changes; rule-based defaults will only fill missing keys.
                # Enforce model_family alignment if AI suggested a model_name without family
                mname = ai_changes.get("model_name")
                if mname:
                    mname_raw = str(mname).strip()
                    mname_str = mname_raw
                    # Canonicalize common synonyms (ignore case/hyphens/spaces)
                    def _canon(name: str, candidates: List[str]) -> Optional[str]:
                        low = name.lower().replace("-", " ").replace("_", " ").strip()
                        for cand in candidates:
                            if cand.lower() == low:
                                return cand
                        for cand in candidates:
                            if cand.lower().replace(" ", "") == low.replace(" ", ""):
                                return cand
                        # fuzzy: token subset
                        for cand in candidates:
                            cand_low = cand.lower()
                            if low in cand_low or cand_low in low:
                                return cand
                        return None

                    known_stat = STATISTICAL_CLASSIFICATION_MODELS + STATISTICAL_REGRESSION_MODELS
                    known_ml = ML_CLASSIFICATION_MODELS + ML_REGRESSION_MODELS
                    canonical = _canon(mname_raw, known_stat + known_ml)
                    if canonical:
                        mname_str = canonical
                    else:
                        # try splitting camelcase/snuggled words
                        mname_compact = mname_raw.replace(" ", "").replace("-", "").replace("_", "").lower()
                        for cand in (known_stat + known_ml):
                            if cand.replace(" ", "").replace("-", "").replace("_", "").lower() == mname_compact:
                                mname_str = cand
                                break
                    ai_changes["model_name"] = mname_str
                    # Determine appropriate family
                    family = None
                    if mname_str in ML_CLASSIFICATION_MODELS or mname_str in ML_REGRESSION_MODELS:
                        family = "ML"
                    if mname_str in STATISTICAL_CLASSIFICATION_MODELS or mname_str in STATISTICAL_REGRESSION_MODELS:
                        family = "Statistical"
                    # If AI didn't include family, add it
                    if "model_family" not in ai_changes and family:
                        ai_changes["model_family"] = family
                        ai_activities.append(f"Set model_family to {family} to match suggested model {mname_str}.")
                    # If model_name isn't in known lists, warn and do not change family
                    known_models = set(STATISTICAL_CLASSIFICATION_MODELS + STATISTICAL_REGRESSION_MODELS + ML_CLASSIFICATION_MODELS + ML_REGRESSION_MODELS)
                    if mname_str not in known_models:
                        ai_activities.append(f"Warning: suggested model '{mname_str}' not in known model lists; application will attempt to apply it but may adjust family.")

                # Normalize potentially invalid model_family values emitted by LLM plans.
                if "model_family" in ai_changes:
                    raw_family = str(ai_changes.get("model_family") or "").strip().lower().replace("-", "_").replace(" ", "_")
                    if raw_family in {"ml", "machine_learning", "machinelearning"}:
                        ai_changes["model_family"] = "ML"
                    elif raw_family in {"statistical", "stats", "statistics"}:
                        ai_changes["model_family"] = "Statistical"
                    elif raw_family in {"decision_tree", "random_forest", "gradient_boosting", "svm", "svr", "knn"}:
                        ai_changes["model_family"] = "ML"
                        ai_activities.append("Normalized model_family to ML based on model token.")
                    elif raw_family in {"logistic_regression", "linear_regression", "ridge", "lasso"}:
                        ai_changes["model_family"] = "Statistical"
                        ai_activities.append("Normalized model_family to Statistical based on model token.")
                    else:
                        # If unknown family, infer from model_name when available.
                        if "model_name" in ai_changes:
                            m = str(ai_changes.get("model_name") or "")
                            if m in (ML_CLASSIFICATION_MODELS + ML_REGRESSION_MODELS):
                                ai_changes["model_family"] = "ML"
                            elif m in (STATISTICAL_CLASSIFICATION_MODELS + STATISTICAL_REGRESSION_MODELS):
                                ai_changes["model_family"] = "Statistical"
                changes.update(ai_changes)
        except Exception:
            # Fall back to rule-based behavior
            pass

    if step == 1:
        # suggest EDA toggles only if absent
        for key in [
            "eda_show_summary",
            "eda_show_corr",
            "eda_show_basic",
            "eda_show_missing",
            "eda_show_target_dist",
            "eda_show_target_rel",
            "eda_show_pairplot",
            "eda_show_outliers",
        ]:
            if key not in state and key not in changes:
                if key == "eda_show_pairplot":
                    changes[key] = bool(df is not None and hasattr(df, "columns") and len(df.columns) > 2)
                else:
                    changes[key] = True
        if changes:
            activities.append("Auto-selected EDA views and visuals for review.")
        if df is not None:
            summary = build_summary(df)
            target_col = _infer_eda_target(df, state, summary)
            if "eda_target_col" not in state and "eda_target_col" not in changes:
                changes.setdefault("eda_target_col", target_col)
            if "eda_show_target_dist" not in state and "eda_show_target_dist" not in changes:
                changes.setdefault("eda_show_target_dist", True)
            if "eda_show_target_rel" not in state and "eda_show_target_rel" not in changes:
                changes.setdefault("eda_show_target_rel", True)
            target_feature = _pick_eda_target_pair(df, target_col, summary)
            if target_feature is not None and "eda_target_feature" not in state and "eda_target_feature" not in changes:
                changes.setdefault("eda_target_feature", target_feature)
                activities.append(f"Auto-selected EDA target pair: {target_col} vs {target_feature}.")
            if summary.numeric_cols and len(summary.numeric_cols) >= 2 and "eda_numeric_column" not in state and "eda_numeric_column" not in changes:
                changes.setdefault("eda_numeric_column", summary.numeric_cols[0])
            if summary.categorical_cols and "eda_target_feature" not in changes and "eda_target_feature" not in state:
                changes.setdefault("eda_target_feature", summary.categorical_cols[0])
            if summary.datetime_cols and "eda_target_col" not in state and "eda_target_col" not in changes:
                changes.setdefault("eda_target_col", summary.datetime_cols[0] if state.get("eda_target_col") not in list(df.columns) else state.get("eda_target_col"))

        if df is not None:
            summary = build_summary(df)
            outlier_counts = count_iqr_outliers(df, summary.numeric_cols)
            if not outlier_counts.empty:
                top_outliers = outlier_counts.head(3)["column"].astype(str).tolist()
                recommendation, recommended_cols, reason = _recommend_outlier_action(df, outlier_counts)
                note = f"Outliers detected in: {', '.join(top_outliers)}. Recommended action: {recommendation.lower()}. {reason}"
                activities.append(note)
                changes.setdefault("cleaning_review_note", note)
                changes.setdefault("outlier_strategy", recommendation)
                changes.setdefault("outlier_remove_cols", recommended_cols if recommended_cols else top_outliers)
                changes.setdefault("outlier_decision_reason", reason)
                if recommendation == "Remove selected outlier columns" and recommended_cols:
                    cleaned_df, removed_rows = remove_iqr_outliers(df, recommended_cols)
                    changes["df_outlier_cleaned"] = cleaned_df
                    changes["summary_outlier_cleaned"] = build_summary(cleaned_df)
                    activities.append(f"Applied recommended outlier removal for {', '.join(recommended_cols)} ({removed_rows} rows removed).")
                else:
                    changes["df_outlier_cleaned"] = df.copy()
                    changes["summary_outlier_cleaned"] = summary

    elif step == 2:
        # suggest imputation preview
        if "missing_strategy" not in state and "missing_strategy" not in changes:
            changes["missing_strategy"] = "impute"
        if "numeric_impute_strategy" not in state and "numeric_impute_strategy" not in changes:
            changes["numeric_impute_strategy"] = "median"
        if "impute_categorical" not in state and "impute_categorical" not in changes:
            changes["impute_categorical"] = True
        base_df = state.get("df_outlier_cleaned") if state.get("df_outlier_cleaned") is not None else df
        if "df_cleaned" not in state and "df_cleaned" not in changes and base_df is not None:
            cleaned = impute_missing_values(base_df.copy(), numeric_strategy="median", impute_categorical=True)
            changes["df_cleaned"] = cleaned
            changes["summary_cleaned"] = build_summary(cleaned)
            changes["cleaning_applied"] = True
            missing_note = "Suggested median/mode imputation for missing values (preview)."
            prior_note = str(state.get("cleaning_review_note", "")).strip()
            if prior_note:
                changes["cleaning_review_note"] = f"{prior_note} {missing_note}"
            else:
                changes["cleaning_review_note"] = missing_note
            activities.append(missing_note)

    elif step == 3:
        # prepare default modeling choices
        if df is not None and len(df.columns) >= 2:
            summary = build_summary(df)
            problem_type = _infer_tabular_problem_type(df, state)
            # Aggregate requirement hints from recorded requirements and current goal
            req_texts = []
            for r in state.get("agent_requirements", []) or []:
                try:
                    if isinstance(r, dict):
                        req_texts.append(str(r.get("text", "")))
                    else:
                        req_texts.append(str(r))
                except Exception:
                    continue
            if state.get("agent_goal"):
                req_texts.append(str(state.get("agent_goal")))
            req_blob = " ".join(req_texts).lower()
            preferences = state.get("agent_preferences") or {}
            preferred_family = preferences.get("preferred_model_family")
            preferred_model_name = preferences.get("preferred_model_name")
            preferred_model_name_family = None
            if preferred_model_name in (ML_CLASSIFICATION_MODELS + ML_REGRESSION_MODELS):
                preferred_model_name_family = "ML"
            elif preferred_model_name in (STATISTICAL_CLASSIFICATION_MODELS + STATISTICAL_REGRESSION_MODELS):
                preferred_model_name_family = "Statistical"
            if "problem_type" not in state and "problem_type" not in changes:
                changes["problem_type"] = problem_type

            if "target_col" not in state and "target_col" not in changes:
                target_col = df.columns[-1]
                if problem_type == "time_series" and summary.numeric_cols:
                    target_col = summary.numeric_cols[-1]
                changes["target_col"] = target_col

            if problem_type == "time_series":
                if "time_col" not in state and "time_col" not in changes and summary.datetime_cols:
                    changes["time_col"] = summary.datetime_cols[0]
                if "time_series_model" not in state and "time_series_model" not in changes:
                    changes["time_series_model"] = "ARIMA"
                if "time_series_mode" not in state and "time_series_mode" not in changes:
                    changes["time_series_mode"] = "Auto ARIMA"
                activities.append("Prepared time-series defaults: time column, model, and Auto ARIMA mode.")
            else:
                if "model_family" not in state and "model_family" not in changes:
                    if preferred_family in {"Statistical", "ML"}:
                        changes["model_family"] = preferred_family
                        activities.append(f"Applied user preference for model family: {preferred_family}.")
                    else:
                        changes["model_family"] = "Statistical" if problem_type == "classification" and len(df.columns) <= 15 else "ML"
                if "model_name" not in state and "model_name" not in changes:
                    if preferred_model_name and (preferred_family is None or preferred_model_name_family == preferred_family):
                        changes["model_name"] = preferred_model_name
                        activities.append(f"Applied user preference for model name: {preferred_model_name}.")
                    # Honor explicit requirement for decision tree if requested by user
                    elif "decision tree" in req_blob or "decision-tree" in req_blob or ("decision" in req_blob and "tree" in req_blob):
                        changes["model_name"] = "Decision Tree"
                        changes.setdefault("model_family", "ML")
                        activities.append("Applied user requirement: Decision Tree model.")
                    else:
                        if problem_type == "classification":
                            changes["model_name"] = "Logistic Regression" if changes.get("model_family", state.get("model_family")) == "Statistical" else "Random Forest"
                        else:
                            changes["model_name"] = "Linear Regression" if changes.get("model_family", state.get("model_family")) == "Statistical" else "Random Forest"
                if "metric_label" not in state and "metric_label" not in changes:
                    changes["metric_label"] = "Accuracy" if problem_type == "classification" else "R2"
                if "proceed_model" not in state and "proceed_model" not in changes:
                    changes["proceed_model"] = True

                feature_options = [c for c in df.columns if c != changes.get("target_col", state.get("target_col"))]
                selected_features = []
                dtype_map = df[feature_options].dtypes.astype(str).to_dict() if feature_options else {}
                for col in feature_options:
                    if _is_useless_feature(col, dtype_map.get(col, "")):
                        if f"feature_{col}" not in state and f"feature_{col}" not in changes:
                            changes[f"feature_{col}"] = False
                    else:
                        selected_features.append(col)
                        if f"feature_{col}" not in state and f"feature_{col}" not in changes:
                            changes[f"feature_{col}"] = True
                if not state.get("selected_features") and "selected_features" not in changes:
                    changes["selected_features"] = selected_features
                if "feature_selection" not in state and "feature_selection" not in changes:
                    if changes.get("model_family", state.get("model_family")) == "Statistical":
                        changes["feature_selection"] = "Stepwise" if len(feature_options) > 1 else "None"
                    else:
                        changes["feature_selection"] = "None" if len(feature_options) <= 6 else "SelectKBest"
                if "confirm_features" not in state and "confirm_features" not in changes:
                    changes["confirm_features"] = True
                if "tuning_method" not in state and "tuning_method" not in changes:
                    changes["tuning_method"] = "Grid Search" if changes.get("model_family", state.get("model_family")) == "ML" else "None"
                if "max_rows_value" not in state and "max_rows_value" not in changes and len(df) > 0:
                    changes["max_rows_value"] = min(len(df), 1000)
                activities.append("Prepared model defaults, selected features, and enabled model setup.")

    return changes, activities
