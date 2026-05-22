from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from ai_agent.data_utils import impute_missing_values, build_summary, count_iqr_outliers


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


def apply_auto_actions_snapshot(state: Dict[str, Any], step: int, df: pd.DataFrame) -> Tuple[Dict[str, Any], List[str]]:
    """Compute a dict of session keys -> new values to apply as non-destructive auto actions.
    Returns (changes, activity_messages). Does NOT mutate the input state.
    """
    changes: Dict[str, Any] = {}
    activities: List[str] = []

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
            if key not in state:
                if key == "eda_show_pairplot":
                    changes[key] = bool(df is not None and hasattr(df, "columns") and len(df.columns) > 2)
                else:
                    changes[key] = True
        if changes:
            activities.append("Auto-selected EDA views and visuals for review.")
        if df is not None:
            summary = build_summary(df)
            target_col = _infer_eda_target(df, state, summary)
            changes.setdefault("eda_target_col", target_col)
            changes.setdefault("eda_show_target_dist", True)
            changes.setdefault("eda_show_target_rel", True)
            target_feature = _pick_eda_target_pair(df, target_col, summary)
            if target_feature is not None:
                changes.setdefault("eda_target_feature", target_feature)
                activities.append(f"Auto-selected EDA target pair: {target_col} vs {target_feature}.")
            if summary.numeric_cols and len(summary.numeric_cols) >= 2:
                changes.setdefault("eda_numeric_column", summary.numeric_cols[0])
            if summary.categorical_cols and "eda_target_feature" not in changes and "eda_target_feature" not in state:
                changes.setdefault("eda_target_feature", summary.categorical_cols[0])
            if summary.datetime_cols:
                changes.setdefault("eda_target_col", summary.datetime_cols[0] if state.get("eda_target_col") not in list(df.columns) else state.get("eda_target_col"))

        if df is not None:
            summary = build_summary(df)
            outlier_counts = count_iqr_outliers(df, summary.numeric_cols)
            if not outlier_counts.empty:
                top_outliers = outlier_counts.head(3)["column"].tolist()
                activities.append(
                    "Outliers detected in: " + ", ".join(top_outliers) + ". Review the outlier summary before deciding whether to remove them."
                )

    elif step == 2:
        # suggest imputation preview
        if "missing_strategy" not in state:
            changes["missing_strategy"] = "impute"
        if "numeric_impute_strategy" not in state:
            changes["numeric_impute_strategy"] = "median"
        if "impute_categorical" not in state:
            changes["impute_categorical"] = True
        if "df_cleaned" not in state and df is not None:
            cleaned = impute_missing_values(df.copy(), numeric_strategy="median", impute_categorical=True)
            changes["df_cleaned"] = cleaned
            changes["summary_cleaned"] = build_summary(cleaned)
            changes["cleaning_applied"] = True
            activities.append("Suggested median/mode imputation for missing values (preview).")

    elif step == 3:
        # prepare default modeling choices
        if df is not None and len(df.columns) >= 2:
            summary = build_summary(df)
            problem_type = _infer_tabular_problem_type(df, state)
            preferences = state.get("agent_preferences") or {}
            preferred_family = preferences.get("preferred_model_family")
            if "problem_type" not in state:
                changes["problem_type"] = problem_type

            if "target_col" not in state:
                target_col = df.columns[-1]
                if problem_type == "time_series" and summary.numeric_cols:
                    target_col = summary.numeric_cols[-1]
                changes["target_col"] = target_col

            if problem_type == "time_series":
                if "time_col" not in state and summary.datetime_cols:
                    changes["time_col"] = summary.datetime_cols[0]
                if "time_series_model" not in state:
                    changes["time_series_model"] = "ARIMA"
                if "time_series_mode" not in state:
                    changes["time_series_mode"] = "Auto ARIMA"
                activities.append("Prepared time-series defaults: time column, model, and Auto ARIMA mode.")
            else:
                if "model_family" not in state:
                    if preferred_family in {"Statistical", "ML"}:
                        changes["model_family"] = preferred_family
                        activities.append(f"Applied user preference for model family: {preferred_family}.")
                    else:
                        changes["model_family"] = "Statistical" if problem_type == "classification" and len(df.columns) <= 15 else "ML"
                if "model_name" not in state:
                    if problem_type == "classification":
                        changes["model_name"] = "Logistic Regression" if changes.get("model_family", state.get("model_family")) == "Statistical" else "Random Forest"
                    else:
                        changes["model_name"] = "Linear Regression" if changes.get("model_family", state.get("model_family")) == "Statistical" else "Random Forest"
                if "metric_label" not in state:
                    changes["metric_label"] = "Accuracy" if problem_type == "classification" else "R2"
                if "proceed_model" not in state:
                    changes["proceed_model"] = True

                feature_options = [c for c in df.columns if c != changes.get("target_col", state.get("target_col"))]
                selected_features = []
                dtype_map = df[feature_options].dtypes.astype(str).to_dict() if feature_options else {}
                for col in feature_options:
                    if _is_useless_feature(col, dtype_map.get(col, "")):
                        changes[f"feature_{col}"] = False
                    else:
                        selected_features.append(col)
                        if f"feature_{col}" not in state:
                            changes[f"feature_{col}"] = True
                if not state.get("selected_features"):
                    changes["selected_features"] = selected_features
                if "feature_selection" not in state:
                    if changes.get("model_family", state.get("model_family")) == "Statistical":
                        changes["feature_selection"] = "Stepwise" if len(feature_options) > 1 else "None"
                    else:
                        changes["feature_selection"] = "None" if len(feature_options) <= 6 else "SelectKBest"
                if "confirm_features" not in state:
                    changes["confirm_features"] = True
                if "tuning_method" not in state:
                    changes["tuning_method"] = "Grid Search" if changes.get("model_family", state.get("model_family")) == "ML" else "None"
                if "max_rows_value" not in state and len(df) > 0:
                    changes["max_rows_value"] = min(len(df), 1000)
                activities.append("Prepared model defaults, selected features, and enabled model setup.")

    return changes, activities
