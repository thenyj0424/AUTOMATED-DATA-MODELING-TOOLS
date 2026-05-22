from itertools import product
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
	roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import (
	GradientBoostingClassifier,
	GradientBoostingRegressor,
	RandomForestClassifier,
	RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import (
	SelectKBest,
	f_classif,
	f_regression,
	SelectFromModel,
	SequentialFeatureSelector,
	RFE,
)


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
	numeric_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)
	return ColumnTransformer(
		transformers=[
			("num", numeric_pipe, numeric_cols),
			("cat", categorical_pipe, categorical_cols),
		]
	)


def get_model(problem_type: str, model_name: str) -> Any:
	if problem_type == "classification":
		if model_name == "Logistic Regression":
			return LogisticRegression(max_iter=1000)
		if model_name == "Decision Tree":
			return DecisionTreeClassifier(random_state=42)
		if model_name == "Random Forest":
			return RandomForestClassifier(n_estimators=200, random_state=42)
		if model_name == "Gradient Boosting":
			return GradientBoostingClassifier(random_state=42)
		if model_name == "SVM":
			return SVC(probability=True)
		return KNeighborsClassifier()

	if model_name == "Linear Regression":
		return LinearRegression()
	if model_name == "Ridge":
		return Ridge(alpha=1.0)
	if model_name == "Lasso":
		return Lasso(alpha=0.01)
	if model_name == "Decision Tree":
		return DecisionTreeRegressor(random_state=42)
	if model_name == "Random Forest":
		return RandomForestRegressor(n_estimators=200, random_state=42)
	if model_name == "Gradient Boosting":
		return GradientBoostingRegressor(random_state=42)
	if model_name == "SVR":
		return SVR()
	return KNeighborsRegressor()


def get_param_grid(problem_type: str, model_name: str) -> Dict[str, Any]:
	if problem_type == "classification":
		if model_name == "Logistic Regression":
			return {"model__C": [0.1, 1.0, 10.0]}
		if model_name == "Decision Tree":
			return {
				"model__max_depth": [None, 3, 5, 10],
				"model__min_samples_split": [2, 5, 10],
			}
		if model_name == "Random Forest":
			return {
				"model__n_estimators": [100, 200],
				"model__max_depth": [None, 5, 10],
			}
		if model_name == "Gradient Boosting":
			return {
				"model__n_estimators": [100, 200],
				"model__learning_rate": [0.05, 0.1],
				"model__max_depth": [3, 5],
			}
		if model_name == "SVM":
			return {
				"model__C": [0.1, 1.0, 10.0],
				"model__kernel": ["rbf", "linear"],
			}
		return {"model__n_neighbors": [3, 5, 7, 9]}

	if model_name == "Linear Regression":
		return {}
	if model_name == "Ridge":
		return {"model__alpha": [0.1, 1.0, 10.0]}
	if model_name == "Lasso":
		return {"model__alpha": [0.001, 0.01, 0.1]}
	if model_name == "Decision Tree":
		return {
			"model__max_depth": [None, 3, 5, 10],
			"model__min_samples_split": [2, 5, 10],
		}
	if model_name == "Random Forest":
		return {
			"model__n_estimators": [100, 200],
			"model__max_depth": [None, 5, 10],
		}
	if model_name == "Gradient Boosting":
		return {
			"model__n_estimators": [100, 200],
			"model__learning_rate": [0.05, 0.1],
			"model__max_depth": [3, 5],
		}
	if model_name == "SVR":
		return {
			"model__C": [0.1, 1.0, 10.0],
			"model__epsilon": [0.05, 0.1, 0.2],
			"model__kernel": ["rbf", "linear"],
		}
	return {"model__n_neighbors": [3, 5, 7, 9]}


def apply_target_missing(
	df: pd.DataFrame,
	target_col: str,
	problem_type: str,
	target_missing: str,
) -> Tuple[pd.DataFrame, Optional[str]]:
	work_df = df.copy()
	if target_missing == "drop":
		work_df = work_df.dropna(subset=[target_col])
	elif target_missing == "impute":
		if problem_type == "classification":
			fill_val = work_df[target_col].mode(dropna=True)
			fill_val = fill_val.iloc[0] if not fill_val.empty else None
			work_df[target_col] = work_df[target_col].fillna(fill_val)
		else:
			median_val = work_df[target_col].median()
			work_df[target_col] = work_df[target_col].fillna(median_val)
	elif work_df[target_col].isna().any():
		return work_df, "Target has missing values. Choose drop or resolve missing values."
	return work_df, None


def estimate_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> List[str]:
	prep = clone(preprocessor)
	prep.fit(X)
	try:
		names = prep.get_feature_names_out()
		return [str(name) for name in names]
	except Exception:
		Xt = prep.transform(X)
		return [f"feature_{idx}" for idx in range(Xt.shape[1])]


def build_selector(
	method: str,
	problem_type: str,
	model_based: str,
	stepwise_direction: str,
	max_features: int,
	feature_names: List[str],
) -> Any:
	if method == "None":
		return "passthrough"

	feature_count = len(feature_names)
	if feature_count <= 1:
		return "passthrough"
	k = min(max_features, feature_count - 1)
	if method == "Stepwise":
		if problem_type == "classification":
			estimator = LogisticRegression(max_iter=1000)
		else:
			estimator = LinearRegression()
		return SequentialFeatureSelector(
			estimator,
			n_features_to_select=k,
			direction=stepwise_direction,
		)

	if method == "RFE":
		if problem_type == "classification":
			estimator = LogisticRegression(max_iter=1000)
		else:
			estimator = LinearRegression()
		return RFE(estimator=estimator, n_features_to_select=k)

	if method == "SelectKBest":
		score_func = f_classif if problem_type == "classification" else f_regression
		return SelectKBest(score_func=score_func, k=k)

	if method == "Model-based":
		if model_based == "L1/Lasso":
			if problem_type == "classification":
				estimator = LogisticRegression(penalty="l1", solver="liblinear")
			else:
				estimator = Lasso(alpha=0.001)
		else:
			if problem_type == "classification":
				estimator = RandomForestClassifier(n_estimators=100, random_state=42)
			else:
				estimator = RandomForestRegressor(n_estimators=100, random_state=42)
		return SelectFromModel(estimator, max_features=max_features)

	return "passthrough"


def build_pipeline(
	preprocessor: ColumnTransformer,
	selector: Any,
	model: Any,
) -> Pipeline:
	return Pipeline(
		steps=[
			("prep", preprocessor),
			("selector", selector),
			("model", model),
		]
	)


def get_selected_feature_names(
	pipeline: Pipeline,
	feature_names: List[str],
) -> List[str]:
	selector = pipeline.named_steps.get("selector")
	if selector is None or selector == "passthrough":
		return feature_names
	if hasattr(selector, "get_support"):
		mask = selector.get_support()
		return [name for name, keep in zip(feature_names, mask) if keep]
	return feature_names


def evaluate_predictions(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	problem_type: str,
) -> Dict[str, float]:
	if problem_type == "classification":
		return {
			"accuracy": float(accuracy_score(y_true, y_pred)),
			"f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
		}
	return {
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
		"r2": float(r2_score(y_true, y_pred)),
	}


def evaluate_roc_auc(model: Any, X: pd.DataFrame, y: pd.Series) -> Optional[float]:
	try:
		if hasattr(model, "predict_proba"):
			proba = model.predict_proba(X)
			if proba.ndim == 2 and proba.shape[1] == 2:
				return float(roc_auc_score(y, proba[:, 1]))
			return float(roc_auc_score(y, proba, multi_class="ovr"))
		if hasattr(model, "decision_function"):
			dec = model.decision_function(X)
			return float(roc_auc_score(y, dec, multi_class="ovr"))
	except Exception:
		return None
	return None


def _auto_select_time_series_model(
	train: pd.Series,
	model_name: str,
	seasonal_periods: int,
) -> Tuple[Any, Dict[str, Any]]:
	from statsmodels.tsa.arima.model import ARIMA
	from statsmodels.tsa.statespace.sarimax import SARIMAX

	if model_name == "ARIMA":
		candidate_orders = list(product(range(0, 4), range(0, 3), range(0, 4)))
		best_result = None
		best_order = None
		best_aic = None
		for order in candidate_orders:
			try:
				result = ARIMA(train, order=order).fit()
				aic = float(result.aic)
			except Exception:
				continue
			if best_aic is None or aic < best_aic:
				best_aic = aic
				best_result = result
				best_order = order
		if best_result is None or best_order is None:
			raise RuntimeError("Auto ARIMA could not fit a valid model.")
		return best_result, {"order": best_order, "aic": best_aic, "selection": "auto"}

	candidate_orders = list(product(range(0, 3), range(0, 2), range(0, 3)))
	candidate_seasonal = list(product(range(0, 2), range(0, 2), range(0, 2)))
	best_result = None
	best_order = None
	best_seasonal_order = None
	best_aic = None
	seasonal_periods = max(2, int(seasonal_periods))
	for order in candidate_orders:
		for seasonal in candidate_seasonal:
			seasonal_order = (seasonal[0], seasonal[1], seasonal[2], seasonal_periods)
			try:
				result = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False)
				aic = float(result.aic)
			except Exception:
				continue
			if best_aic is None or aic < best_aic:
				best_aic = aic
				best_result = result
				best_order = order
				best_seasonal_order = seasonal_order
		if best_result is None or best_order is None or best_seasonal_order is None:
			raise RuntimeError("Auto SARIMA could not fit a valid model.")
	return best_result, {
		"order": best_order,
		"seasonal_order": best_seasonal_order,
		"aic": best_aic,
		"selection": "auto",
	}


def train_time_series(
	df: pd.DataFrame,
	time_col: str,
	target_col: str,
	model_name: str,
	target_missing: str,
	params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	if not pd.api.types.is_numeric_dtype(df[target_col]):
		return {"error": "Target must be numeric for time series."}
	params = params or {}

	df_ts = df.copy()
	if target_missing == "drop":
		df_ts = df_ts.dropna(subset=[target_col])
	elif target_missing == "impute":
		median_val = df_ts[target_col].median()
		df_ts[target_col] = df_ts[target_col].fillna(median_val)
	elif df_ts[target_col].isna().any():
		return {"error": "Target has missing values. Choose drop or resolve missing values."}
	df_ts[time_col] = pd.to_datetime(df_ts[time_col], errors="coerce")
	df_ts = df_ts.dropna(subset=[time_col, target_col])
	if df_ts.empty:
		return {"error": "No valid rows after parsing time column."}

	df_ts = df_ts.sort_values(time_col)
	series = df_ts.set_index(time_col)[target_col].astype(float)
	if series.shape[0] < 10:
		return {"error": "Not enough rows for time series modeling."}

	split_idx = int(len(series) * 0.8)
	train = series.iloc[:split_idx]
	test = series.iloc[split_idx:]
	if test.empty:
		return {"error": "Not enough rows after split for testing."}

	try:
		from statsmodels.tsa.holtwinters import ExponentialSmoothing
	except Exception as exc:
		return {"error": f"statsmodels is required for time series models: {exc}"}

	model_details: Dict[str, Any] = {}
	try:
		if model_name in ["ARIMA", "SARIMA"]:
			mode = params.get("mode", "Manual")
			if mode == "Auto ARIMA":
				seasonal_periods = int(params.get("seasonal_periods", 12))
				result, model_details = _auto_select_time_series_model(train, model_name, seasonal_periods)
				preds = result.forecast(steps=len(test))
			else:
				order = params.get("order", (1, 1, 1))
				if model_name == "ARIMA":
					from statsmodels.tsa.arima.model import ARIMA

					model = ARIMA(train, order=order)
					result = model.fit()
					preds = result.forecast(steps=len(test))
					model_details = {"order": order, "selection": "manual"}
				else:
					from statsmodels.tsa.statespace.sarimax import SARIMAX

					seasonal_order = params.get("seasonal_order", (1, 1, 1, 12))
					model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
					result = model.fit(disp=False)
					preds = result.forecast(steps=len(test))
					model_details = {
						"order": order,
						"seasonal_order": seasonal_order,
						"selection": "manual",
					}
		else:
			seasonal_periods = params.get("seasonal_periods", 12)
			trend = params.get("trend", "add")
			seasonal = params.get("seasonal", "mul")
			model = ExponentialSmoothing(
				train,
				trend=trend,
				seasonal=seasonal,
				seasonal_periods=seasonal_periods,
			)
			result = model.fit()
			preds = result.forecast(steps=len(test))
			model_details = {
				"trend": trend,
				"seasonal": seasonal,
				"seasonal_periods": seasonal_periods,
			}
	except Exception as exc:
		return {"error": f"Time series model fitting failed: {exc}"}

	test_values = test.to_numpy()
	pred_values = np.asarray(preds)
	metrics = {
		"mae": float(mean_absolute_error(test_values, pred_values)),
		"rmse": float(mean_squared_error(test_values, pred_values, squared=False)),
		"r2": float(r2_score(test_values, pred_values)),
	}
	residuals = (test_values - pred_values).tolist()
	return {
		"metrics": metrics,
		"y_true": test.tolist(),
		"preds": pred_values.tolist(),
		"residuals": residuals,
		"model_details": model_details,
	}


def train_baseline(
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	y_test: pd.Series,
	problem_type: str,
	model_name: str,
	selector: Any,
	preprocessor: ColumnTransformer,
	feature_names: List[str],
	metric_label: str,
) -> Dict[str, Any]:
	model = get_model(problem_type, model_name)
	pipeline = build_pipeline(preprocessor, selector, model)
	pipeline.fit(X_train, y_train)
	preds = pipeline.predict(X_test)
	metrics = evaluate_predictions(y_test, preds, problem_type)

	roc_auc = None
	if problem_type == "classification" and metric_label == "ROC-AUC (ovr)":
		roc_auc = evaluate_roc_auc(pipeline, X_test, y_test)
		if roc_auc is not None:
			metrics["roc_auc_ovr"] = roc_auc

	selected = get_selected_feature_names(pipeline, feature_names)

	return {
		"pipeline": pipeline,
		"metrics": metrics,
		"selected_features": selected,
	}
