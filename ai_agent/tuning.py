from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score


def is_optuna_available() -> bool:
	try:
		import optuna  # noqa: F401
		return True
	except Exception:
		return False


def _suggest_from_grid(trial: Any, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
	params: Dict[str, Any] = {}
	for name, values in param_grid.items():
		if not values:
			continue
		params[name] = trial.suggest_categorical(name, values)
	return params


@dataclass
class OptunaSearch:
	pipeline: Any
	param_grid: Dict[str, List[Any]]
	scoring: str
	cv: int
	n_trials: int

	best_estimator_: Any = None
	best_params_: Dict[str, Any] = None

	def fit(self, X: Any, y: Any) -> "OptunaSearch":
		try:
			import optuna
		except Exception as exc:
			raise RuntimeError("Optuna is required for TPE tuning.") from exc

		sampler = optuna.samplers.TPESampler(seed=42)
		study = optuna.create_study(direction="maximize", sampler=sampler)

		def objective(trial: Any) -> float:
			params = _suggest_from_grid(trial, self.param_grid)
			model = clone(self.pipeline)
			model.set_params(**params)
			scores = cross_val_score(
				model,
				X,
				y,
				scoring=self.scoring,
				cv=self.cv,
				n_jobs=-1,
			)
			return float(np.mean(scores))

		study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
		self.best_params_ = study.best_params
		self.best_estimator_ = clone(self.pipeline)
		self.best_estimator_.set_params(**self.best_params_)
		self.best_estimator_.fit(X, y)
		return self


def run_tuning(
	pipeline: Any,
	search_type: str,
	param_grid: Dict[str, List[Any]],
	scoring: str,
	cv: int,
	n_trials: int,
) -> Any:
	if search_type == "Grid Search":
		search = GridSearchCV(
			pipeline,
			param_grid=param_grid,
			scoring=scoring,
			cv=cv,
			n_jobs=-1,
		)
		return search

	if search_type == "TPE (Optuna)":
		if not param_grid:
			raise RuntimeError("Optuna tuning requires a non-empty parameter grid.")
		return OptunaSearch(
			pipeline=pipeline,
			param_grid=param_grid,
			scoring=scoring,
			cv=cv,
			n_trials=n_trials,
		)

	return None
