from __future__ import annotations

from typing import Any, List, Optional

import nbformat


def create_markdown_cell(text: str) -> nbformat.NotebookNode:
	"""Create a markdown notebook cell."""
	return nbformat.v4.new_markdown_cell(text)


def create_code_cell(code: str) -> nbformat.NotebookNode:
	"""Create a code notebook cell."""
	return nbformat.v4.new_code_cell(code)


def _find_step(workflow: Any, step_type: str) -> Optional[Any]:
	steps = getattr(workflow, "steps", []) or []
	for step in steps:
		if getattr(step, "type", None) == step_type:
			return step
	return None


def _step_params(step: Any) -> dict:
	return dict(getattr(step, "params", {}) or {})


def build_title_cell(workflow: Any) -> nbformat.NotebookNode:
	"""Create the title markdown cell for the exported notebook."""
	dataset = getattr(workflow, "dataset", "dataset.csv")
	target = getattr(workflow, "target", "target")
	model_name = getattr(workflow, "model_name", "Random Forest")
	problem_type = getattr(workflow, "problem_type", "classification")
	markdown = (
		f"# Reproducible Modeling Notebook\n\n"
		f"Dataset: `{dataset}`  \n"
		f"Target: `{target}`  \n"
		f"Model: `{model_name}`  \n"
		f"Problem type: `{problem_type}`\n\n"
		"This notebook was generated from the automated workflow metadata so the analysis can be replayed outside Streamlit."
	)
	return create_markdown_cell(markdown)


def build_imports_cell() -> nbformat.NotebookNode:
	"""Create the imports cell used by the notebook export."""
	code = (
		"from pathlib import Path\n"
		"\n"
		"import matplotlib.pyplot as plt\n"
		"import pandas as pd\n"
		"import seaborn as sns\n"
		"from sklearn.compose import ColumnTransformer\n"
		"from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor\n"
		"from sklearn.impute import SimpleImputer\n"
		"from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso\n"
		"from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score\n"
		"from sklearn.model_selection import train_test_split\n"
		"from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor\n"
		"from sklearn.pipeline import Pipeline\n"
		"from sklearn.preprocessing import OneHotEncoder\n"
		"from sklearn.svm import SVC, SVR\n"
		"from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor\n"
	)
	return create_code_cell(code)


def build_loading_cell(workflow: Any) -> nbformat.NotebookNode:
	"""Create the dataset loading cell."""
	dataset = getattr(workflow, "dataset", "dataset.csv")
	target = getattr(workflow, "target", "target")
	code = (
		"# Load the dataset\n"
		f"data_path = Path(r\"{dataset}\")\n"
		"df = pd.read_csv(data_path)\n"
		f"target = \"{target}\"\n"
		"df.head()\n"
	)
	return create_code_cell(code)


def build_preprocessing_cell(workflow: Any) -> nbformat.NotebookNode:
	"""Create the preprocessing cell, including outlier and missing-value handling."""
	outlier_step = _find_step(workflow, "outlier")
	fillna_step = _find_step(workflow, "fillna")
	outlier_strategy = getattr(outlier_step, "strategy", "Keep outliers") if outlier_step else "Keep outliers"
	outlier_cols = list((_step_params(outlier_step).get("columns") if outlier_step else []) or [])
	numeric_strategy = _step_params(fillna_step).get("numeric_strategy", "median") if fillna_step else "median"
	impute_categorical = bool(_step_params(fillna_step).get("impute_categorical", True)) if fillna_step else True
	missing_strategy = getattr(fillna_step, "strategy", "impute") if fillna_step else "impute"

	lines: List[str] = [
		"# Preprocessing\n",
		"df = df.copy()\n",
		"\n",
		"# Outlier review from the workflow metadata\n",
	]
	if outlier_cols and outlier_strategy == "Remove selected outlier columns":
		lines.extend(
			[
				f"outlier_columns = {outlier_cols!r}\n",
				"mask = pd.Series(False, index=df.index)\n",
				"for column in outlier_columns:\n",
				"    series = df[column].dropna()\n",
				"    if series.empty:\n",
				"        continue\n",
				"    q1 = series.quantile(0.25)\n",
				"    q3 = series.quantile(0.75)\n",
				"    iqr = q3 - q1\n",
				"    low = q1 - 1.5 * iqr\n",
				"    high = q3 + 1.5 * iqr\n",
				"    mask |= (df[column] < low) | (df[column] > high)\n",
				"df = df.loc[~mask].copy()\n",
			]
		)
	else:
		lines.append("# Outliers were kept in the workflow review.\n")
	lines.extend(
		[
			"\n",
			"# Missing-value handling\n",
		]
	)
	if missing_strategy == "drop":
		lines.append("df = df.dropna()\n")
	else:
		lines.extend(
			[
				f"numeric_strategy = {numeric_strategy!r}\n",
				f"impute_categorical = {impute_categorical!r}\n",
				"numeric_cols = df.select_dtypes(include=['number']).columns.tolist()\n",
				"categorical_cols = [col for col in df.columns if col not in numeric_cols and col != target]\n",
				"if numeric_strategy in {'mean', 'median'}:\n",
				"    for column in numeric_cols:\n",
				"        if df[column].isna().any():\n",
				"            fill_value = df[column].mean() if numeric_strategy == 'mean' else df[column].median()\n",
				"            df[column] = df[column].fillna(fill_value)\n",
				"if impute_categorical:\n",
				"    for column in categorical_cols:\n",
				"        if df[column].isna().any():\n",
				"            mode = df[column].mode(dropna=True)\n",
				"            if not mode.empty:\n",
				"                df[column] = df[column].fillna(mode.iloc[0])\n",
			]
		)
	return create_code_cell("".join(lines))


def build_feature_engineering_cell(workflow: Any) -> nbformat.NotebookNode:
	"""Create the feature engineering cell."""
	steps = list(getattr(workflow, "steps", []) or [])
	onehot_requested = any(getattr(step, "type", "") == "onehot" for step in steps)
	selected_features = list(getattr(workflow, "selected_features", []) or [])
	selected_features_literal = repr(selected_features)
	code = [
		"# Feature engineering\n",
		"X = df.drop(columns=[target])\n",
		"y = df[target]\n",
		f"selected_features = {selected_features_literal}\n",
		"if selected_features:\n",
		"    X = X[selected_features]\n",
		"numeric_features = X.select_dtypes(include=['number']).columns.tolist()\n",
		"categorical_features = [column for column in X.columns if column not in numeric_features]\n",
		"numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])\n",
	]
	if onehot_requested or True:
		code.extend(
			[
				"categorical_transformer = Pipeline(steps=[\n",
				"    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
				"    ('onehot', OneHotEncoder(handle_unknown='ignore'))\n",
				"] )\n",
			]
		)
	code.extend(
		[
			"transformers = []\n",
			"if numeric_features:\n",
			"    transformers.append(('num', numeric_transformer, numeric_features))\n",
			"if categorical_features:\n",
			"    transformers.append(('cat', categorical_transformer, categorical_features))\n",
			"preprocessor = ColumnTransformer(transformers=transformers)\n",
		]
	)
	return create_code_cell("".join(code))


def _build_model_code(workflow: Any) -> str:
	problem_type = str(getattr(workflow, "problem_type", "classification")).lower()
	model_name = str(getattr(workflow, "model_name", "Random Forest"))
	if problem_type == "classification":
		mapping = {
			"Random Forest": "RandomForestClassifier(n_estimators=200, random_state=42)",
			"Decision Tree": "DecisionTreeClassifier(random_state=42)",
			"Logistic Regression": "LogisticRegression(max_iter=1000)",
			"Gradient Boosting": "GradientBoostingClassifier(random_state=42)",
			"KNN": "KNeighborsClassifier()",
			"SVM": "SVC(probability=True)",
		}
		return f"model = {mapping.get(model_name, 'RandomForestClassifier(n_estimators=200, random_state=42)')}\n"
	mapping = {
		"Random Forest": "RandomForestRegressor(n_estimators=200, random_state=42)",
		"Decision Tree": "DecisionTreeRegressor(random_state=42)",
		"Linear Regression": "LinearRegression()",
		"Ridge": "Ridge()",
		"Lasso": "Lasso()",
		"Gradient Boosting": "GradientBoostingRegressor(random_state=42)",
		"KNN": "KNeighborsRegressor()",
		"SVR": "SVR()",
	}
	return f"model = {mapping.get(model_name, 'RandomForestRegressor(n_estimators=200, random_state=42)')}\n"


def build_training_cell(workflow: Any) -> nbformat.NotebookNode:
	"""Create the model training cell."""
	problem_type = str(getattr(workflow, "problem_type", "classification")).lower()
	if problem_type == "classification":
		code = [
			"# Model training\n",
			_build_model_code(workflow),
			"pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])\n",
			"stratify = y if y.nunique() > 1 else None\n",
			"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)\n",
			"pipeline.fit(X_train, y_train)\n",
			"predictions = pipeline.predict(X_test)\n",
		]
	else:
		code = [
			"# Model training\n",
			_build_model_code(workflow),
			"pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])\n",
			"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
			"pipeline.fit(X_train, y_train)\n",
			"predictions = pipeline.predict(X_test)\n",
		]
	return create_code_cell("".join(code))


def build_evaluation_cell(workflow: Any) -> nbformat.NotebookNode:
	"""Create the evaluation cell."""
	problem_type = str(getattr(workflow, "problem_type", "classification")).lower()
	if problem_type == "classification":
		code = (
			"# Evaluation\n"
			"accuracy = accuracy_score(y_test, predictions)\n"
			"print(f'Accuracy: {accuracy:.4f}')\n"
			"print(classification_report(y_test, predictions))\n"
			"confusion = confusion_matrix(y_test, predictions)\n"
		)
	else:
		code = (
			"# Evaluation\n"
			"rmse = mean_squared_error(y_test, predictions, squared=False)\n"
			"mae = mean_absolute_error(y_test, predictions)\n"
			"r2 = r2_score(y_test, predictions)\n"
			"print(f'RMSE: {rmse:.4f}')\n"
			"print(f'MAE: {mae:.4f}')\n"
			"print(f'R2: {r2:.4f}')\n"
			"residuals = y_test - predictions\n"
		)
	return create_code_cell(code)


def build_visualization_cell(workflow: Any) -> nbformat.NotebookNode:
	"""Create the visualization cell."""
	problem_type = str(getattr(workflow, "problem_type", "classification")).lower()
	if problem_type == "classification":
		code = (
			"# Visualization\n"
			"fig, ax = plt.subplots(figsize=(5, 4))\n"
			"sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax)\n"
			"ax.set_xlabel('Predicted')\n"
			"ax.set_ylabel('Actual')\n"
			"plt.tight_layout()\n"
			"plt.show()\n"
		)
	else:
		code = (
			"# Visualization\n"
			"fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
			"sns.scatterplot(x=y_test, y=predictions, ax=axes[0])\n"
			"axes[0].set_xlabel('Actual')\n"
			"axes[0].set_ylabel('Predicted')\n"
			"sns.histplot(residuals, kde=True, ax=axes[1])\n"
			"axes[1].set_title('Residual Distribution')\n"
			"plt.tight_layout()\n"
			"plt.show()\n"
		)
	return create_code_cell(code)
