from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import nbformat

from ai_agent.cell_builders import (
	build_evaluation_cell,
	build_feature_engineering_cell,
	build_imports_cell,
	build_loading_cell,
	build_preprocessing_cell,
	build_title_cell,
	build_training_cell,
	build_visualization_cell,
	create_markdown_cell,
)


@dataclass
class NotebookWorkflowStep:
	type: str
	strategy: Optional[str] = None
	params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotebookWorkflowMetadata:
	dataset: str
	target: str
	problem_type: str
	model_family: str
	model_name: str
	steps: List[NotebookWorkflowStep] = field(default_factory=list)
	selected_features: List[str] = field(default_factory=list)
	notes: List[str] = field(default_factory=list)
	output_name: Optional[str] = None


def _sanitize_filename(text: str) -> str:
	cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "notebook")).strip("_")
	return cleaned or "notebook"


def _coerce_step(step: Any) -> NotebookWorkflowStep:
	if isinstance(step, NotebookWorkflowStep):
		return step
	if isinstance(step, dict):
		return NotebookWorkflowStep(
			type=str(step.get("type", "step")),
			strategy=step.get("strategy"),
			params=dict(step.get("params", {}) or {}),
		)
	return NotebookWorkflowStep(type=str(step))


def build_notebook_metadata_from_context(
	results: Optional[Dict[str, Any]],
	summary: Any,
	state: Optional[Dict[str, Any]] = None,
) -> NotebookWorkflowMetadata:
	"""Build notebook metadata from the current workflow context."""
	results = results or {}
	state = state or {}
	dataset = str(results.get("dataset_name") or state.get("dataset_name") or "dataset.csv")
	target = str(results.get("target_col") or state.get("target_col") or "target")
	problem_type = str(results.get("problem_type") or state.get("problem_type") or "classification")
	model_family = str(state.get("model_family") or results.get("model_family") or "ML")
	model_name = str(results.get("model_name") or state.get("model_name") or "Random Forest")
	selected_features = list(results.get("selected_features") or state.get("selected_features") or [])
	notes: List[str] = []
	if state.get("cleaning_review_note"):
		notes.append(str(state.get("cleaning_review_note")))
	if state.get("analysis_ready_message"):
		notes.append(str(state.get("analysis_ready_message")))

	steps: List[NotebookWorkflowStep] = []
	outlier_strategy = state.get("outlier_strategy")
	outlier_cols = list(state.get("outlier_remove_cols") or [])
	if outlier_strategy or outlier_cols:
		steps.append(
			dict(
				type="outlier",
				strategy=outlier_strategy or "Keep outliers",
				params={"columns": outlier_cols},
			)
		)
	missing_strategy = state.get("missing_strategy") or "impute"
	steps.append(
		dict(
			type="fillna",
			strategy=missing_strategy,
			params={
				"numeric_strategy": state.get("numeric_impute_strategy", "median"),
				"impute_categorical": bool(state.get("impute_categorical", True)),
			},
		)
	)
	if getattr(summary, "categorical_cols", None):
		steps.append(dict(type="onehot", strategy="onehot", params={"columns": list(getattr(summary, "categorical_cols", []) or [])}))
	steps.append(dict(type=model_name.lower().replace(" ", "_"), strategy=model_name, params={"family": model_family}))

	return NotebookWorkflowMetadata(
		dataset=dataset,
		target=target,
		problem_type=problem_type,
		model_family=model_family,
		model_name=model_name,
		steps=[_coerce_step(step) for step in steps],
		selected_features=selected_features,
		notes=notes,
	)


class NotebookGenerator:
	"""Build and export reproducible notebooks from workflow metadata."""

	def __init__(self, template_dir: Optional[Path] = None, export_dir: Optional[Path] = None) -> None:
		base_dir = Path(__file__).resolve().parents[1]
		self.template_dir = Path(template_dir) if template_dir else base_dir / "templates"
		self.export_dir = Path(export_dir) if export_dir else base_dir / "exported_notebooks"
		self.export_dir.mkdir(parents=True, exist_ok=True)

	def _load_template(self, template_name: str, fallback: str) -> str:
		path = self.template_dir / template_name
		if path.exists():
			return path.read_text(encoding="utf-8")
		return fallback

	def _render_template(self, template: str, metadata: NotebookWorkflowMetadata) -> str:
		return (
			template.replace("{{dataset}}", metadata.dataset)
			.replace("{{target}}", metadata.target)
			.replace("{{model_name}}", metadata.model_name)
			.replace("{{problem_type}}", metadata.problem_type)
		)

	def build_notebook(self, metadata: NotebookWorkflowMetadata) -> nbformat.NotebookNode:
		"""Assemble the notebook from deterministic cell builders."""
		title_template = self._load_template(
			"notebook_title.md",
			"# Reproducible Modeling Notebook\n\nDataset: `{{dataset}}`  \nTarget: `{{target}}`  \nModel: `{{model_name}}`  \nProblem type: `{{problem_type}}`",
		)
		cells = [
			create_markdown_cell(self._render_template(title_template, metadata)),
			build_imports_cell(),
			build_loading_cell(metadata),
			build_preprocessing_cell(metadata),
			build_feature_engineering_cell(metadata),
			build_training_cell(metadata),
			build_evaluation_cell(metadata),
			build_visualization_cell(metadata),
		]
		return nbformat.v4.new_notebook(
			cells=cells,
			metadata={
				"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
				"language_info": {"name": "python"},
				"workflow": asdict(metadata),
			},
		)

	def _default_filename(self, metadata: NotebookWorkflowMetadata) -> str:
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		dataset_slug = _sanitize_filename(Path(metadata.dataset).stem or metadata.dataset)
		model_slug = _sanitize_filename(metadata.model_name)
		return f"{dataset_slug}_{model_slug}_{stamp}.ipynb"

	def export_notebook(self, metadata: NotebookWorkflowMetadata, output_name: Optional[str] = None) -> Path:
		"""Write the notebook to disk and return the exported path."""
		notebook = self.build_notebook(metadata)
		file_name = output_name or metadata.output_name or self._default_filename(metadata)
		export_path = self.export_dir / file_name
		with export_path.open("w", encoding="utf-8") as handle:
			nbformat.write(notebook, handle)
		return export_path

	def export_notebook_bytes(self, metadata: NotebookWorkflowMetadata, output_name: Optional[str] = None) -> tuple[Path, bytes]:
		"""Export the notebook and return both the path and file bytes for download."""
		export_path = self.export_notebook(metadata, output_name=output_name)
		return export_path, export_path.read_bytes()
