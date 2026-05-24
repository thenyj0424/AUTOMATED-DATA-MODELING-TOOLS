from pathlib import Path

import nbformat

from ai_agent.notebook_generator import NotebookGenerator, NotebookWorkflowStep, build_notebook_metadata_from_context


def test_notebook_generator_exports_expected_cell_order(tmp_path):
    metadata = build_notebook_metadata_from_context(
        {
            "dataset_name": "customer.csv",
            "target_col": "churn",
            "problem_type": "classification",
            "model_family": "ML",
            "model_name": "Random Forest",
            "selected_features": ["age", "income"],
        },
        summary=type("Summary", (), {"categorical_cols": ["city"], "numeric_cols": ["age", "income"]})(),
        state={
            "dataset_name": "customer.csv",
            "target_col": "churn",
            "model_family": "ML",
            "model_name": "Random Forest",
            "missing_strategy": "impute",
            "numeric_impute_strategy": "median",
            "impute_categorical": True,
            "outlier_strategy": "Keep outliers",
            "outlier_remove_cols": [],
            "selected_features": ["age", "income"],
        },
    )
    generator = NotebookGenerator(export_dir=tmp_path)
    export_path = generator.export_notebook(metadata)

    assert export_path.exists()
    assert export_path.suffix == ".ipynb"

    with export_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    assert [cell.cell_type for cell in notebook.cells] == ["markdown", "code", "code", "code", "code", "code", "code", "code"]
    assert "customer.csv" in notebook.cells[0].source
    assert "RandomForestClassifier" in notebook.cells[5].source
    assert notebook.metadata["workflow"]["dataset"] == "customer.csv"


def test_build_notebook_metadata_includes_workflow_steps():
    metadata = build_notebook_metadata_from_context(
        {"dataset_name": "customer.csv", "target_col": "churn", "model_name": "Random Forest"},
        summary=type("Summary", (), {"categorical_cols": ["city"], "numeric_cols": ["age"]})(),
        state={
            "dataset_name": "customer.csv",
            "target_col": "churn",
            "model_family": "ML",
            "model_name": "Random Forest",
            "missing_strategy": "impute",
            "numeric_impute_strategy": "median",
            "impute_categorical": True,
            "outlier_strategy": "Keep outliers",
            "outlier_remove_cols": ["age"],
        },
    )

    assert metadata.dataset == "customer.csv"
    assert metadata.target == "churn"
    assert any(step.type == "fillna" for step in metadata.steps)
    assert any(step.type == "onehot" for step in metadata.steps)
    assert any(step.type == "random_forest" for step in metadata.steps)
