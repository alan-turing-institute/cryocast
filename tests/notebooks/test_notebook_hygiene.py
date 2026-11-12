"""Regression tests for notebook hygiene without erasing instructional content."""

import json
from pathlib import Path

import pytest

from scripts.notebook_hygiene import check_or_clean, notebook_paths

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
TRACKED_NOTEBOOKS = notebook_paths(NOTEBOOKS)


def test_existing_notebook_inventory_is_not_silently_reduced() -> None:
    """Keep every reviewed notebook when changing the CI or cleaning outputs."""
    expected = {
        "argo_data.ipynb",
        "case_study_whale_corridors.ipynb",
        "degrid_and_visualise.ipynb",
        "demo_pipeline.ipynb",
        "extract_anomalies.ipynb",
        "layer_diagnostics.ipynb",
        "metrics_examples.ipynb",
    }
    assert expected <= {path.name for path in TRACKED_NOTEBOOKS}


def test_demo_walkthrough_and_config_are_preserved() -> None:
    """Guard the educational material raised in IFenton's PR #510 review."""
    notebook = json.loads((NOTEBOOKS / "demo_pipeline.ipynb").read_text())
    cells = notebook["cells"]
    assert len(cells) >= 90
    markdown = [c for c in cells if c["cell_type"] == "markdown"]
    assert len(markdown) >= 65
    narrative = "\n".join(source for cell in markdown for source in cell["source"])
    assert "Section 1: End-to-End Training Pipeline" in narrative
    assert "Section 2: Model Flexibility" in narrative
    assert "Section 3: Full flexibility" in narrative
    config_dir = NOTEBOOKS.parent / "cryocast" / "config"
    assert (config_dir / "demo_notebook.yaml").is_file()
    assert (config_dir / "synthetic.yaml").is_file()


@pytest.mark.parametrize("path", TRACKED_NOTEBOOKS, ids=lambda p: p.name)
def test_repository_notebooks_are_clean(path: Path) -> None:
    """Reject stored data, execution counts, timings, and empty cells on any PR."""
    assert check_or_clean(path) == []


@pytest.mark.parametrize("path", TRACKED_NOTEBOOKS, ids=lambda p: p.name)
def test_repository_notebooks_have_consistent_structure(path: Path) -> None:
    """Ensure every tracked file has valid basic Jupyter notebook structure."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    assert isinstance(notebook["metadata"], dict)
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "raw", "code"}
        assert isinstance(cell["source"], (str, list))
        assert isinstance(cell["metadata"], dict)


def _fixture_notebook() -> dict[str, object]:
    """Make a small notebook with both valuable content and runtime artifacts."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "jupytext": {"formats": "ipynb,py:percent"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "explain",
                "metadata": {},
                "source": ["# Scientific example\n", "Preserve this explanation.\n"],
            },
            {
                "cell_type": "code",
                "id": "compute",
                "metadata": {
                    "tags": ["parameters"],
                    "execution": {"iopub.execute_input": "2026-10-10T01:00:00Z"},
                },
                "source": ["# Scientific calculation\n", "answer = 42\n", "answer"],
                "execution_count": 7,
                "outputs": [
                    {
                        "output_type": "execute_result",
                        "execution_count": 7,
                        "data": {"text/plain": "42"},
                        "metadata": {},
                    }
                ],
            },
        ],
    }


def test_cleaner_preserves_sources_order_and_nonruntime_metadata(
    tmp_path: Path,
) -> None:
    """The clean action must never remove or rewrite a notebook's teaching cells."""
    original = _fixture_notebook()
    path = tmp_path / "scientific.ipynb"
    path.write_text(json.dumps(original, indent=1) + "\n", encoding="utf-8")

    assert check_or_clean(path) == [
        "cell 1: stored output",
        "cell 1: stored execution count",
        "cell 1: saved execution timing",
    ]
    assert json.loads(path.read_text(encoding="utf-8")) == original

    assert check_or_clean(path, clean=True) == []
    cleaned = json.loads(path.read_text(encoding="utf-8"))
    original_cells = original["cells"]
    assert isinstance(original_cells, list)
    assert len(cleaned["cells"]) == len(original_cells)
    assert [c["source"] for c in cleaned["cells"]] == [
        c["source"] for c in original_cells
    ]
    assert [c["id"] for c in cleaned["cells"]] == [c["id"] for c in original_cells]
    assert cleaned["cells"][0] == original_cells[0]
    assert cleaned["cells"][1]["metadata"] == {"tags": ["parameters"]}
    assert cleaned["cells"][1]["outputs"] == []
    assert cleaned["cells"][1]["execution_count"] is None
    assert cleaned["metadata"] == original["metadata"]

    first_result = path.read_bytes()
    assert check_or_clean(path, clean=True) == []
    assert path.read_bytes() == first_result
    assert check_or_clean(path) == []


def test_empty_markdown_cell_is_reported_and_never_deleted(tmp_path: Path) -> None:
    """Clean mode must report empty cells without silently cutting notebook text."""
    notebook = _fixture_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    cells[0]["source"] = [""]
    path = tmp_path / "empty.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

    assert "cell 0: empty cell" in check_or_clean(path)
    assert check_or_clean(path, clean=True) == ["cell 0: empty cell"]
    assert len(json.loads(path.read_text(encoding="utf-8"))["cells"]) == 2


def test_invalid_structure_is_rejected(tmp_path: Path) -> None:
    """A non-notebook JSON file must fail instead of passing the hygiene gate."""
    path = tmp_path / "invalid.ipynb"
    path.write_text('{"nbformat": 3, "cells": []}', encoding="utf-8")
    assert check_or_clean(path) == ["not a valid Jupyter notebook v4 structure"]


def test_notebook_discovery_includes_new_nested_files(tmp_path: Path) -> None:
    """New notebooks are checked automatically without editing any allowlist."""
    subdir = tmp_path / "more"
    subdir.mkdir()
    (tmp_path / "one.ipynb").touch()
    (subdir / "two.ipynb").touch()
    assert {p.name for p in notebook_paths(tmp_path)} == {
        "one.ipynb",
        "two.ipynb",
    }


def test_duplicate_cell_ids_are_rejected_without_deleting_cells(
    tmp_path: Path,
) -> None:
    """Reject invalid cell identities without changing the notebook structure."""
    notebook = _fixture_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    cells[1]["id"] = cells[0]["id"]
    path = tmp_path / "duplicate.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    assert "cell 1: duplicate cell id 'explain'" in check_or_clean(path)
    assert "cell 1: duplicate cell id 'explain'" in check_or_clean(path, clean=True)
    assert len(json.loads(path.read_text(encoding="utf-8"))["cells"]) == 2


@pytest.mark.parametrize("indent", [None, 4])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_clean_mode_preserves_already_clean_file_bytes(
    tmp_path: Path, indent: int | None, newline: str
) -> None:
    """A clean file must not be reformatted or have its line endings rewritten."""
    notebook = _fixture_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    cells[1]["outputs"] = []
    cells[1]["execution_count"] = None
    del cells[1]["metadata"]["execution"]
    original = (
        (json.dumps(notebook, indent=indent) + "\n").replace("\n", newline).encode()
    )
    path = tmp_path / "already-clean.ipynb"
    path.write_bytes(original)

    assert check_or_clean(path, clean=True) == []
    assert path.read_bytes() == original


@pytest.mark.parametrize("clean", [False, True])
def test_zero_execution_count_is_not_treated_as_missing(
    tmp_path: Path, *, clean: bool
) -> None:
    """An execution count of zero is saved state even without output objects."""
    notebook = _fixture_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    cells[1]["outputs"] = []
    cells[1]["execution_count"] = 0
    del cells[1]["metadata"]["execution"]
    path = tmp_path / "zero-count.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    before = path.read_bytes()

    problems = check_or_clean(path, clean=clean)

    if clean:
        assert problems == []
        assert json.loads(path.read_text())["cells"][1]["execution_count"] is None
    else:
        assert problems == ["cell 1: stored execution count"]
        assert path.read_bytes() == before


@pytest.mark.parametrize("string_source", [False, True])
def test_cleaner_retains_attachments_raw_cells_and_display_metadata(
    tmp_path: Path, *, string_source: bool
) -> None:
    """Only code-cell runtime state may be removed from a rich notebook."""
    notebook = _fixture_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    cells[0]["attachments"] = {"figure.svg": {"image/svg+xml": "<svg>sea ice</svg>"}}
    cells[0]["metadata"] = {"tags": ["teaching"], "custom": {"keep": True}}
    cells[1]["metadata"]["collapsed"] = True
    cells.append(
        {
            "cell_type": "raw",
            "id": "notes",
            "source": ["Arctic area in km²"],
            "metadata": {"format": "text/latex"},
        }
    )
    if string_source:
        for cell in cells:
            cell["source"] = "".join(cell["source"])
    path = tmp_path / "rich.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    expected = json.loads(path.read_text())
    expected["cells"][1]["outputs"] = []
    expected["cells"][1]["execution_count"] = None
    del expected["cells"][1]["metadata"]["execution"]

    assert check_or_clean(path, clean=True) == []
    assert json.loads(path.read_text()) == expected
