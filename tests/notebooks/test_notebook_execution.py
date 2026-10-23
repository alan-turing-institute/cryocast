"""Check that notebook CI verifies displayed figures as well as successful cells."""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.check_notebook_execution import check_execution, main


def _pair(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Write a minimal original notebook and construct its successful execution."""
    source: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "id": "explanation",
                "metadata": {},
                "source": ["# Keep the scientific explanation"],
            },
            {
                "cell_type": "code",
                "id": "figure",
                "metadata": {},
                "source": ["import matplotlib.pyplot as plt\n", "plt.show()"],
                "execution_count": None,
                "outputs": [],
            },
        ],
    }
    source_path = tmp_path / "source.ipynb"
    output_path = tmp_path / "executed.ipynb"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    executed = copy.deepcopy(source)
    executed["cells"][1]["execution_count"] = 1
    executed["cells"][1]["outputs"] = [
        {
            "output_type": "display_data",
            "data": {"image/png": "test-image-payload"},
            "metadata": {},
        }
    ]
    return source_path, output_path, executed


@pytest.mark.parametrize("string_source", [False, True])
def test_successful_execution_is_read_only_and_accepts_source_encodings(
    tmp_path: Path, *, string_source: bool
) -> None:
    """JSON string/list source representations have the same notebook semantics."""
    source, output, executed = _pair(tmp_path)
    if string_source:
        for cell in executed["cells"]:
            cell["source"] = "".join(cell["source"])
    output.write_text(json.dumps(executed), encoding="utf-8")
    before = source.read_bytes(), output.read_bytes()

    assert check_execution(source, output) == []
    assert (source.read_bytes(), output.read_bytes()) == before


@pytest.mark.parametrize("execution_count", [None, 0, 2])
def test_skipped_or_out_of_order_execution_fails(
    tmp_path: Path, execution_count: int | None
) -> None:
    """A notebook must execute every code cell, starting from a fresh kernel."""
    source, output, executed = _pair(tmp_path)
    executed["cells"][1]["execution_count"] = execution_count
    output.write_text(json.dumps(executed), encoding="utf-8")

    assert "code cells were not all executed once in notebook order" in check_execution(
        source, output
    )


@pytest.mark.parametrize("output_kind", ["none", "text-only", "empty-image"])
def test_pyplot_display_requires_an_observed_image(
    tmp_path: Path, output_kind: str
) -> None:
    """Successful cells under a non-displaying backend must not pass plot CI."""
    source, output, executed = _pair(tmp_path)
    cell = executed["cells"][1]
    if output_kind == "none":
        cell["outputs"] = []
    elif output_kind == "text-only":
        cell["outputs"][0]["data"] = {"text/plain": "<Figure>"}
    else:
        cell["outputs"][0]["data"]["image/png"] = ""
    output.write_text(json.dumps(executed), encoding="utf-8")

    assert check_execution(source, output) == [
        "cell 1: pyplot display produced no PNG figure"
    ]


def test_error_output_is_rejected_even_when_a_figure_exists(tmp_path: Path) -> None:
    """An allowed-error cell must still fail this notebook's execution contract."""
    source, output, executed = _pair(tmp_path)
    executed["cells"][1]["outputs"].append(
        {
            "output_type": "error",
            "ename": "ValueError",
            "evalue": "bad data",
            "traceback": [],
        }
    )
    output.write_text(json.dumps(executed), encoding="utf-8")

    assert "cell 1: execution produced an error" in check_execution(source, output)


@pytest.mark.parametrize("field", ["source", "id", "cell_type", "attachments"])
def test_execution_cannot_alter_instructional_content(
    tmp_path: Path, field: str
) -> None:
    """A different explanation, attachment, identity or cell order must be visible."""
    source, output, executed = _pair(tmp_path)
    replacements = {
        "source": ["Replaced explanation"],
        "id": "different-identity",
        "cell_type": "raw",
        "attachments": {"image.svg": {"image/svg+xml": "<svg/>"}},
    }
    executed["cells"][0][field] = replacements[field]
    output.write_text(json.dumps(executed), encoding="utf-8")

    assert check_execution(source, output) == [
        "cell 0: execution changed cell content or identity"
    ]


def test_execution_cannot_remove_cells(tmp_path: Path) -> None:
    """Executing only a shortened notebook does not satisfy the full example."""
    source, output, executed = _pair(tmp_path)
    executed["cells"].pop(0)
    output.write_text(json.dumps(executed), encoding="utf-8")

    assert check_execution(source, output) == [
        "execution changed the notebook's cell count"
    ]


@pytest.mark.parametrize(
    "code", ["# plt.show()\nanswer = 42", 'message = "plt.show()"']
)
def test_comments_and_strings_do_not_require_figures(tmp_path: Path, code: str) -> None:
    """Parse actual display calls rather than matching text inside explanations."""
    source, output, executed = _pair(tmp_path)
    original = json.loads(source.read_text())
    original["cells"][1]["source"] = code
    source.write_text(json.dumps(original), encoding="utf-8")
    executed["cells"][1]["source"] = code
    executed["cells"][1]["outputs"] = []
    output.write_text(json.dumps(executed), encoding="utf-8")

    assert check_execution(source, output) == []


@pytest.mark.parametrize("with_image", [False, True])
def test_execution_checker_cli_reports_failure_without_writing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    with_image: bool,
) -> None:
    """The same read-only command is used by CI and local validation."""
    source, output, executed = _pair(tmp_path)
    if not with_image:
        executed["cells"][1]["outputs"] = []
    output.write_text(json.dumps(executed), encoding="utf-8")
    before = source.read_bytes(), output.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["check_notebook_execution", str(source), str(output)]
    )

    assert main() == (0 if with_image else 1)
    captured = capsys.readouterr()
    assert ("Verified notebook" in captured.out) is with_image
    assert ("no PNG figure" in captured.err) is (not with_image)
    assert (source.read_bytes(), output.read_bytes()) == before
