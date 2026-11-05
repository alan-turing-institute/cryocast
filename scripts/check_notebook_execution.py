"""Verify the executed metrics notebook, including its explicit pyplot displays.

This checks two schema-validated notebook files without modifying either one.
It deliberately recognizes the metrics example's ``plt.show()`` calls, rather
than claiming to infer arbitrary notebook visualization behavior.
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


def _source(cell: dict[str, Any]) -> str:
    """Normalize notebook JSON's two equivalent source encodings."""
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def _expects_figure(cell: dict[str, Any]) -> bool:
    """Detect explicit pyplot displays without matching comments or string literals."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "plt"
        and node.func.attr == "show"
        for node in ast.walk(ast.parse(_source(cell)))
    )


def check_execution(source_path: Path, executed_path: Path) -> list[str]:
    """Reject skipped/erroring cells, changed source content and missing figures."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    executed = json.loads(executed_path.read_text(encoding="utf-8"))
    if len(source["cells"]) != len(executed["cells"]):
        return ["execution changed the notebook's cell count"]

    problems = []
    counts = []
    for index, (before, after) in enumerate(
        zip(source["cells"], executed["cells"], strict=True)
    ):
        if (
            before["cell_type"] != after["cell_type"]
            or _source(before) != _source(after)
            or before.get("attachments") != after.get("attachments")
            or before.get("id") != after.get("id")
        ):
            problems.append(f"cell {index}: execution changed cell content or identity")
        if before["cell_type"] != "code":
            continue
        counts.append(after.get("execution_count"))
        outputs = after.get("outputs", [])
        if any(output.get("output_type") == "error" for output in outputs):
            problems.append(f"cell {index}: execution produced an error")
        if _expects_figure(before) and not any(
            output.get("data", {}).get("image/png") for output in outputs
        ):
            problems.append(f"cell {index}: pyplot display produced no PNG figure")

    if not counts or counts != list(range(1, len(counts) + 1)):
        problems.append("code cells were not all executed once in notebook order")
    return problems


def main() -> int:
    """Validate a source/observed execution pair for the offline metrics example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("executed", type=Path)
    args = parser.parse_args()
    problems = check_execution(args.source, args.executed)
    if problems:
        sys.stderr.write("\n".join(problems) + "\n")
        return 1
    sys.stdout.write(
        "Verified notebook execution, cell contents and expected figures.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
