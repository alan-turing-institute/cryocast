"""Check or clean saved notebook outputs without editing teaching or research content.

The default mode is read-only and can be run by pre-commit, pytest or CI.
Use --clean after an interactive notebook run to remove its saved results and
execution timestamps. Cell sources, ordering and notebook metadata are kept.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
NBFORMAT_VERSION = 4


def notebook_paths(directory: Path) -> list[Path]:
    """Find all notebooks in the checked directory, including newly added ones."""
    return sorted(directory.rglob("*.ipynb"))


def _runtime_problems(cell: dict[str, object], index: int, *, clean: bool) -> list[str]:
    """Check generated state without looking inside scientific code or outputs."""
    problems = []
    for key, label, replacement in (
        ("outputs", "stored output", []),
        ("execution_count", "stored execution count", None),
    ):
        if cell.get(key):
            if clean:
                cell[key] = replacement
            else:
                problems.append(f"cell {index}: {label}")

    # An execution count of zero is still saved state, but not a missing value.
    if cell.get("execution_count") == 0:
        if clean:
            cell["execution_count"] = None
        else:
            problems.append(f"cell {index}: stored execution count")

    metadata = cell.get("metadata")
    if isinstance(metadata, dict) and "execution" in metadata:
        if clean:
            del metadata["execution"]
        else:
            problems.append(f"cell {index}: saved execution timing")
    return problems


def _cell_problems(cell: dict[str, object], index: int, *, clean: bool) -> list[str]:
    """Protect source text and report blank, malformed, or executed cells."""
    source = cell.get("source", [])
    if isinstance(source, str):
        contents = source
    elif isinstance(source, list) and all(isinstance(line, str) for line in source):
        contents = "".join(source)
    else:
        return [f"cell {index}: source is not text"]

    problems = [f"cell {index}: empty cell"] if not contents.strip() else []
    if cell.get("cell_type") == "code":
        problems.extend(_runtime_problems(cell, index, clean=clean))
    return problems


def check_or_clean(path: Path, *, clean: bool = False) -> list[str]:
    """Return cleanliness problems, optionally removing runtime-only state."""
    original = path.read_text(encoding="utf-8")
    notebook = json.loads(original)
    if notebook.get("nbformat") != NBFORMAT_VERSION or not isinstance(
        notebook.get("cells"), list
    ):
        return ["not a valid Jupyter notebook v4 structure"]

    problems = []
    ids: set[str] = set()
    for index, cell in enumerate(notebook["cells"]):
        cell_id = cell.get("id")
        if isinstance(cell_id, str):
            if cell_id in ids:
                problems.append(f"cell {index}: duplicate cell id {cell_id!r}")
            ids.add(cell_id)
        problems.extend(_cell_problems(cell, index, clean=clean))

    if clean and notebook != json.loads(original):
        # The writer matches the repository's original notebook JSON layout.
        # Do not re-write anything unless generated state actually changed.
        encoded = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        if encoded != original:
            path.write_text(encoded, encoding="utf-8")
    return problems


def main() -> int:
    """Check every notebook, or clean only generated execution state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Strip saved outputs/counts/timings without changing notebook cells",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_NOTEBOOKS,
        help="Notebook directory (defaults to the repository notebooks directory)",
    )
    args = parser.parse_args()
    files = notebook_paths(args.directory)
    if not files:
        parser.error(f"No notebooks found under {args.directory}")

    errors = [
        f"{path}: {problem}"
        for path in files
        for problem in check_or_clean(path, clean=args.clean)
    ]
    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        return 1
    verb = "Cleaned" if args.clean else "Checked"
    sys.stdout.write(
        f"{verb} {len(files)} notebooks; all code and markdown cells retained.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
