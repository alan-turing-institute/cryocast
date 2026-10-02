from pathlib import Path
from typing import ClassVar

from icenet_mp.ingestion.preprocessors import CompositePreprocessor
from icenet_mp.ingestion.preprocessors.ipreprocessor import IPreprocessor


class _RecordingPreprocessor(IPreprocessor):
    calls: ClassVar[list[str]] = []

    def __init__(self, base_path: Path, dataset_name: str, label: str) -> None:
        super().__init__(base_path, dataset_name)
        self.label = label

    def process(self, *, overwrite: bool) -> None:  # noqa: ARG002
        _RecordingPreprocessor.calls.append(self.label)


class TestCompositePreprocessor:
    """Tests for CompositePreprocessor."""

    def test_process_calls_each_child_in_order(self, tmp_path: Path) -> None:
        """process() calls each child preprocessor in config order."""
        _RecordingPreprocessor.calls = []
        cfgs = {
            "first": {
                "_target_": f"{__name__}._RecordingPreprocessor",
                "label": "first",
            },
            "second": {
                "_target_": f"{__name__}._RecordingPreprocessor",
                "label": "second",
            },
        }
        composite = CompositePreprocessor("test", cfgs, tmp_path)
        composite.process(overwrite=True)

        assert _RecordingPreprocessor.calls == ["first", "second"]

    def test_no_children_for_empty_config(self, tmp_path: Path) -> None:
        """An empty preprocessor config builds a composite with no children."""
        composite = CompositePreprocessor("test", {}, tmp_path)
        assert composite.children == []
        composite.process(overwrite=False)  # no-op, must not raise
