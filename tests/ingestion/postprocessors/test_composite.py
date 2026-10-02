from pathlib import Path
from typing import ClassVar

from icenet_mp.ingestion.postprocessors import CompositePostprocessor
from icenet_mp.ingestion.postprocessors.ipostprocessor import IPostprocessor


class _RecordingPostprocessor(IPostprocessor):
    calls: ClassVar[list[str]] = []

    def __init__(self, base_path: Path, dataset_name: str, label: str) -> None:
        super().__init__(base_path, dataset_name)
        self.label = label

    def process(self, path_dataset: Path, *, overwrite: bool) -> None:  # noqa: ARG002
        _RecordingPostprocessor.calls.append(self.label)


class TestCompositePostprocessor:
    """Tests for CompositePostprocessor."""

    def test_process_calls_each_child_in_order(self, tmp_path: Path) -> None:
        """process() calls each child postprocessor in config order."""
        _RecordingPostprocessor.calls = []
        cfgs = {
            "first": {
                "_target_": f"{__name__}._RecordingPostprocessor",
                "label": "first",
            },
            "second": {
                "_target_": f"{__name__}._RecordingPostprocessor",
                "label": "second",
            },
        }
        composite = CompositePostprocessor("test", cfgs, tmp_path)
        composite.process(tmp_path, overwrite=True)

        assert _RecordingPostprocessor.calls == ["first", "second"]

    def test_no_children_for_empty_config(self, tmp_path: Path) -> None:
        """An empty postprocessor config builds a composite with no children."""
        composite = CompositePostprocessor("test", {}, tmp_path)
        assert composite.children == []
        composite.process(tmp_path, overwrite=False)  # no-op, must not raise
