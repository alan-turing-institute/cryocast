from pathlib import Path
from unittest.mock import MagicMock

import pytest

from icenet_mp.data.single_dataset import SingleDataset


class TestDatasetCache:
    def test_anemoi_dataset_is_loaded_once_per_fileset(
        self, mock_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated reads of one fileset should reuse the class-level dataset cache."""
        mock_open_dataset = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.data.single_dataset.open_dataset", mock_open_dataset
        )
        monkeypatch.setattr(SingleDataset, "anemoi_cache", {})
        input_files = (mock_dataset,)

        first = SingleDataset.load_dataset(input_files)
        second = SingleDataset.load_dataset(input_files)

        assert first is second
        mock_open_dataset.assert_called_once_with(input_files)

    def test_subsets_share_cached_anemoi_dataset(
        self, mock_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SingleDataset subsets should not reopen their shared source files."""
        mock_open_dataset = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.data.single_dataset.open_dataset", mock_open_dataset
        )
        monkeypatch.setattr(SingleDataset, "anemoi_cache", {})
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        subset = dataset.subset(variables=["ice_conc"])

        first = dataset.load_dataset(dataset._input_files)
        second = subset.load_dataset(subset._input_files)

        assert first is second
        mock_open_dataset.assert_called_once_with((mock_dataset,))
