from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from icenet_mp.ingestion.downloaders import build_downloaders
from icenet_mp.ingestion.postprocessors import (
    StatusFlagMaskGenerator,
    SyntheticMaskGenerator,
)


class TestBuildDownloaders:
    """Tests for the build_downloaders function."""

    STATUS_FLAG_TARGET = "icenet_mp.ingestion.postprocessors.StatusFlagMaskGenerator"
    SYNTHETIC_TARGET = "icenet_mp.ingestion.postprocessors.SyntheticMaskGenerator"

    def _config(self, tmp_path: Path, dataset_overrides: dict[str, Any]) -> DictConfig:
        """Build a minimal Hydra-style config for a single dataset named "test"."""
        return OmegaConf.create(
            {
                "base_path": str(tmp_path),
                "data": {
                    "datasets": {"test": self._dataset("test", dataset_overrides)}
                },
            }
        )

    def _dataset(
        self, name: str, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Build a minimal dataset entry, e.g. for the "datasets" key of a Hydra-style config."""
        return {
            "name": name,
            "dates": {"start": "2020-01-01", "end": "2020-01-31", "frequency": "24h"},
            **(overrides or {}),
        }

    def test_missing_pre_and_postprocessors_build_empty_composites(
        self, tmp_path: Path
    ) -> None:
        """A dataset with no preprocessors/postprocessors keys builds empty composites."""
        downloader = build_downloaders(self._config(tmp_path, {}))[0]
        assert downloader.preprocessor.children == []
        assert downloader.postprocessor.children == []

    @pytest.mark.parametrize(
        ("postprocessors", "expected_types"),
        [
            (
                {"status_flag_masks": {"_target_": STATUS_FLAG_TARGET}},
                [StatusFlagMaskGenerator],
            ),
            (
                {
                    "status_flag_masks": {"_target_": STATUS_FLAG_TARGET},
                    "synthetic_masks": {"_target_": SYNTHETIC_TARGET},
                },
                [StatusFlagMaskGenerator, SyntheticMaskGenerator],
            ),
        ],
        ids=["single", "multiple_in_order"],
    )
    def test_postprocessor_specs_build_composite(
        self,
        tmp_path: Path,
        postprocessors: dict[str, Any],
        expected_types: list[type],
    ) -> None:
        """Postprocessor specs build a composite preserving config order."""
        config = self._config(tmp_path, {"postprocessors": postprocessors})
        children = build_downloaders(config)[0].postprocessor.children
        assert [type(child) for child in children] == expected_types

    def test_multiple_datasets_build_one_downloader_each_in_order(
        self, tmp_path: Path
    ) -> None:
        """Each dataset entry in the config builds its own downloader, in config order."""
        config = OmegaConf.create(
            {
                "base_path": str(tmp_path),
                "data": {
                    "datasets": {
                        "first": self._dataset("first"),
                        "second": self._dataset("second"),
                    }
                },
            }
        )
        downloaders = build_downloaders(config)
        assert [downloader.name for downloader in downloaders] == ["first", "second"]
