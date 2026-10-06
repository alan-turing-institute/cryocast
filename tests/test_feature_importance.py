from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from cryocast.data import CommonDataModule
from cryocast.feature_importance import compute_feature_importance
from cryocast.types import ArrayTCHW


def _cfg(
    base_path: Path, datasets: dict[str, dict[str, str]], target_group: str
) -> DictConfig:
    """Build the shared feature-importance config for the mock dataset."""
    return DictConfig(
        {
            "base_path": str(base_path),
            "data": {
                "datasets": datasets,
                "split": {
                    "predict": [{"start": None, "end": None}],
                    "test": [{"start": None, "end": None}],
                    "train": [{"start": None, "end": None}],
                    "validate": [{"start": None, "end": None}],
                },
            },
            "variables": {
                "input": {},
                "target": {target_group: ["ice_conc"]},
            },
            "window": {
                "batch_size": 2,
                "n_forecast_steps": 1,
                "n_history_steps": 1,
            },
        }
    )


class TestComputeFeatureImportance:
    def test_returns_one_importance_per_variable(self, mock_dataset: Path) -> None:
        base_path = mock_dataset.parents[2]
        config = _cfg(
            base_path,
            {"ds1": {"name": "mock_dataset", "group_as": "group1"}},
            target_group="group1",
        )

        ranked = compute_feature_importance(config, n_estimators=10)

        assert {name for name, _ in ranked} == {
            "group1/ice_conc",
            "group1/ice_thickness",
            "group1/temperature",
        }
        importances = np.array([score for _, score in ranked])
        assert np.all(importances >= 0)
        assert np.isclose(importances.sum(), 1.0)
        # Sorted most important first.
        assert list(importances) == sorted(importances, reverse=True)
        # The near-constant variable carries no signal and must rank last.
        assert ranked[-1][0] == "group1/ice_thickness"

    def test_is_reproducible_across_runs(self, mock_dataset: Path) -> None:
        """Repeated calls must return identical importances (unshuffled training data)."""
        base_path = mock_dataset.parents[2]
        config = _cfg(
            base_path,
            {"ds1": {"name": "mock_dataset", "group_as": "group1"}},
            target_group="group1",
        )

        first = compute_feature_importance(config, n_estimators=10)
        second = compute_feature_importance(config, n_estimators=10)

        assert first == second

    def test_two_dataset_groups_are_both_used_as_features(
        self, mock_dataset: Path
    ) -> None:
        """Every configured dataset group contributes features, not just the target."""
        base_path = mock_dataset.parents[2]
        config = _cfg(
            base_path,
            {
                "ds1": {"name": "mock_dataset", "group_as": "inputs"},
                "ds2": {"name": "mock_dataset", "group_as": "sic-target"},
            },
            target_group="sic-target",
        )

        ranked = compute_feature_importance(config, n_estimators=10)

        assert {name for name, _ in ranked} == {
            "inputs/ice_conc",
            "inputs/ice_thickness",
            "inputs/temperature",
            "sic-target/ice_conc",
            "sic-target/ice_thickness",
            "sic-target/temperature",
        }

    def test_raises_on_nan_input(
        self, mock_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NaN cell should raise a clear error rather than an opaque sklearn one."""
        base_path = mock_dataset.parents[2]
        config = _cfg(
            base_path,
            {"ds1": {"name": "mock_dataset", "group_as": "group1"}},
            target_group="group1",
        )
        original_train_dataloader = CommonDataModule.train_dataloader

        def _nan_train_dataloader(
            self: CommonDataModule,
            *,
            shuffle: bool = True,
        ) -> Iterator[dict[str, ArrayTCHW]]:
            for batch in original_train_dataloader(self, shuffle=shuffle):
                batch["group1"][0, 0, 0, 0, 0] = float("nan")
                yield batch

        monkeypatch.setattr(CommonDataModule, "train_dataloader", _nan_train_dataloader)

        with pytest.raises(ValueError, match="NaN or inf"):
            compute_feature_importance(config, n_estimators=10)
