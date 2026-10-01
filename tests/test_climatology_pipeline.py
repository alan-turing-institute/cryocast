"""Integration tests for the climatology baseline model through a real ModelService.

Unlike tests/test_model_service.py, which mocks the data module and model instantiation,
these tests build a real CommonDataModule on the multi-year synthetic zarr and a real
Climatology model, then check that the climatology field flows from the data loaders
through to the model outputs.
"""

import warnings
from pathlib import Path

import torch
from omegaconf import DictConfig

from icenet_mp.model_service import ModelService
from icenet_mp.models import Climatology
from tests.conftest import CLIMATOLOGY_VARIABLES


def _cfg(base_path: Path) -> DictConfig:
    """Build a full service config for the climatology model on the synthetic zarr."""
    return DictConfig(
        {
            "base_path": str(base_path),
            "data": {
                "datasets": {"sic": {"name": "sic_south", "group_as": "sic"}},
                "split": {
                    "batch_size": 2,
                    "predict": [{"start": "2019-12-01", "end": "2019-12-31"}],
                    "test": [{"start": "2019-09-01", "end": "2019-11-30"}],
                    "train": [
                        {"start": "2017-01-01", "end": "2018-12-31"},
                        {"start": "2019-01-01", "end": "2019-06-30"},
                    ],
                    "validate": [{"start": "2019-07-01", "end": "2019-08-31"}],
                },
            },
            "evaluate": {"callbacks": {}},
            "reporting": {
                "loggers": {},
                "metrics": ["mae", "rmse"],
            },
            "loss": {"_target_": "torch.nn.HuberLoss", "delta": 0.5},
            "model": {
                "_target_": "icenet_mp.models.Climatology",
                "name": "climatology",
            },
            "predict": {
                "target": {"group_name": "sic", "variables": CLIMATOLOGY_VARIABLES},
                "n_forecast_steps": 2,
                "n_history_steps": 1,
            },
            "train": {
                "callbacks": {},
                "optimizer": {},
                "scheduler": {},
                "lr_scheduler": {},
                "trainer": {},
            },
        }
    )


class TestClimatologyPipeline:
    def test_model_service_builds_climatology(self, climatology_zarr: Path) -> None:
        """from_config instantiates a Climatology model on the real data module."""
        base_path = climatology_zarr.parents[2]
        service = ModelService.from_config(_cfg(base_path))
        assert isinstance(service.model, Climatology)

    def test_all_splits_batches_contain_climatology(
        self, climatology_zarr: Path
    ) -> None:
        """Every split's first batch carries a correctly-shaped climatology key."""
        base_path = climatology_zarr.parents[2]
        service = ModelService.from_config(_cfg(base_path))
        dm = service.data_module
        for name in ("train", "val", "test", "predict"):
            batch = next(iter(getattr(dm, f"{name}_dataloader")()))
            assert "climatology" in batch
            # shape: batch x n_forecast_steps x C_target x H x W
            assert batch["climatology"].shape == (2, 2, 2, 2, 2)

    def test_test_step_returns_climatology(self, climatology_zarr: Path) -> None:
        """A real test step predicts the batch's climatology field exactly."""
        base_path = climatology_zarr.parents[2]
        service = ModelService.from_config(_cfg(base_path))
        dm = service.data_module
        batch = next(iter(dm.test_dataloader()))
        expected = batch["climatology"].clone()
        expected_target = batch["target"].clone()

        with warnings.catch_warnings():
            # Logging without a trainer only warns; keep the test output clean.
            warnings.simplefilter("ignore")
            output = service.model.test_step(batch, 0)

        assert torch.equal(output.prediction, expected)
        assert torch.equal(output.target, expected_target)
