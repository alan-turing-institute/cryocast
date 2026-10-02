import logging
from pathlib import Path
from typing import Any

import lightning
import numpy as np
import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanSquaredError, Metric

import icenet_mp
from icenet_mp.losses import LeadTimeWeightedLoss
from icenet_mp.losses.amse_loss import AMSELoss
from icenet_mp.losses.rmse_loss import RMSELoss
from icenet_mp.losses.weighted_bce_loss import WeightedBCEWithLogitsLoss
from icenet_mp.losses.weighted_l1_loss import WeightedL1Loss
from icenet_mp.losses.weighted_mse_loss import WeightedMSELoss
from icenet_mp.metrics import (
    CentroidErrorPerForecastDay,
    DistanceAveragedIceEdgeErrorPerForecastDay,
    FractionalSkillScorePerForecastDay,
    IceNetAccuracyPerForecastDay,
    IntegratedIceEdgeErrorPerForecastDay,
    MAEPerForecastDay,
    RMSEPerForecastDay,
    SeaIceExtentErrorPerForecastDay,
    SpatialMeanGroundTruthPerForecastDay,
    SpatialMeanPredictionPerForecastDay,
    SSIMPerForecastDay,
)
from icenet_mp.models import BaseModel, Persistence
from icenet_mp.types import Hemisphere, ModelStepOutput, TensorNTCHW

# The metrics configured by default, which also checks that the shipped config is valid
DEFAULT_METRICS: list[Any] = OmegaConf.to_container(  # type: ignore[assignment]
    OmegaConf.load(
        Path(icenet_mp.__file__).parent / "config/reporting/metrics/default.yaml"
    )
)


def metric_spec(name: str, metric_type: type, **kwargs: Any) -> dict[str, Any]:
    """Build a metric config entry for this metric class."""
    return {
        "name": name,
        "_target_": f"{metric_type.__module__}.{metric_type.__qualname__}",
        **kwargs,
    }


NON_FSS_METRIC_TYPES = {
    "accuracy": IceNetAccuracyPerForecastDay,
    "mae": MAEPerForecastDay,
    "rmse": RMSEPerForecastDay,
    "sieerror": SeaIceExtentErrorPerForecastDay,
    "iiee": IntegratedIceEdgeErrorPerForecastDay,
    "diiee": DistanceAveragedIceEdgeErrorPerForecastDay,
    "centroid_error": CentroidErrorPerForecastDay,
    "ssim": SSIMPerForecastDay,
    "spatial_mean_ground_truth": SpatialMeanGroundTruthPerForecastDay,
    "spatial_mean_prediction": SpatialMeanPredictionPerForecastDay,
}


class FakeDataModel(BaseModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise a fake data model for testing purposes."""
        loss_cfg = kwargs.pop(
            "loss", OmegaConf.create({"_target_": "torch.nn.HuberLoss", "delta": 0.5})
        )
        metrics = kwargs.pop("metrics", DEFAULT_METRICS)
        super().__init__(
            *args, loss=loss_cfg, metrics=metrics, hemisphere=Hemisphere.NORTH, **kwargs
        )
        self.model = torch.nn.Linear(1, 1)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """FakeData forward method."""
        b = next(iter(inputs.values())).shape[0]
        return torch.randn(b, self.n_forecast_steps, *self.output_space.chw)


class EchoForecastModel(BaseModel):
    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        """Return the "forecast" input unchanged to isolate metric accumulation."""
        return inputs["forecast"]


class TestBaseModel:
    @pytest.mark.parametrize(
        ("n_forecast_steps", "n_history_steps", "match"),
        [
            (0, 1, r"Number of forecast steps must be greater than 0."),
            (1, 0, r"Number of history steps must be greater than 0."),
        ],
        ids=["zero-forecast-steps", "zero-history-steps"],
    )
    def test_init_invalid_steps(
        self, n_forecast_steps: int, n_history_steps: int, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            FakeDataModel(
                name="fake data",
                input_spaces=[{"channels": 1, "name": "input", "shape": (2, 2)}],
                n_forecast_steps=n_forecast_steps,
                n_history_steps=n_history_steps,
                output_space={"channels": 1, "name": "target", "shape": (2, 2)},
                optimizer=DictConfig({}),
                scheduler=DictConfig({}),
                lr_scheduler=DictConfig({}),
            )

    @pytest.mark.parametrize(
        "test_input_chw",
        [(4, 512, 512), (1, 10, 20)],
        ids=lambda chw: "input-{}x{}x{}".format(*chw),
    )
    @pytest.mark.parametrize(
        "test_output_chw",
        [(1, 432, 432), (19, 10, 20)],
        ids=lambda chw: "output-{}x{}x{}".format(*chw),
    )
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 2, 5])
    @pytest.mark.parametrize("test_n_history_steps", [1, 2, 5])
    def test_init_valid(
        self,
        test_input_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_output_chw: tuple[int, int, int],
    ) -> None:
        input_space = DictConfig(
            {
                "channels": test_input_chw[0],
                "name": "input",
                "shape": test_input_chw[1:],
            }
        )
        output_space = DictConfig(
            {
                "channels": test_output_chw[0],
                "name": "target",
                "shape": test_output_chw[1:],
            }
        )
        model = FakeDataModel(
            name="fake data",
            input_spaces=[input_space],
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )
        assert model.name == "fake data"
        assert model.input_spaces[0].channels == test_input_chw[0]
        assert model.input_spaces[0].name == "input"
        assert model.input_spaces[0].shape == test_input_chw[1:]
        assert model.n_forecast_steps == test_n_forecast_steps
        assert model.n_history_steps == test_n_history_steps
        assert model.output_space.channels == test_output_chw[0]
        assert model.output_space.name == "target"
        assert model.output_space.shape == test_output_chw[1:]
        assert model.checkpoint_epoch is None

    def test_init_mask_dir_without_land_mask_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        model = FakeDataModel(
            name="fake data",
            input_spaces=[{"channels": 1, "name": "input", "shape": (2, 2)}],
            mask_dir=tmp_path,
            n_forecast_steps=1,
            n_history_steps=1,
            output_space={"channels": 1, "name": "target", "shape": (2, 2)},
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )
        assert getattr(model.train_metrics["accuracy"], "land_mask", None) is None

    def test_legacy_checkpoint_loads_with_replacement_metrics(
        self, tmp_path: Path
    ) -> None:
        """Checkpoints that saved plain metric names load when given new metrics."""
        model = FakeDataModel(
            name="fake data",
            input_spaces=[{"channels": 1, "name": "input", "shape": (2, 2)}],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space={"channels": 1, "name": "target", "shape": (2, 2)},
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )
        checkpoint_path = tmp_path / "legacy.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                # FakeDataModel supplies its own hemisphere, so it is not saved here
                "hyper_parameters": {
                    **{k: v for k, v in model.hparams.items() if k != "hemisphere"},
                    "metrics": ["accuracy", "mae"],
                },
                "pytorch-lightning_version": lightning.__version__,
            },
            checkpoint_path,
        )

        with pytest.raises(TypeError, match="must be a mapping"):
            FakeDataModel.load_from_checkpoint(checkpoint_path, weights_only=False)

        replacement = [metric_spec("mae", MAEPerForecastDay)]
        loaded = FakeDataModel.load_from_checkpoint(
            checkpoint_path, metrics=replacement, weights_only=False
        )
        assert set(loaded.test_metrics) == {"mae"}
        # Metric configs are supplied at load time, so are not saved in checkpoints
        assert "metrics" not in model.hparams
        assert "metrics" not in loaded.hparams

    def test_init_mask_dir_with_land_mask_is_used(self, tmp_path: Path) -> None:
        np.save(tmp_path / "land_mask.npy", np.ones((2, 2), dtype=np.uint8))
        model = FakeDataModel(
            name="fake data",
            input_spaces=[{"channels": 1, "name": "input", "shape": (2, 2)}],
            mask_dir=tmp_path,
            n_forecast_steps=1,
            n_history_steps=1,
            output_space={"channels": 1, "name": "target", "shape": (2, 2)},
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )
        land_mask = getattr(model.train_metrics["accuracy"], "land_mask")  # noqa: B009
        assert land_mask.all()
        # The land mask is passed to metrics without being written into their configs
        assert all("land_mask" not in spec for spec in model.metric_cfgs.values())
        assert all("land_mask" not in spec for spec in DEFAULT_METRICS)

    def test_loss(
        self, cfg_input_space: DictConfig, cfg_output_space: DictConfig
    ) -> None:
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )
        # Test loss
        prediction = torch.zeros(1, 1, 1, 1)
        target = torch.ones(1, 1, 1, 1)
        assert model.loss(prediction, target) == torch.tensor(0.375)

    def test_optimizer(
        self,
        cfg_input_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )
        opt_sched_cfg = model.configure_optimizers()
        assert isinstance(opt_sched_cfg, dict)
        optimizer = opt_sched_cfg.get("optimizer", None)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 5e-4

    def test_scheduler(
        self,
        cfg_input_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_output_space: DictConfig,
        cfg_scheduler: DictConfig,
    ) -> None:
        # Use non-default values so that the test fails if the config is ignored
        lr_scheduler = DictConfig(
            {"frequency": 2, "interval": "step", "monitor": "validation_loss"}
        )
        model = FakeDataModel(
            name="dummy",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=cfg_scheduler,
            lr_scheduler=lr_scheduler,
        )
        opt_sched_cfg = model.configure_optimizers()
        assert isinstance(opt_sched_cfg, dict)
        lr_scheduler_cfg = opt_sched_cfg.get("lr_scheduler", None)
        assert isinstance(lr_scheduler_cfg, dict)
        scheduler = lr_scheduler_cfg.get("scheduler", None)
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
        assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)
        assert scheduler.start_factor == 0.2
        assert scheduler.end_factor == 0.8
        assert lr_scheduler_cfg.get("frequency") == 2
        assert lr_scheduler_cfg.get("interval") == "step"
        assert lr_scheduler_cfg.get("monitor") == "validation_loss"

    @pytest.mark.parametrize(
        "step_name",
        ["test_step", "training_step", "validation_step"],
        ids=["test_step", "training_step", "validation_step"],
    )
    def test_step_output_shapes(
        self,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        *,
        step_name: str,
    ) -> None:
        batch_size = n_history_steps = n_forecast_steps = 1
        batch = {
            cfg_input_space["name"]: torch.randn(
                batch_size,
                n_history_steps,
                cfg_input_space["channels"],
                cfg_input_space["shape"][0],
                cfg_input_space["shape"][1],
            ),
            cfg_output_space["name"]: torch.randn(
                batch_size,
                n_forecast_steps,
                cfg_output_space["channels"],
                cfg_output_space["shape"][0],
                cfg_output_space["shape"][1],
            ),
        }
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=cfg_scheduler,
            lr_scheduler=DictConfig({}),
        )
        output_shape = batch["target"].shape
        output = getattr(model, step_name)(batch, 0)
        assert isinstance(output, ModelStepOutput)
        assert output.prediction.shape == output_shape
        assert output.target.shape == output_shape
        assert output.loss.shape == torch.Size([])

    @pytest.mark.parametrize(
        "has_climatology", [True, False], ids=["climatology", "no-climatology"]
    )
    def test_test_step_updates_climatology_metrics(
        self,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
        *,
        has_climatology: bool,
    ) -> None:
        """The climatology baseline is only accumulated from batches that carry it."""
        output_shape = (
            1,
            1,
            cfg_output_space["channels"],
            *cfg_output_space["shape"],
        )
        batch = {
            cfg_input_space["name"]: torch.randn(
                1, 1, cfg_input_space["channels"], *cfg_input_space["shape"]
            ),
            cfg_output_space["name"]: torch.rand(output_shape),
        }
        if has_climatology:
            batch["climatology"] = torch.rand(output_shape)
        model = FakeDataModel(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=cfg_optimizer,
            scheduler=cfg_scheduler,
            lr_scheduler=DictConfig({}),
        )

        model.test_step(batch, 0)

        assert model.test_metrics["mae"].update_called
        assert model.climatology_metrics["mae"].update_called is has_climatology
        assert set(model.climatology_metrics) == set(model.test_metrics)

    def test_test_step_two_channel_target_skips_single_channel_metrics(
        self,
        caplog: pytest.LogCaptureFixture,
        cfg_input_space: DictConfig,
        cfg_optimizer: DictConfig,
        cfg_scheduler: DictConfig,
    ) -> None:
        """A second target channel must not be folded into single-channel metrics."""
        two_channel_output_space = DictConfig(
            {"channels": 2, "name": "target", "shape": (16, 16)}
        )
        batch_size = n_history_steps = n_forecast_steps = 1
        batch = {
            cfg_input_space["name"]: torch.randn(
                batch_size,
                n_history_steps,
                cfg_input_space["channels"],
                cfg_input_space["shape"][0],
                cfg_input_space["shape"][1],
            ),
            two_channel_output_space["name"]: torch.randn(
                batch_size,
                n_forecast_steps,
                two_channel_output_space["channels"],
                two_channel_output_space["shape"][0],
                two_channel_output_space["shape"][1],
            ),
        }
        with caplog.at_level(logging.WARNING):
            model = FakeDataModel(
                name="fake data",
                input_spaces=[cfg_input_space],
                n_forecast_steps=n_forecast_steps,
                n_history_steps=n_history_steps,
                output_space=two_channel_output_space,
                optimizer=cfg_optimizer,
                scheduler=cfg_scheduler,
                lr_scheduler=DictConfig({}),
            )
        assert "Disabling single-channel metrics for FakeDataModel" in caplog.text
        assert "accuracy" in caplog.text
        assert set(model.test_metrics.keys()) == {
            "mae",
            "rmse",
            "ssim",
            "spatial_mean_ground_truth",
            "spatial_mean_prediction",
        }
        # The requested list is kept so that it can be passed on unchanged
        assert "accuracy" in model.metric_cfgs
        output = model.test_step(batch, 0)
        assert isinstance(output, ModelStepOutput)


class TestBaseModelMetricSelection:
    @staticmethod
    def _build_model(metrics: list[Any]) -> FakeDataModel:
        return FakeDataModel(
            name="fake data",
            input_spaces=[{"channels": 1, "name": "input", "shape": (2, 2)}],
            metrics=metrics,
            n_forecast_steps=1,
            n_history_steps=1,
            output_space={"channels": 1, "name": "target", "shape": (2, 2)},
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )

    def test_default_config_builds_every_metric(self) -> None:
        model = self._build_model(DEFAULT_METRICS)

        assert set(model.train_metrics.keys()) == set(model.metric_cfgs)
        assert len(model.metric_cfgs) == 13

    @pytest.mark.parametrize(
        ("metric_name", "metric_type"),
        list(NON_FSS_METRIC_TYPES.items()),
        ids=list(NON_FSS_METRIC_TYPES.keys()),
    )
    def test_non_fss_metric_selection(
        self, metric_name: str, metric_type: type
    ) -> None:
        model = self._build_model([metric_spec(metric_name, metric_type)])

        assert set(model.train_metrics.keys()) == {metric_name}
        assert isinstance(model.train_metrics[metric_name], metric_type)

    @pytest.mark.parametrize("neighbourhood_size", [1, 3, 7, 15])
    def test_fss_metric_neighbourhood_size_from_config(
        self, neighbourhood_size: int
    ) -> None:
        spec = metric_spec(
            "fss",
            FractionalSkillScorePerForecastDay,
            neighbourhood_size=neighbourhood_size,
        )
        model = self._build_model([spec])

        metric = model.train_metrics["fss"]
        assert isinstance(metric, FractionalSkillScorePerForecastDay)
        assert metric.neighbourhood_size == neighbourhood_size

    def test_multiple_metrics_are_all_present_and_exclusive(self) -> None:
        specs = [
            metric_spec("accuracy", IceNetAccuracyPerForecastDay),
            metric_spec("rmse", RMSEPerForecastDay),
            metric_spec(
                "fss_7", FractionalSkillScorePerForecastDay, neighbourhood_size=7
            ),
            metric_spec("ssim", SSIMPerForecastDay),
        ]
        model = self._build_model(specs)

        assert set(model.train_metrics.keys()) == {"accuracy", "rmse", "fss_7", "ssim"}

    def test_metric_collections_are_built_identically(self) -> None:
        """train/test/validation metrics are independent copies of one selection."""
        model = self._build_model(
            [
                metric_spec("accuracy", IceNetAccuracyPerForecastDay),
                metric_spec("mae", MAEPerForecastDay),
            ]
        )

        assert (
            set(model.train_metrics.keys())
            == set(model.test_metrics.keys())
            == set(model.validation_metrics.keys())
        )
        assert model.train_metrics["mae"] is not model.test_metrics["mae"]

    def test_empty_metrics_list_builds_no_metrics(self) -> None:
        model = self._build_model([])

        assert set(model.train_metrics.keys()) == set()

    def test_model_metric_names_and_specs(self) -> None:
        specs = [
            metric_spec("accuracy", IceNetAccuracyPerForecastDay),
            metric_spec(
                "fss_9", FractionalSkillScorePerForecastDay, neighbourhood_size=9
            ),
        ]
        model = self._build_model(specs)

        assert list(model.metric_cfgs) == ["accuracy", "fss_9"]
        # The original config entries are kept so that they can be passed on unchanged
        assert list(model.metric_cfgs.values()) == specs

    def test_metric_accepts_dictconfig(self) -> None:
        spec = OmegaConf.create(metric_spec("my_mae", MAEPerForecastDay))
        model = self._build_model([spec])

        assert isinstance(model.train_metrics["my_mae"], MAEPerForecastDay)

    def test_metric_without_land_mask_mixin_is_not_given_land_mask(self) -> None:
        """Non-LandMaskMixin metrics need not accept a land_mask argument."""
        model = self._build_model([metric_spec("mse", MeanSquaredError)])

        assert isinstance(model.train_metrics["mse"], MeanSquaredError)

    @pytest.mark.parametrize(
        "spec",
        [
            "mae",
            {"_target_": "icenet_mp.metrics.MAEPerForecastDay"},
            {"name": "mae"},
            {"name": 1, "_target_": "icenet_mp.metrics.MAEPerForecastDay"},
        ],
        ids=["plain-name", "no-name", "no-target", "non-string-name"],
    )
    def test_invalid_metric_config_raises(self, spec: object) -> None:
        with pytest.raises(
            TypeError, match="must be a mapping with 'name' and '_target_' keys"
        ):
            self._build_model([spec])

    def test_duplicate_metric_name_raises(self) -> None:
        specs = [
            metric_spec("mae", MAEPerForecastDay),
            metric_spec("mae", RMSEPerForecastDay),
        ]
        with pytest.raises(ValueError, match="'mae' is configured more than once"):
            self._build_model(specs)

    def test_fss_even_neighbourhood_size_raises(self) -> None:
        spec = metric_spec(
            "fss", FractionalSkillScorePerForecastDay, neighbourhood_size=4
        )
        with pytest.raises(InstantiationException, match="positive odd integer"):
            self._build_model([spec])


class TestBaseModelMetricAccumulation:
    @staticmethod
    def _score_batches(
        stage: str,
        metric_names: list[str],
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch_size: int,
        mask_dir: Path,
    ) -> dict[str, torch.Tensor]:
        """Accumulate metrics through the real model step for the selected stage."""
        model = EchoForecastModel(
            hemisphere=Hemisphere.NORTH,
            input_spaces=[
                DictConfig({"name": "forecast", "channels": 1, "shape": [1, 1]})
            ],
            output_space=DictConfig({"name": "target", "channels": 1, "shape": [1, 1]}),
            mask_dir=mask_dir,
            n_history_steps=2,
            n_forecast_steps=2,
            name="echo forecast",
            metrics=[
                metric_spec(name, NON_FSS_METRIC_TYPES[name]) for name in metric_names
            ],
            loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
        )
        step, metrics = {
            "train": (model.training_step, model.train_metrics),
            "validation": (model.validation_step, model.validation_metrics),
            "test": (model.test_step, model.test_metrics),
        }[stage]
        for batch_idx, start in enumerate(range(0, prediction.shape[0], batch_size)):
            step(
                {
                    "forecast": prediction[start : start + batch_size],
                    "target": target[start : start + batch_size],
                },
                batch_idx,
            )
        return metrics.compute()

    @pytest.mark.parametrize(
        ("prediction", "target", "expected_scores"),
        [
            # Equal overprediction and underprediction cancel only the signed error:
            # one 25 km by 25 km ocean cell disagrees in each sample and forecast lead.
            (
                torch.tensor([1.0, 0.0]).reshape(2, 1, 1, 1, 1).repeat(1, 2, 1, 1, 1),
                torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1, 1).repeat(1, 2, 1, 1, 1),
                [
                    (
                        "iiee",
                        IntegratedIceEdgeErrorPerForecastDay,
                        torch.full((2,), 625.0),
                    ),
                    ("sieerror", SeaIceExtentErrorPerForecastDay, torch.zeros(2)),
                ],
            ),
            # MAE and RMSE must retain their different update rules across batches.
            (
                torch.tensor([1.0, 0.5]).reshape(2, 1, 1, 1, 1).repeat(1, 2, 1, 1, 1),
                torch.zeros(2, 2, 1, 1, 1),
                [
                    ("mae", MAEPerForecastDay, torch.full((2,), 0.75)),
                    ("rmse", RMSEPerForecastDay, torch.full((2,), 0.625).sqrt()),
                ],
            ),
        ],
        ids=["extent-errors", "mae-rmse"],
    )
    @pytest.mark.parametrize(
        "stage", ["train", "validation", "test"], ids=lambda s: f"stage-{s}"
    )
    @pytest.mark.parametrize("batch_size", [1, 2], ids=lambda n: f"batch_size-{n}")
    def test_metrics_are_independent_of_batch_size(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        expected_scores: list[tuple[str, type[Metric], torch.Tensor]],
        stage: str,
        batch_size: int,
        tmp_path: Path,
    ) -> None:
        """Metrics accumulated over batches must match a single full-data update."""
        scores = self._score_batches(
            stage,
            [name for name, _, _ in expected_scores],
            prediction,
            target,
            batch_size,
            tmp_path,
        )

        for name, metric_type, expected in expected_scores:
            metric = metric_type()
            metric.update(prediction, target)
            assert torch.allclose(metric.compute(), expected)
            assert torch.allclose(scores[name], expected)


class TestBaseModelLossConfig:
    """Tests that the loss function config is correctly picked up by BaseModel."""

    def test_missing_loss_raises(
        self, cfg_input_space: DictConfig, cfg_output_space: DictConfig
    ) -> None:
        with pytest.raises(TypeError, match=r"argument: 'loss'"):
            Persistence(
                target_variable_indices=[0],
                hemisphere=Hemisphere.NORTH,
                name="persistence",
                input_spaces=[cfg_input_space],
                n_forecast_steps=1,
                n_history_steps=1,
                output_space=cfg_output_space,
                optimizer=DictConfig({}),
                scheduler=DictConfig({}),
                lr_scheduler=DictConfig({}),
                metrics=[],
            )

    @pytest.mark.parametrize(
        ("loss_cfg", "loss_type"),
        [
            pytest.param(
                OmegaConf.create({"_target_": "torch.nn.MSELoss"}),
                torch.nn.MSELoss,
                id="mse",
            ),
            pytest.param(
                OmegaConf.create({"_target_": "torch.nn.L1Loss"}),
                torch.nn.L1Loss,
                id="mae",
            ),
            pytest.param(
                OmegaConf.create({"_target_": "torch.nn.HuberLoss", "delta": 0.5}),
                torch.nn.HuberLoss,
                id="huber",
            ),
            pytest.param(
                OmegaConf.create({"_target_": "torch.nn.SmoothL1Loss", "beta": 0.5}),
                torch.nn.SmoothL1Loss,
                id="smooth_l1",
            ),
            pytest.param(
                OmegaConf.create({"_target_": "icenet_mp.losses.rmse_loss.RMSELoss"}),
                RMSELoss,
                id="rmse",
            ),
            pytest.param(
                OmegaConf.create({"_target_": "icenet_mp.losses.amse_loss.AMSELoss"}),
                AMSELoss,
                id="amse",
            ),
            pytest.param(
                OmegaConf.create(
                    {
                        "_target_": "icenet_mp.losses.lead_time_weighted_loss.LeadTimeWeightedLoss",
                        "wrapped_loss": {
                            "_target_": "torch.nn.HuberLoss",
                            "delta": 0.5,
                        },
                        "lead_time_exponent": 2.0,
                    }
                ),
                LeadTimeWeightedLoss,
                id="lead_time_weighted",
            ),
            pytest.param(
                OmegaConf.create(
                    {
                        "_target_": (
                            "icenet_mp.losses.weighted_bce_loss.WeightedBCEWithLogitsLoss"
                        )
                    }
                ),
                WeightedBCEWithLogitsLoss,
                id="weighted_bce",
            ),
            pytest.param(
                OmegaConf.create(
                    {"_target_": "icenet_mp.losses.weighted_l1_loss.WeightedL1Loss"}
                ),
                WeightedL1Loss,
                id="weighted_l1",
            ),
            pytest.param(
                OmegaConf.create(
                    {"_target_": "icenet_mp.losses.weighted_mse_loss.WeightedMSELoss"}
                ),
                WeightedMSELoss,
                id="weighted_mse",
            ),
        ],
    )
    def test_loss_type(
        self,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        *,
        loss_cfg: DictConfig,
        loss_type: type[torch.nn.Module],
    ) -> None:
        model = Persistence(
            target_variable_indices=[0],
            hemisphere=Hemisphere.NORTH,
            name="persistence",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            loss=loss_cfg,
            metrics=[],
        )
        assert isinstance(model.loss_fn, loss_type)

    @pytest.mark.parametrize("struct", [False, True], ids=["plain", "struct"])
    def test_lead_time_exponent_wraps_loss(
        self,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
        *,
        struct: bool,
    ) -> None:
        loss_cfg = OmegaConf.create(
            {"_target_": "torch.nn.HuberLoss", "delta": 0.5, "lead_time_exponent": 2.0}
        )
        OmegaConf.set_struct(loss_cfg, struct)
        model = Persistence(
            target_variable_indices=[0],
            hemisphere=Hemisphere.NORTH,
            name="persistence",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            loss=loss_cfg,
            metrics=[],
        )
        assert isinstance(model.loss_fn, LeadTimeWeightedLoss)
        assert model.loss_fn.exponent == pytest.approx(2.0)
        assert isinstance(model.loss_fn.wrapped_loss, torch.nn.HuberLoss)
        assert model.loss_fn.wrapped_loss.delta == pytest.approx(0.5)
        # The stored config retains the exponent so that checkpoints round-trip
        assert model.loss_cfg.lead_time_exponent == pytest.approx(2.0)

    def test_nonexistent_loss_raises(
        self, cfg_input_space: DictConfig, cfg_output_space: DictConfig
    ) -> None:
        bad_loss = OmegaConf.create(
            {"_target_": "icenet_mp.losses.does_not_exist.FakeLoss"}
        )
        with pytest.raises(
            InstantiationException,
            match=r"Error locating target 'icenet_mp\.losses\.does_not_exist\.FakeLoss'",
        ):
            Persistence(
                target_variable_indices=[0],
                hemisphere=Hemisphere.NORTH,
                name="persistence",
                input_spaces=[cfg_input_space],
                n_forecast_steps=1,
                n_history_steps=1,
                output_space=cfg_output_space,
                optimizer=DictConfig({}),
                scheduler=DictConfig({}),
                lr_scheduler=DictConfig({}),
                loss=bad_loss,
                metrics=[],
            )
