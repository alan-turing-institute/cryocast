from typing import Any

import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import DictConfig, OmegaConf

from icenet_mp.losses.amse_loss import AMSELoss
from icenet_mp.losses.rmse_loss import RMSELoss
from icenet_mp.losses.weighted_bce_loss import WeightedBCEWithLogitsLoss
from icenet_mp.losses.weighted_l1_loss import WeightedL1Loss
from icenet_mp.losses.weighted_mse_loss import WeightedMSELoss
from icenet_mp.models import BaseModel
from icenet_mp.types import Hemisphere, TensorNTCHW


class FakeDataModelNoDefault(BaseModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise a fake data model with no default loss for testing purposes."""
        super().__init__(*args, hemisphere=Hemisphere.NORTH, **kwargs)
        self.model = torch.nn.Linear(1, 1)

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        b = next(iter(inputs.values())).shape[0]
        return torch.randn(b, 1, 1, 1, 1)


LOSS_CONFIGS = {
    "mse": OmegaConf.create({"_target_": "torch.nn.MSELoss"}),
    "mae": OmegaConf.create({"_target_": "torch.nn.L1Loss"}),
    "huber": OmegaConf.create({"_target_": "torch.nn.HuberLoss", "delta": 0.5}),
    "smooth_l1": OmegaConf.create({"_target_": "torch.nn.SmoothL1Loss", "beta": 0.5}),
    "rmse": OmegaConf.create({"_target_": "icenet_mp.losses.rmse_loss.RMSELoss"}),
    "amse": OmegaConf.create({"_target_": "icenet_mp.losses.amse_loss.AMSELoss"}),
    "weighted_bce": OmegaConf.create(
        {"_target_": "icenet_mp.losses.weighted_bce_loss.WeightedBCEWithLogitsLoss"}
    ),
    "weighted_l1": OmegaConf.create(
        {"_target_": "icenet_mp.losses.weighted_l1_loss.WeightedL1Loss"}
    ),
    "weighted_mse": OmegaConf.create(
        {"_target_": "icenet_mp.losses.weighted_mse_loss.WeightedMSELoss"}
    ),
}

LOSS_TYPES = {
    "mse": torch.nn.MSELoss,
    "mae": torch.nn.L1Loss,
    "huber": torch.nn.HuberLoss,
    "smooth_l1": torch.nn.SmoothL1Loss,
    "rmse": RMSELoss,
    "amse": AMSELoss,
    "weighted_bce": WeightedBCEWithLogitsLoss,
    "weighted_l1": WeightedL1Loss,
    "weighted_mse": WeightedMSELoss,
}


class TestLossConfig:
    """Tests that the loss function config is correctly picked up by BaseModel."""

    # 1. Model cannot be initialised without a loss section
    def test_missing_loss_raises(
        self, cfg_input_space: DictConfig, cfg_output_space: DictConfig
    ) -> None:
        with pytest.raises(TypeError):
            FakeDataModelNoDefault(
                name="fake data",
                input_spaces=[cfg_input_space],
                n_forecast_steps=1,
                n_history_steps=1,
                output_space=cfg_output_space,
                optimizer=DictConfig({}),
                scheduler=DictConfig({}),
                lr_scheduler=DictConfig({}),
                # loss intentionally omitted
            )

    # 2. Initialising with loss X creates a model that uses loss X
    @pytest.mark.parametrize("loss_name", list(LOSS_CONFIGS.keys()))
    def test_loss_type(
        self,
        loss_name: str,
        cfg_input_space: DictConfig,
        cfg_output_space: DictConfig,
    ) -> None:
        model = FakeDataModelNoDefault(
            name="fake data",
            input_spaces=[cfg_input_space],
            n_forecast_steps=1,
            n_history_steps=1,
            output_space=cfg_output_space,
            optimizer=DictConfig({}),
            scheduler=DictConfig({}),
            lr_scheduler=DictConfig({}),
            loss=LOSS_CONFIGS[loss_name],
            metrics=[],
        )
        assert isinstance(model.loss_fn, LOSS_TYPES[loss_name])

    # 3. Initialising with a loss function that doesn't exist fails
    def test_nonexistent_loss_raises(
        self, cfg_input_space: DictConfig, cfg_output_space: DictConfig
    ) -> None:
        bad_loss = OmegaConf.create(
            {"_target_": "icenet_mp.losses.does_not_exist.FakeLoss"}
        )
        # AFTER
        with pytest.raises(InstantiationException):
            FakeDataModelNoDefault(
                name="fake data",
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
