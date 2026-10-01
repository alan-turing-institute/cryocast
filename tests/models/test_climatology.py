from typing import Any

import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models import Climatology


class TestClimatology:
    @pytest.mark.parametrize(
        "test_output_shape", [(16, 16, 1), (10, 20, 19)], ids=["16x16x1", "10x20x19"]
    )
    def test_forward_returns_climatology(
        self,
        test_output_shape: tuple[int, int, int],
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        # Climatology only reads the "climatology" batch entry, so the "input"
        # space shape, history length and batch size have no effect on the result.
        batch_size, n_forecast_steps, n_history_steps = 2, 3, 2
        input_space = {
            "channels": 1,
            "name": "input",
            "shape": (16, 16),
        }
        output_space = {
            "channels": test_output_shape[2],
            "name": "target",
            "shape": test_output_shape[0:2],
        }
        model = Climatology(
            name="climatology",
            hemisphere="north",
            input_spaces=[input_space],
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
            output_space=output_space,
            optimizer={},
            scheduler={},
            lr_scheduler={},
            target_variable_indices=list(range(test_output_shape[2])),
        )
        batch = {
            "input": torch.randn(batch_size, n_history_steps, 1, 16, 16),
            "climatology": torch.randn(
                batch_size,
                n_forecast_steps,
                test_output_shape[2],
                test_output_shape[0],
                test_output_shape[1],
            ),
        }
        result: torch.Tensor = model(batch)
        assert torch.equal(result, batch["climatology"])

    def test_optimizer(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        model = Climatology(
            name="climatology",
            hemisphere="north",
            input_spaces=[
                {
                    "channels": 1,
                    "name": "input",
                    "shape": (1, 1),
                }
            ],
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=1,
            n_history_steps=1,
            output_space={
                "channels": 1,
                "name": "target",
                "shape": (1, 1),
            },
            optimizer={},
            scheduler={},
            lr_scheduler={},
            target_variable_indices=[0],
        )
        assert model.configure_optimizers() is None, (
            "No optimizer should be initialized"
        )
