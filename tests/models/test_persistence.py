from typing import Any

import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models import Persistence


class TestPersistence:
    @pytest.mark.parametrize(
        "test_target_variable_indices",
        [[0], [1, 3], [0, 1, 2, 3]],
        ids=lambda indices: f"indices={indices}",
    )
    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=["batch1", "batch2"])
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 2, 5], ids=["forecast1", "forecast2", "forecast5"]
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 2, 5], ids=["history1", "history2", "history5"]
    )
    def test_forward_repeats_last_history_frame(
        self,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_target_variable_indices: list[int],
        cfg_loss: DictConfig,
        cfg_metrics: list[dict[str, Any]],
    ) -> None:
        """Every lead is the last observed frame of the selected target variables."""
        # The target group is a 4-channel input, of which only some are predicted
        target_group = {"channels": 4, "name": "sic", "shape": (10, 20)}
        other_group = {"channels": 2, "name": "era5", "shape": (10, 20)}
        output_space = {
            "channels": len(test_target_variable_indices),
            "name": "sic",
            "shape": (10, 20),
        }
        model = Persistence(
            name="persistence",
            hemisphere="north",
            input_spaces=[target_group, other_group],
            loss=cfg_loss,
            metrics=cfg_metrics,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=output_space,
            optimizer={},
            scheduler={},
            lr_scheduler={},
            target_variable_indices=test_target_variable_indices,
        )
        batch = {
            "sic": torch.rand(test_batch_size, test_n_history_steps, 4, 10, 20),
            "era5": torch.rand(test_batch_size, test_n_history_steps, 2, 10, 20),
        }

        result: torch.Tensor = model(batch)

        last_frame = batch["sic"][:, -1, test_target_variable_indices]
        expected = last_frame.unsqueeze(1).expand(-1, test_n_forecast_steps, -1, -1, -1)
        assert torch.equal(result, expected)

    def test_forward_ignores_climatology_key_and_has_no_optimizer(
        self, cfg_loss: DictConfig, cfg_metrics: list[dict[str, Any]]
    ) -> None:
        """An extra climatology batch key must not change the output; no optimizer is configured."""
        model = Persistence(
            name="persistence",
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
        batch_without = {
            "input": torch.randn(1, 1, 1, 1, 1),
            "target": torch.randn(1, 1, 1, 1, 1),
        }
        batch_with = {
            **batch_without,
            "climatology": torch.randn(1, 1, 1, 1, 1),
        }

        assert torch.equal(model(batch_without), model(batch_with))
        assert model.configure_optimizers() is None, (
            "No optimizer should be initialized"
        )
