import pytest
import torch
from omegaconf import DictConfig

from icenet_mp.models import Climatology


class TestClimatology:
    @pytest.mark.parametrize("test_input_shape", [(16, 16, 4), (20, 20, 1)])
    @pytest.mark.parametrize("test_output_shape", [(16, 16, 1), (10, 20, 19)])
    @pytest.mark.parametrize("test_batch_size", [1, 2])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 2, 5])
    @pytest.mark.parametrize("test_n_history_steps", [1, 2, 5])
    def test_forward_returns_climatology(
        self,
        test_batch_size: int,
        test_input_shape: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_output_shape: tuple[int, int, int],
        cfg_loss: DictConfig,
        cfg_metrics: list[str],
    ) -> None:
        input_space = {
            "channels": test_input_shape[2],
            "name": "input",
            "shape": test_input_shape[0:2],
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
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            output_space=output_space,
            optimizer={},
            scheduler={},
            lr_scheduler={},
            target_variable_indices=list(range(test_output_shape[2])),
        )
        batch = {
            "input": torch.randn(
                test_batch_size,
                test_n_history_steps,
                test_input_shape[2],
                test_input_shape[0],
                test_input_shape[1],
            ),
            "target": torch.randn(
                test_batch_size,
                test_n_forecast_steps,
                test_output_shape[2],
                test_output_shape[0],
                test_output_shape[1],
            ),
            "climatology": torch.randn(
                test_batch_size,
                test_n_forecast_steps,
                test_output_shape[2],
                test_output_shape[0],
                test_output_shape[1],
            ),
        }
        result: torch.Tensor = model(batch)
        assert result.shape == batch["target"].shape
        assert torch.equal(result, batch["climatology"])

    def test_optimizer(self, cfg_loss: DictConfig, cfg_metrics: list[str]) -> None:
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
