import pytest
import torch
from omegaconf import DictConfig

from cryocast.models import BaseModel
from cryocast.types import Hemisphere, TensorNTCHW


class IdentityModel(BaseModel):
    """Minimal model used to exercise BaseModel loss dispatch."""

    def forward(self, inputs: dict[str, TensorNTCHW]) -> TensorNTCHW:
        return inputs["input"]


def _make_model() -> IdentityModel:
    return IdentityModel(
        hemisphere=Hemisphere.NORTH,
        input_spaces=[DictConfig({"name": "input", "channels": 1, "shape": (2, 2)})],
        loss=DictConfig(
            {
                "_target_": "cryocast.losses.UncertaintyWeightedLoss",
                "delta": 0.5,
            }
        ),
        metrics=[],
        n_forecast_steps=1,
        n_history_steps=1,
        name="identity",
        optimizer=DictConfig({}),
        output_space=DictConfig({"name": "target", "channels": 1, "shape": (2, 2)}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
    )


def test_base_model_dispatches_uncertainty_to_loss() -> None:
    """Dispatch uncertainty tensors to uncertainty-aware losses."""
    model = _make_model()
    prediction = torch.tensor([[[[[1.0, 2.0], [0.0, 0.0]]]]])
    target = torch.zeros_like(prediction)
    uncertainty = torch.tensor([[[[[0.1, 0.2], [0.1, 0.1]]]]])

    expected = model.loss_fn(prediction, target, uncertainty)

    torch.testing.assert_close(model.loss(prediction, target, uncertainty), expected)


def test_base_model_reports_missing_uncertainty() -> None:
    """Raise a clear error when required target uncertainty is absent."""
    model = _make_model()
    prediction = torch.zeros(1, 1, 1, 2, 2)

    with pytest.raises(ValueError, match="requires target uncertainty"):
        model.loss(prediction, prediction)
