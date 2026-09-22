from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from cryocast.models import Downscaler, DownscalingPipeline


def _coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_lat = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    source_lon = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    target_lat = np.array([[1.5, 1.5], [0.5, 0.5]])
    target_lon = np.array([[0.5, 1.5], [0.5, 1.5]])
    return source_lat, source_lon, target_lat, target_lon


def _make_downscaler(mask_dir: Path | None = None) -> Downscaler:
    source_lat, source_lon, target_lat, target_lon = _coordinates()
    return Downscaler(
        hemisphere="north",
        input_spaces=[
            DictConfig(
                {
                    "name": "sic-osisaf",
                    "channels": 2,
                    "shape": source_lat.shape,
                }
            )
        ],
        latitudes_fn=lambda: {
            "sic-osisaf": source_lat.ravel().tolist(),
            "sic-carra2": target_lat.ravel().tolist(),
        },
        longitudes_fn=lambda: {
            "sic-osisaf": source_lon.ravel().tolist(),
            "sic-carra2": target_lon.ravel().tolist(),
        },
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        mask_dir=mask_dir,
        lr_scheduler=DictConfig({}),
        metrics=[],
        n_forecast_steps=1,
        n_history_steps=1,
        name="downscaler",
        optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
        output_space=DictConfig(
            {"name": "sic-carra2", "channels": 1, "shape": target_lat.shape}
        ),
        scheduler=DictConfig({}),
        source_group_name="sic-osisaf",
        source_variable="ice_conc",
        source_crs="EPSG:4326",
        variable_names=DictConfig(
            {
                "sic-osisaf": ["total_standard_uncertainty", "ice_conc"],
                "sic-carra2": ["ice_conc"],
            }
        ),
        hidden_channels=4,
        n_residual_blocks=1,
    )


class TestDownscaler:
    """Tests for trainable residual spatial downscaling."""

    def test_starts_at_geographic_interpolation_baseline(self) -> None:
        """Zero-initialised residuals make the initial model equal interpolation."""
        model = _make_downscaler()
        source_lat, source_lon, _, _ = _coordinates()
        low_resolution = torch.from_numpy((source_lat + source_lon) / 4.0).float()
        low_resolution = low_resolution.view(1, 1, 1, 3, 3)

        baseline = model.interpolation_baseline(low_resolution)
        downscaled = model.downscale(low_resolution)

        torch.testing.assert_close(downscaled, baseline)

    def test_forward_selects_source_variable_by_name(self) -> None:
        """The configured SIC channel is selected even when it is not channel zero."""
        model = _make_downscaler()
        source_lat, source_lon, target_lat, target_lon = _coordinates()
        sic = torch.from_numpy((source_lat + source_lon) / 4.0).float()
        irrelevant = torch.full_like(sic, 42.0)
        inputs = torch.stack((irrelevant, sic), dim=0).view(1, 1, 2, 3, 3)

        result = model({"sic-osisaf": inputs})

        expected = torch.from_numpy((target_lat + target_lon) / 4.0).float()
        torch.testing.assert_close(result[0, 0, 0], expected)

    def test_residual_refiner_receives_training_gradient(self) -> None:
        """Paired high-resolution targets provide a gradient to the residual model."""
        model = _make_downscaler()
        source_lat, source_lon, _, _ = _coordinates()
        sic = torch.from_numpy((source_lat + source_lon) / 4.0).float()
        irrelevant = torch.zeros_like(sic)
        inputs = torch.stack((irrelevant, sic), dim=0).view(1, 1, 2, 3, 3)
        prediction = model({"sic-osisaf": inputs})
        target = torch.clamp(prediction.detach() + 0.1, max=1.0)

        model.loss(prediction, target).backward()

        assert model.refiner.tail.weight.grad is not None
        assert torch.any(model.refiner.tail.weight.grad != 0)

    def test_loss_ignores_nan_and_masked_target_cells(self, tmp_path: Path) -> None:
        """CARRA2 land/invalid cells do not contaminate the training objective."""
        np.save(tmp_path / "land_mask.npy", np.array([[1, 1], [0, 1]], dtype=np.uint8))
        model = _make_downscaler(tmp_path)
        prediction = torch.tensor([[[[[0.0, 0.0], [100.0, 0.0]]]]])
        target = torch.tensor([[[[[1.0, float("nan")], [0.0, 3.0]]]]])

        loss = model.loss(prediction, target)

        assert loss == pytest.approx(5.0)

    def test_downscaled_output_is_bounded(self) -> None:
        """Residual correction cannot produce invalid concentration values."""
        model = _make_downscaler()
        assert model.refiner.tail.bias is not None
        with torch.no_grad():
            model.refiner.tail.bias.fill_(100.0)
        low_resolution = torch.zeros((1, 1, 1, 3, 3))

        result = model.downscale(low_resolution)

        assert torch.all(result == 1.0)


class _Forecast(nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.prediction = prediction

    def forward(self, _inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.prediction


class TestDownscalingPipeline:
    """Tests for forecast-to-downscaler composition."""

    def test_forecast_prediction_is_downscaled(self) -> None:
        """All forecast lead times are passed through the trained downscaler."""
        downscaler = _make_downscaler()
        source_lat, source_lon, _, _ = _coordinates()
        field = torch.from_numpy((source_lat + source_lon) / 4.0).float()
        low_resolution_prediction = field.view(1, 1, 1, 3, 3).repeat(1, 3, 1, 1, 1)
        pipeline = DownscalingPipeline(
            _Forecast(low_resolution_prediction),
            downscaler,
        )

        result = pipeline({"unused": torch.empty(0)})

        assert result.shape == (1, 3, 1, 2, 2)
        torch.testing.assert_close(
            result,
            downscaler.interpolation_baseline(low_resolution_prediction),
        )
