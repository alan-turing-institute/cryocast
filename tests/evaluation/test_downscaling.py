from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from cryocast.evaluation import compare_downscaler
from cryocast.models import Downscaler


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
            DictConfig({"name": "sic-osisaf", "channels": 1, "shape": source_lat.shape})
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
        variable_names=DictConfig({"sic-osisaf": ["ice_conc"]}),
        hidden_channels=4,
        n_residual_blocks=1,
    )


def _batch(target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    source_lat, source_lon, _, _ = _coordinates()
    source = torch.from_numpy((source_lat + source_lon) / 4.0).float()
    source = source.view(1, 1, 1, 3, 3)
    if target is None:
        target = torch.tensor([[[[[0.2, 0.6], [0.4, 0.8]]]]], dtype=torch.float32)
    return {"sic-osisaf": source, "target": target}


class TestCompareDownscaler:
    def test_initial_model_matches_interpolation_metrics(self) -> None:
        model = _make_downscaler()

        comparison = compare_downscaler(model, [_batch()])

        assert comparison.batches == 1
        assert comparison.valid_values == 4
        assert comparison.interpolation.mae == pytest.approx(comparison.learned.mae)
        assert comparison.interpolation.rmse == pytest.approx(comparison.learned.rmse)
        assert comparison.interpolation.gradient_rmse == pytest.approx(
            comparison.learned.gradient_rmse
        )
        assert comparison.mae_improvement_percent == pytest.approx(0.0)

    def test_learned_residual_can_improve_over_interpolation(self) -> None:
        model = _make_downscaler()
        batch = _batch()
        source = batch["sic-osisaf"]
        interpolation = model.interpolation_baseline(source)
        target = torch.clamp(interpolation + 0.1, max=1.0)
        assert model.refiner.tail.bias is not None

        with torch.no_grad():
            model.refiner.tail.bias.fill_(0.4)

        comparison = compare_downscaler(
            model,
            [{"sic-osisaf": source, "target": target}],
        )

        assert comparison.learned.mae < comparison.interpolation.mae
        assert comparison.mae_improvement_percent is not None
        assert comparison.mae_improvement_percent > 0

    def test_persistent_mask_is_excluded(self, tmp_path: Path) -> None:
        np.save(tmp_path / "land_mask.npy", np.array([[1, 1], [0, 1]], dtype=np.uint8))
        model = _make_downscaler(tmp_path)
        target = torch.tensor([[[[[0.2, 0.6], [100.0, 0.8]]]]])

        comparison = compare_downscaler(model, [_batch(target)])

        assert comparison.valid_values == 3

    def test_nan_target_cell_is_excluded(self) -> None:
        model = _make_downscaler()
        target = torch.tensor([[[[[0.2, float("nan")], [0.4, 0.8]]]]])

        comparison = compare_downscaler(model, [_batch(target)])

        assert comparison.valid_values == 3

    def test_requires_source_and_target(self) -> None:
        model = _make_downscaler()

        with pytest.raises(KeyError, match="source group"):
            compare_downscaler(model, [{"target": torch.zeros(1, 1, 1, 2, 2)}])

    @pytest.mark.parametrize("cutoff", [0.0, 1.0])
    def test_rejects_invalid_high_frequency_cutoff(self, cutoff: float) -> None:
        model = _make_downscaler()

        with pytest.raises(ValueError, match="between 0 and 1"):
            compare_downscaler(model, [_batch()], high_frequency_cutoff=cutoff)

    def test_max_batches_limits_evaluation(self) -> None:
        model = _make_downscaler()

        comparison = compare_downscaler(model, [_batch(), _batch()], max_batches=1)

        assert comparison.batches == 1
