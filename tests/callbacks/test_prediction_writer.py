from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from lightning import LightningModule, Trainer
from netCDF4 import Dataset as NetCDFDataset

from icenet_mp.callbacks import PredictionWriter
from icenet_mp.data import CombinedDataset
from icenet_mp.types import DataSpace


def _combined_dataset() -> CombinedDataset:
    dataset = cast("Any", object.__new__(CombinedDataset))
    dataset.n_forecast_steps = 2
    dataset.n_history_steps = 2
    dataset.frequency = np.timedelta64(1, "D")
    dataset.dates = [
        np.datetime64("2026-01-01T12:00:00"),
        np.datetime64("2026-01-02T12:00:00"),
        np.datetime64("2026-01-03T12:00:00"),
    ]
    dataset.target = SimpleNamespace(
        hemisphere="north",
        latitudes=[80.0, 80.0, 79.0, 79.0],
        longitudes=[0.0, 1.0, 0.0, 1.0],
        space=DataSpace(channels=1, name="sic-osisaf", shape=(2, 2)),
        statistics={
            "minimum": np.asarray([0.2], dtype=np.float32),
            "maximum": np.asarray([0.8], dtype=np.float32),
        },
        variable_names=["ice_conc"],
    )
    return cast("CombinedDataset", dataset)


def _trainer(dataset: CombinedDataset, *, world_size: int = 1) -> Any:  # noqa: ANN401
    return SimpleNamespace(
        test_dataloaders=SimpleNamespace(dataset=dataset),
        world_size=world_size,
    )


class TestPredictionWriter:
    def test_disabled_writer_is_a_noop(self) -> None:
        writer = PredictionWriter()

        writer.on_test_start(
            cast("Trainer", SimpleNamespace(world_size=4)), LightningModule()
        )
        writer.on_test_batch_end(
            cast("Trainer", SimpleNamespace()),
            LightningModule(),
            None,
            None,
            0,
        )
        writer.on_test_end(cast("Trainer", SimpleNamespace()), LightningModule())

    def test_rejects_enabled_writer_with_no_output_path(self) -> None:
        writer = PredictionWriter(enabled=True)

        with pytest.raises(RuntimeError, match="no output_path was set"):
            writer.on_test_start(
                _trainer(_combined_dataset()),
                LightningModule(),
            )

    def test_rejects_multi_process_export(self, tmp_path: Path) -> None:
        writer = PredictionWriter(enabled=True)
        writer.output_path = tmp_path / "predictions.nc"

        with pytest.raises(RuntimeError, match="single-process evaluation"):
            writer.on_test_start(
                _trainer(_combined_dataset(), world_size=2),
                LightningModule(),
            )

    def test_writes_denormalised_predictions_and_coordinates(
        self, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "predictions.nc"
        dataset = _combined_dataset()
        trainer = _trainer(dataset)
        writer = PredictionWriter(enabled=True)
        writer.output_path = output_path

        writer.on_test_start(trainer, LightningModule())

        first_batch = torch.full((2, 2, 1, 2, 2), 0.5, dtype=torch.float32)
        second_batch = torch.ones((1, 2, 1, 2, 2), dtype=torch.float32)

        writer.on_test_batch_end(
            trainer,
            LightningModule(),
            {"prediction": first_batch, "target": torch.zeros_like(first_batch)},
            None,
            0,
        )
        writer.on_test_batch_end(
            trainer,
            LightningModule(),
            {"prediction": second_batch, "target": torch.full_like(second_batch, 0.5)},
            None,
            1,
        )
        writer.on_test_end(trainer, LightningModule())

        with NetCDFDataset(output_path) as netcdf:
            assert netcdf.getncattr("Conventions") == "CF-1.10"
            assert netcdf.getncattr("hemisphere") == "north"
            assert len(netcdf.dimensions["forecast_reference_time"]) == 3
            assert len(netcdf.dimensions["lead_time"]) == 2

            prediction = np.asarray(netcdf.variables["ice_conc"][:])
            assert prediction.shape == (3, 2, 2, 2)
            assert np.allclose(prediction[:2], 0.5)
            assert np.allclose(prediction[2:], 0.8)
            assert netcdf.variables["ice_conc"].standard_name == "sea_ice_area_fraction"

            observed = np.asarray(netcdf.variables["ice_conc_observed"][:])
            assert observed.shape == (3, 2, 2, 2)
            assert np.allclose(observed[:2], 0.2)
            assert np.allclose(observed[2:], 0.5)
            assert (
                netcdf.variables["ice_conc_observed"].standard_name
                == "sea_ice_area_fraction"
            )
            assert (
                netcdf.variables["ice_conc_observed"].long_name
                == "observed sea ice concentration"
            )

            assert np.array_equal(
                np.asarray(netcdf.variables["lead_time"][:]),
                np.asarray([86400, 172800]),
            )

            reference = np.asarray(netcdf.variables["forecast_reference_time"][:])
            expected_reference = (
                np.asarray(
                    [
                        np.datetime64("2026-01-02T12:00:00"),
                        np.datetime64("2026-01-03T12:00:00"),
                        np.datetime64("2026-01-04T12:00:00"),
                    ],
                )
                .astype("datetime64[s]")
                .astype(np.int64)
            )
            assert np.array_equal(reference, expected_reference)

            valid = np.asarray(netcdf.variables["valid_time"][:])
            assert np.array_equal(
                valid[0],
                expected_reference[0] + np.asarray([86400, 172800]),
            )
            assert np.allclose(
                np.asarray(netcdf.variables["latitude"][:]),
                np.asarray([[80.0, 80.0], [79.0, 79.0]]),
            )
            assert np.allclose(
                np.asarray(netcdf.variables["longitude"][:]),
                np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            )

    def test_writes_available_masks(self, tmp_path: Path) -> None:
        output_path = tmp_path / "predictions.nc"
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        land_mask = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
        active_mask = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
        np.save(mask_dir / "land_mask.npy", land_mask)
        np.save(mask_dir / "active_mask.npy", active_mask)
        trainer = _trainer(_combined_dataset())
        writer = PredictionWriter(enabled=True)
        writer.output_path = output_path
        writer.mask_dir = mask_dir

        writer.on_test_start(trainer, LightningModule())
        writer.on_test_end(trainer, LightningModule())

        with NetCDFDataset(output_path) as netcdf:
            assert np.array_equal(netcdf.variables["land_mask"][:], land_mask)
            assert np.array_equal(netcdf.variables["active_mask"][:], active_mask)
            assert netcdf.variables["land_mask"].dimensions == ("y", "x")
            assert netcdf.variables["land_mask"].flag_meanings == "land ocean"
            assert np.array_equal(netcdf.variables["land_mask"].flag_values, [0, 1])
            for name in ("ice_conc", "ice_conc_observed"):
                assert (
                    netcdf.variables[name].ancillary_variables
                    == "land_mask active_mask"
                )

    def test_skips_missing_masks(self, tmp_path: Path) -> None:
        output_path = tmp_path / "predictions.nc"
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        np.save(mask_dir / "land_mask.npy", np.ones((2, 2), dtype=np.uint8))
        trainer = _trainer(_combined_dataset())
        writer = PredictionWriter(enabled=True)
        writer.output_path = output_path
        writer.mask_dir = mask_dir

        writer.on_test_start(trainer, LightningModule())
        writer.on_test_end(trainer, LightningModule())

        with NetCDFDataset(output_path) as netcdf:
            assert "land_mask" in netcdf.variables
            assert "active_mask" not in netcdf.variables
            assert netcdf.variables["ice_conc"].ancillary_variables == "land_mask"

    def test_writes_no_masks_without_mask_dir(self, tmp_path: Path) -> None:
        output_path = tmp_path / "predictions.nc"
        trainer = _trainer(_combined_dataset())
        writer = PredictionWriter(enabled=True)
        writer.output_path = output_path

        writer.on_test_start(trainer, LightningModule())
        writer.on_test_end(trainer, LightningModule())

        with NetCDFDataset(output_path) as netcdf:
            assert "land_mask" not in netcdf.variables
            assert "active_mask" not in netcdf.variables
            assert "ancillary_variables" not in netcdf.variables["ice_conc"].ncattrs()

    def test_rejects_mask_shape_mismatch(self, tmp_path: Path) -> None:
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        np.save(mask_dir / "land_mask.npy", np.ones((3, 3), dtype=np.uint8))
        writer = PredictionWriter(enabled=True)
        writer.output_path = tmp_path / "predictions.nc"
        writer.mask_dir = mask_dir

        with pytest.raises(ValueError, match="land mask shape"):
            writer.on_test_start(_trainer(_combined_dataset()), LightningModule())
        assert writer._file is None
        assert not writer.output_path.exists()

    def test_rejects_prediction_channel_mismatch(self, tmp_path: Path) -> None:
        dataset = _combined_dataset()
        trainer = _trainer(dataset)
        output_path = tmp_path / "predictions.nc"
        writer = PredictionWriter(enabled=True)
        writer.output_path = output_path
        writer.on_test_start(trainer, LightningModule())

        batch = torch.zeros((1, 2, 2, 2, 2))
        with pytest.raises(ValueError, match="channel count"):
            writer.on_test_batch_end(
                trainer,
                LightningModule(),
                {"prediction": batch, "target": batch},
                None,
                0,
            )

    def test_rejects_outputs_without_target(self, tmp_path: Path) -> None:
        trainer = _trainer(_combined_dataset())
        writer = PredictionWriter(enabled=True)
        writer.output_path = tmp_path / "predictions.nc"
        writer.on_test_start(trainer, LightningModule())

        with pytest.raises(TypeError, match="containing a target tensor"):
            writer.on_test_batch_end(
                trainer,
                LightningModule(),
                {"prediction": torch.zeros((1, 2, 1, 2, 2))},
                None,
                0,
            )

    def test_rejects_target_prediction_shape_mismatch(self, tmp_path: Path) -> None:
        trainer = _trainer(_combined_dataset())
        writer = PredictionWriter(enabled=True)
        writer.output_path = tmp_path / "predictions.nc"
        writer.on_test_start(trainer, LightningModule())

        with pytest.raises(ValueError, match="shapes do not match"):
            writer.on_test_batch_end(
                trainer,
                LightningModule(),
                {
                    "prediction": torch.zeros((2, 2, 1, 2, 2)),
                    "target": torch.zeros((1, 2, 1, 2, 2)),
                },
                None,
                0,
            )

    def test_on_exception_closes_partial_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "predictions.nc"
        trainer = _trainer(_combined_dataset())
        writer = PredictionWriter(enabled=True)
        writer.output_path = output_path
        writer.on_test_start(trainer, LightningModule())

        writer.on_exception(trainer, LightningModule(), RuntimeError("boom"))

        assert writer._file is None
        with NetCDFDataset(str(output_path)) as netcdf:
            assert "ice_conc" in netcdf.variables
