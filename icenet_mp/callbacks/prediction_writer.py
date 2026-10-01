import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from netCDF4 import Dataset as NetCDFDataset
from torch import Tensor

from icenet_mp.data import CombinedDataset
from icenet_mp.types import MaskType

if TYPE_CHECKING:  # per rule TC003
    from pathlib import Path

logger = logging.getLogger(__name__)

_TIME_UNITS = "seconds since 1970-01-01 00:00:00"
_TIME_CALENDAR = "proleptic_gregorian"
_NTCHW_NDIM = 5
_OBSERVED_SUFFIX = "_observed"
_MASK_ATTRIBUTES = {
    MaskType.LAND: {
        "long_name": "ocean mask (1 = ocean, 0 = land)",
        "flag_meanings": "land ocean",
    },
    MaskType.ACTIVE: {
        "long_name": "active grid cell mask (1 = active ocean, 0 = land or inactive)",
        "flag_meanings": "inactive active",
    },
}


def _add_variable(
    netcdf: NetCDFDataset,
    name: str,
    dtype: str,
    dimensions: tuple[str, ...],
    data: np.ndarray | None = None,
    /,
    **attributes: object,
) -> None:
    """Create a NetCDF variable, set its attributes, and optionally fill it."""
    variable = netcdf.createVariable(name, dtype, dimensions)
    variable.setncatts(attributes)
    if data is not None:
        variable[:] = data


class PredictionWriter(Callback):
    """Write evaluation predictions to a CF-style NetCDF file in the run directory."""

    def __init__(self, *, enabled: bool = False) -> None:
        """Configure whether prediction export is enabled."""
        super().__init__()
        self.enabled = enabled
        self.output_path: Path | None = None
        self.mask_dir: Path | None = None
        self._dataset: CombinedDataset | None = None
        self._file: Any | None = None
        self._sample_offset = 0

    @staticmethod
    def _load_dataset(trainer: Trainer) -> CombinedDataset:
        """Return the single CombinedDataset used for testing."""
        dataloaders: Any = trainer.test_dataloaders
        if isinstance(dataloaders, Sequence):
            if len(dataloaders) != 1:
                msg = "Prediction export supports exactly one test dataloader."
                raise ValueError(msg)
            dataloader = dataloaders[0]
        else:
            dataloader = dataloaders

        if dataloader is None or not isinstance(dataloader.dataset, CombinedDataset):
            msg = "Prediction export requires a CombinedDataset test dataloader."
            raise TypeError(msg)
        return dataloader.dataset

    @staticmethod
    def _seconds(values: np.ndarray) -> np.ndarray:
        """Convert datetime64 values to integer Unix seconds."""
        return values.astype("datetime64[s]").astype(np.int64)

    @staticmethod
    def _denormalise(
        prediction: np.ndarray,
        dataset: CombinedDataset,
    ) -> np.ndarray:
        """Convert model outputs from target normalisation back to source units."""
        minimum = np.asarray(
            dataset.target.statistics["minimum"], dtype=prediction.dtype
        )
        maximum = np.asarray(
            dataset.target.statistics["maximum"], dtype=prediction.dtype
        )
        if prediction.shape[2] != len(minimum) or len(minimum) != len(maximum):
            msg = (
                "Prediction channel count does not match target dataset statistics: "
                f"{prediction.shape[2]} channels vs {len(minimum)} statistics entries."
            )
            raise ValueError(msg)
        shape = (1, 1, -1, 1, 1)
        return prediction * (maximum - minimum).reshape(shape) + minimum.reshape(shape)

    def _load_masks(self, shape: tuple[int, int]) -> dict[MaskType, np.ndarray]:
        """Load whichever land/active masks exist in the mask directory."""
        masks: dict[MaskType, np.ndarray] = {}
        if self.mask_dir is None:
            return masks
        for mask_type in _MASK_ATTRIBUTES:
            mask_path = self.mask_dir / f"{mask_type}_mask.npy"
            if not mask_path.exists():
                logger.debug("No %s mask found at %s.", mask_type, mask_path)
                continue
            mask = np.load(mask_path)
            if mask.shape != shape:
                msg = (
                    f"{mask_type} mask shape {mask.shape} does not match the target "
                    f"grid shape {shape}."
                )
                raise ValueError(msg)
            masks[mask_type] = mask.astype(np.int8)
        return masks

    def _write_masks(
        self,
        netcdf: NetCDFDataset,
        masks: Mapping[MaskType, np.ndarray],
    ) -> list[str]:
        """Write static (y, x) mask variables and return their variable names."""
        names = []
        for mask_type, mask in masks.items():
            name = f"{mask_type}_mask"
            _add_variable(
                netcdf,
                name,
                "i1",
                ("y", "x"),
                mask,
                flag_values=np.asarray([0, 1], dtype=np.int8),
                coordinates="latitude longitude",
                **_MASK_ATTRIBUTES[mask_type],
            )
            names.append(name)
        return names

    def _initialise_file(self, dataset: CombinedDataset) -> None:
        """Create NetCDF dimensions, coordinates, and prediction variables."""
        if self.output_path is None:
            msg = (
                "PredictionWriter is enabled but no output_path was set. "
                "ModelService.build_trainer should set it from the run directory."
            )
            raise RuntimeError(msg)

        height, width = dataset.target.space.shape
        latitudes = np.asarray(dataset.target.latitudes, dtype=np.float32)
        longitudes = np.asarray(dataset.target.longitudes, dtype=np.float32)
        if latitudes.size != height * width or longitudes.size != height * width:
            msg = (
                "Target latitude/longitude coordinate counts do not match the "
                f"target grid shape {dataset.target.space.shape}."
            )
            raise ValueError(msg)
        masks = self._load_masks((height, width))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        netcdf = self._file = NetCDFDataset(
            str(self.output_path), "w", format="NETCDF4"
        )
        netcdf.setncattr("Conventions", "CF-1.10")
        netcdf.setncattr("title", "IceNet-MP model predictions")
        netcdf.setncattr("hemisphere", dataset.target.hemisphere)

        netcdf.createDimension("forecast_reference_time", None)
        netcdf.createDimension("lead_time", dataset.n_forecast_steps)
        netcdf.createDimension("y", height)
        netcdf.createDimension("x", width)

        time_attributes = {"units": _TIME_UNITS, "calendar": _TIME_CALENDAR}
        _add_variable(
            netcdf,
            "forecast_reference_time",
            "i8",
            ("forecast_reference_time",),
            standard_name="forecast_reference_time",
            **time_attributes,
        )

        frequency_seconds = int(
            dataset.frequency.astype("timedelta64[s]").astype(np.int64)
        )
        lead_seconds = (
            np.arange(1, dataset.n_forecast_steps + 1, dtype=np.int64)
            * frequency_seconds
        )
        _add_variable(
            netcdf,
            "lead_time",
            "i8",
            ("lead_time",),
            lead_seconds,
            standard_name="forecast_period",
            long_name="forecast lead time",
            units="seconds",
        )
        _add_variable(
            netcdf,
            "valid_time",
            "i8",
            ("forecast_reference_time", "lead_time"),
            standard_name="time",
            **time_attributes,
        )
        _add_variable(
            netcdf,
            "latitude",
            "f4",
            ("y", "x"),
            latitudes.reshape(height, width),
            standard_name="latitude",
            units="degrees_north",
        )
        _add_variable(
            netcdf,
            "longitude",
            "f4",
            ("y", "x"),
            longitudes.reshape(height, width),
            standard_name="longitude",
            units="degrees_east",
        )

        mask_names = self._write_masks(netcdf, masks)
        for variable_name in dataset.target.variable_names:
            self._create_field_variable(
                netcdf, variable_name, mask_names, observed=False
            )
            self._create_field_variable(
                netcdf, variable_name, mask_names, observed=True
            )

    def _create_field_variable(
        self,
        netcdf: NetCDFDataset,
        variable_name: str,
        mask_names: list[str],
        *,
        observed: bool,
    ) -> None:
        """Create one (forecast_reference_time, lead_time, y, x) field variable.

        Predictions are written under the target variable name; the corresponding
        ground truth is written alongside it with an ``_observed`` suffix.
        """
        variable = netcdf.createVariable(
            f"{variable_name}{_OBSERVED_SUFFIX}" if observed else variable_name,
            "f4",
            ("forecast_reference_time", "lead_time", "y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.nan,
        )
        variable.coordinates = (
            "forecast_reference_time lead_time valid_time latitude longitude"
        )
        if mask_names:
            variable.ancillary_variables = " ".join(mask_names)
        if variable_name == "ice_conc":
            variable.standard_name = "sea_ice_area_fraction"
            variable.long_name = (
                "observed sea ice concentration"
                if observed
                else "predicted sea ice concentration"
            )
            variable.units = "1"
        elif not observed:
            logger.warning(
                "No CF standard_name/units mapping for prediction variable '%s'; "
                "it will be written without them, despite this file's Conventions "
                "attribute declaring CF-1.10.",
                variable_name,
            )

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Initialise prediction export before evaluation batches run."""
        if not self.enabled:
            return
        if getattr(trainer, "world_size", 1) != 1:
            msg = (
                "NetCDF prediction export currently requires single-process evaluation. "
                "Set evaluate.trainer.devices=1 when using --save-predictions."
            )
            raise RuntimeError(msg)

        self._dataset = self._load_dataset(trainer)
        self._sample_offset = 0
        self._initialise_file(self._dataset)

    def on_test_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,  # noqa: ANN401, ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Append one evaluation batch to the NetCDF file."""
        if not self.enabled:
            return
        self._write_batch(outputs)

    @staticmethod
    def _field_from_outputs(
        outputs: Tensor | Mapping[str, Any] | None,
        key: str,
        dataset: CombinedDataset,
    ) -> np.ndarray:
        """Extract, validate, and denormalise one NTCHW field from test outputs."""
        if not isinstance(outputs, Mapping) or not isinstance(
            tensor := outputs.get(key),
            Tensor,
        ):
            msg = f"Prediction writer expected test outputs containing a {key} tensor."
            raise TypeError(msg)

        field = tensor.detach().float().cpu().numpy()
        if field.ndim != _NTCHW_NDIM:
            msg = (
                f"Prediction writer expected NTCHW {key} values, "
                f"received shape {field.shape}."
            )
            raise ValueError(msg)
        if field.shape[1] != dataset.n_forecast_steps:
            msg = (
                f"{key.capitalize()} forecast-step count does not match the test "
                f"dataset: {field.shape[1]} vs {dataset.n_forecast_steps}."
            )
            raise ValueError(msg)
        return PredictionWriter._denormalise(field, dataset)

    def _write_batch(self, outputs: Tensor | Mapping[str, Any] | None) -> None:
        """Validate and append one evaluation batch to the NetCDF file."""
        if self._dataset is None or self._file is None:
            msg = "Prediction writer was not initialised before receiving a test batch."
            raise RuntimeError(msg)

        prediction = self._field_from_outputs(outputs, "prediction", self._dataset)
        target = self._field_from_outputs(outputs, "target", self._dataset)
        if target.shape != prediction.shape:
            msg = (
                "Target and prediction shapes do not match: "
                f"{target.shape} vs {prediction.shape}."
            )
            raise ValueError(msg)

        batch_size = prediction.shape[0]
        start = self._sample_offset
        end = start + batch_size
        start_dates = np.asarray(self._dataset.dates[start:end])
        if len(start_dates) != batch_size:
            msg = (
                "Prediction writer received more samples than are available in the "
                "test dataset."
            )
            raise IndexError(msg)

        reference_dates = np.asarray(
            [self._dataset.get_history_steps(date)[-1] for date in start_dates]
        )
        reference_seconds = self._seconds(reference_dates)
        valid_dates = np.asarray(
            [self._dataset.get_forecast_steps(date) for date in start_dates]
        )
        valid_seconds = self._seconds(valid_dates)

        self._file.variables["forecast_reference_time"][start:end] = reference_seconds
        self._file.variables["valid_time"][start:end, :] = valid_seconds
        for channel_idx, variable_name in enumerate(
            self._dataset.target.variable_names
        ):
            self._file.variables[variable_name][start:end, :, :, :] = prediction[
                :, :, channel_idx, :, :
            ].astype(np.float32, copy=False)
            self._file.variables[f"{variable_name}{_OBSERVED_SUFFIX}"][
                start:end, :, :, :
            ] = target[:, :, channel_idx, :, :].astype(np.float32, copy=False)

        self._sample_offset = end

    def _close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def on_test_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Close the NetCDF file after evaluation."""
        if not self.enabled:
            return
        self._close()
        logger.info(
            "Saved %d model prediction window(s) to %s.",
            self._sample_offset,
            self.output_path,
        )

    def on_exception(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        exception: BaseException,  # noqa: ARG002
    ) -> None:
        """Close the NetCDF file if evaluation fails, leaving the partial output readable."""
        self._close()
