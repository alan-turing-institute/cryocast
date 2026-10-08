import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path, PurePath

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf
from wandb.wandb_run import Run

from cryocast.types import DataSpace

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def datetime_from_npdatetime(dt: np.datetime64) -> datetime:
    """Convert numpy datetime64 to aware datetime in UTC."""
    # NumPy datetimes are timezone-naive UTC, so attach UTC rather than converting
    # (astimezone would treat the naive value as local time)
    return dt.astype("datetime64[ms]").astype(datetime).replace(tzinfo=UTC)


def get_device_name(accelerator_name: str) -> str:
    """Get the device name for the given accelerator."""
    if accelerator_name == "cuda":
        try:
            return torch.cuda.get_device_name()
        except AssertionError:
            return "Unknown CUDA device"
    if accelerator_name == "mps":
        return "Apple Silicon GPU"
    if accelerator_name == "xpu":
        try:
            return torch.xpu.get_device_name()
        except AssertionError:
            return "Unknown XPU device"
    return "CPU"


def get_timestamp() -> str:
    """Return the current time as a string."""
    return datetime.now(tz=UTC).strftime(r"%Y%m%d_%H%M%S")


def get_wandb_run(trainer: Trainer) -> Run | None:
    """Get the Wandb Run instance if it exists."""
    for lightning_logger in trainer.loggers:
        if isinstance(lightning_logger, WandbLogger) and isinstance(
            experiment := lightning_logger.experiment, Run
        ):
            return experiment
    return None


def iso_from_date(dt: date | datetime) -> str:
    """Format a date/datetime as an ISO date string (YYYY-MM-DD) for titles/keys."""
    if isinstance(dt, datetime):
        return dt.date().isoformat()
    return dt.isoformat()


def mask_dir(base_path: Path, dataset_name: str) -> Path:
    """Path finder for holding the active masks.

    On-disk active mask layout is defined here once and used everywhere, single source,
    used both when active masks are written (in dataset creation) and when they are read
    (during model build), so they never diverge.
    """
    return base_path / "data" / "masks" / dataset_name


def npdatetime_from_datetime(dt: datetime) -> np.datetime64:
    """Convert an aware or naive datetime to numpy datetime64, dropping tzinfo."""
    return np.datetime64(dt.replace(tzinfo=None))


def safe_nanmin(arr: np.ndarray, default: float = 0.0) -> float:
    """Safely compute nanmin with fallback for empty or all-NaN arrays.

    Args:
        arr: Array to compute minimum from.
        default: Default value if array is empty or all NaN.

    Returns:
        Minimum value or default.

    """
    finite = arr[np.isfinite(arr)]
    return float(np.min(finite)) if finite.size else default


def safe_nanmax(arr: np.ndarray, default: float = 1.0) -> float:
    """Safely compute nanmax with fallback for empty or all-NaN arrays.

    Args:
        arr: Array to compute maximum from.
        default: Default value if array is empty or all NaN.

    Returns:
        Maximum value or default.

    """
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else default


def sanitise_filename(text: str) -> str:
    """Replace characters unsafe for filenames/logger keys with an underscore."""
    return _UNSAFE_FILENAME_CHARS.sub("_", text)


def to_list(value: str | Sequence[str]) -> list[str]:
    """Convert a value or sequence of values to a list of values."""
    if isinstance(value, str):
        return [value]
    return value if isinstance(value, list) else list(value)


def to_plain_types(value: object) -> object:  # noqa: PLR0911
    """Recursively convert enums, paths, DataSpaces and OmegaConf containers to plain types.

    This is useful for values that need to be stored, for example in checkpoints, which
    can then be loaded without needing to import the original classes.
    """
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, DataSpace):
        return to_plain_types(value.to_dict())
    if isinstance(value, DictConfig | ListConfig):
        return to_plain_types(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {key: to_plain_types(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(to_plain_types(item) for item in value)
    return value
