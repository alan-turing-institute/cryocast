from dataclasses import dataclass
from typing import NamedTuple, TypedDict

from anemoi.datasets.create.recipe import Recipe
from torch import Tensor

from .annotations import TensorNTCHW


@dataclass
class AnemoiCleanupArgs:
    """Arguments for anemoi cleanup."""

    path: str
    command: str = "unused"


class AnemoiDatasetStatus(NamedTuple):
    """Status of an Anemoi dataset."""

    copy_in_progress: bool
    download_complete: bool
    is_finalised: bool


@dataclass
class AnemoiFinaliseArgs:
    """Arguments for anemoi finalise."""

    path: str
    recipe: Recipe
    command: str = "unused"


@dataclass
class AnemoiInitArgs:
    """Arguments for anemoi init."""

    path: str
    recipe: Recipe
    command: str = "unused"
    overwrite: bool = False


@dataclass
class AnemoiInspectArgs:
    """Arguments for anemoi inspect."""

    detailed: bool
    path: str
    progress: bool
    size: bool
    statistics: bool


@dataclass
class AnemoiLoadArgs:
    """Arguments for anemoi load."""

    path: str
    recipe: Recipe
    command: str = "unused"


class DataloaderArgs(TypedDict):
    """Arguments for the data loader."""

    batch_sampler: None
    batch_size: int
    drop_last: bool
    num_workers: int
    prefetch_factor: int | None
    persistent_workers: bool
    sampler: None
    worker_init_fn: None


@dataclass(frozen=True)
class ProcessorOutput:
    """Output of a processor rollout step."""

    prediction: TensorNTCHW
    loss: Tensor | None = None
