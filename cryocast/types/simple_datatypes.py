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
    """Output of a processor rollout.

    Attributes:
        prediction: Latent prediction of shape (N, T, C, H, W). At inference, T is
            n_forecast_steps. During training, a processor that computes its loss in
            latent space can return a prediction with gradients back to the network, and
            T can be 1. For example, an autoregressive diffusion processor predicts only
            the first forecast step.
        loss: Latent loss for the backward pass. A processor must set it during training
            if `computes_loss_in_latent_space=True`. For other processors, this is
            ignored.
        sample_mask: Optional boolean mask of shape (N,). True for the samples to use in
            the prediction loss. None means that all samples are used.

    """

    prediction: TensorNTCHW
    loss: Tensor | None = None
    sample_mask: Tensor | None = None
