from collections.abc import Sequence
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class SupportsImageLogging(Protocol):
    def log_image(
        self, key: str, images: list[Any], step: int | None = None, **kwargs: Any
    ) -> None: ...


@runtime_checkable
class SupportsMetadataInput(Protocol):
    """A single named source of variables, as used by `SupportsMetadataFromDataset`."""

    name: str

    @cached_property
    def variable_names(self) -> list[str]: ...


@runtime_checkable
class SupportsMetadataFromDataset(Protocol):
    """Can be used by `Metadata.from_dataset` to build a `Metadata` instance."""

    @property
    def inputs(self) -> Sequence[SupportsMetadataInput]: ...

    @property
    def start_date(self) -> np.datetime64: ...

    @property
    def end_date(self) -> np.datetime64: ...

    @property
    def n_history_steps(self) -> int: ...

    def __len__(self) -> int:
        """Return the number of points in the dataset."""
        ...


@runtime_checkable
class SupportsPerLeadTimeLoss(Protocol):
    """A loss that can evaluate every lead time of an NTCHW input in one call."""

    def per_lead_time_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return the [T] per-lead-time losses for NTCHW prediction/target."""
        ...


@runtime_checkable
class SupportsVideoLogging(Protocol):
    def log_video(
        self, key: str, videos: list[Any], step: int | None = None, **kwargs: Any
    ) -> None: ...
