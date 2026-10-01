from collections.abc import Sequence
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsImageLogging(Protocol):
    def log_image(
        self, key: str, images: list[Any], step: int | None = None, **kwargs: Any
    ) -> None: ...


@runtime_checkable
class SupportsVideoLogging(Protocol):
    def log_video(
        self, key: str, videos: list[Any], step: int | None = None, **kwargs: Any
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
