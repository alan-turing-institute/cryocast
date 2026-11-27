import logging
from typing import Any

from anemoi.datasets.usage import store

log = logging.getLogger(__name__)


def _no_analytics(*_args: Any, **_kwargs: Any) -> None:
    """Do not collect analytics."""


def patch_collect_analytics() -> None:
    """Stop anemoi-datasets from starting a process when it opens a zarr store.

    anemoi-datasets 0.5.50 sends every `open_zarr_store` call to a ProcessPoolExecutor
    for analytics. Since macOS DataLoader workers are daemon processes, they raise an
    AssertionError when trying to start child processes.

    """
    if store.collect_analytics is not _no_analytics:
        store.collect_analytics = _no_analytics
        log.debug("Disabled anemoi-datasets analytics collection")
