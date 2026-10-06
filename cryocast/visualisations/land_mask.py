import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LandMask:
    """Load and apply land masks to data arrays."""

    def __init__(self, land_mask_path: Path | None) -> None:
        """A helper class to apply land masks to data arrays."""
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._ignored: set[tuple[int, int]] = set()
        if land_mask_path and land_mask_path.exists():
            try:
                self.add_mask(np.load(land_mask_path))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Failed to load land mask from %s: %s", land_mask_path, exc
                )

    def add_mask(self, mask_array: np.ndarray) -> None:
        """Add a land mask to the cache, keyed by its shape."""
        self._cache[mask_array.shape] = mask_array.astype(bool)

    def apply_to(self, data_array: np.ndarray) -> np.ndarray:
        """Apply a land mask to an array."""
        shape = data_array.shape[-2:]
        # If there is no mask in the cache, return the array unchanged
        if shape not in self._cache:
            if shape not in self._ignored:
                logger.debug(
                    "No land mask associated with this dataset has shape %s.", shape
                )
                self._ignored.add(shape)
            return data_array
        # Otherwise, apply the mask (mask out land to NaN)
        # N.B. the mask is inverted as we want to hide the land
        return np.where(~self._cache[shape], np.nan, data_array)
