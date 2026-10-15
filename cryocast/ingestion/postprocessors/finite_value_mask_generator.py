import logging
from pathlib import Path

import numpy as np
from anemoi.datasets import open_dataset

from cryocast.utils import mask_dir

from .ipostprocessor import IPostprocessor

logger = logging.getLogger(__name__)


class FiniteValueMaskGenerator(IPostprocessor):
    """Generate masks from cells that are finite at every available timestep."""

    def __init__(
        self,
        base_path: Path,
        dataset_name: str,
        *,
        variable: str,
    ) -> None:
        """Initialise a finite-value mask generator for one target variable."""
        super().__init__(base_path=base_path, dataset_name=dataset_name)
        if not variable:
            msg = "variable must not be empty."
            raise ValueError(msg)
        self.variable = variable

    def process(self, path_dataset: Path, *, overwrite: bool) -> None:
        """Save valid-cell masks for the configured variable."""
        path_masks = mask_dir(self.base_path, self.dataset_name)
        path_masks.mkdir(parents=True, exist_ok=True)
        land_mask_path = path_masks / "land_mask.npy"
        active_mask_path = path_masks / "active_mask.npy"

        if land_mask_path.exists() and active_mask_path.exists() and not overwrite:
            logger.debug("Both finite-value masks already exist, skipping creation.")
            return

        dataset = open_dataset(path_dataset, select=self.variable)
        missing_indices = set(getattr(dataset, "missing", None) or [])
        available_indices = [
            index for index in range(len(dataset)) if index not in missing_indices
        ]
        if not available_indices:
            msg = (
                f"No available timesteps in {self.variable!r} for dataset "
                f"{self.dataset_name}."
            )
            raise RuntimeError(msg)

        field_shape = tuple(dataset.field_shape[-2:])
        valid_mask = np.ones(field_shape, dtype=bool)
        for index in available_indices:
            values = np.asarray(dataset[index]).reshape(field_shape)
            valid_mask &= np.isfinite(values)

        mask = valid_mask.astype(np.uint8)
        np.save(land_mask_path, mask)
        np.save(active_mask_path, mask)
        logger.info(
            "Created finite-value masks for %s (%d/%d valid cells).",
            self.dataset_name,
            int(mask.sum()),
            mask.size,
        )
