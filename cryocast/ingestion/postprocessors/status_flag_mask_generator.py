import logging
from pathlib import Path

import numpy as np
from anemoi.datasets import open_dataset

from cryocast.utils import mask_dir

from .ipostprocessor import IPostprocessor

logger = logging.getLogger(__name__)


class StatusFlagMaskGenerator(IPostprocessor):
    def process(self, path_dataset: Path, *, overwrite: bool) -> None:
        """Generate land and active grid cell masks from the status_flag variable."""
        logger.debug("Generating land and active grid cell masks from status_flag.")

        path_masks = mask_dir(self.base_path, self.dataset_name)
        path_masks.mkdir(parents=True, exist_ok=True)
        land_mask_path = path_masks / "land_mask.npy"
        active_mask_path = path_masks / "active_mask.npy"

        if land_mask_path.exists() and active_mask_path.exists() and not overwrite:
            logger.debug("Both masks already exist, skipping creation.")
            return

        # Unpack status flags into a binary array, skipping any missing dates
        ds_sf = open_dataset(path_dataset, select="status_flag")
        missing_indices: set[int] = (
            set(m) if (m := getattr(ds_sf, "missing", None)) is not None else set()
        )
        if missing_indices:
            logger.warning(
                "Skipping %d missing dates when generating masks.", len(missing_indices)
            )
        available_indices = [i for i in range(len(ds_sf)) if i not in missing_indices]
        if not available_indices:
            msg = f"No available timesteps in status_flag for dataset {self.dataset_name}."
            raise RuntimeError(msg)
        # Attempting to read the entire array and then index leads to MissingDateError.
        # Instead iterate over the available indices and recombine.
        status_flag = np.stack(
            [np.asarray(ds_sf[i], dtype=np.uint8) for i in available_indices]
        )
        binary = np.unpackbits(status_flag, axis=-1).reshape(*status_flag.shape, 8)

        # land mask: land = 0, sea = 1
        if land_mask_path.exists() and not overwrite:
            logger.debug("Land mask already exists, skipping creation.")
            land_mask = np.load(land_mask_path)
        else:
            land_mask = np.squeeze(binary[..., [7]]).sum(axis=0)
            # convert to binary mask
            land_mask = 1 - (land_mask > 0).astype(np.uint8)
            # reshape to 2D grid
            land_mask = land_mask.reshape(ds_sf.field_shape[-2:])
            # save the land mask for later use
            np.save(land_mask_path, land_mask)
            logger.info("Land mask created and saved.")

        # active mask: active grid cells = 1, inactive = 0
        if active_mask_path.exists() and not overwrite:
            logger.debug("Active mask already exists, skipping creation.")
        else:
            # Identify grid cells that are inactive for all time steps
            inactive_count = np.squeeze(binary[..., [0]]).sum(axis=0)
            inactive_mask = inactive_count >= len(available_indices)
            # convert to binary mask, and set to 1 for active grid cells
            active_mask = 1 - (inactive_mask > 0).astype(np.uint8)
            # reshape to 2D grid
            active_mask = active_mask.reshape(ds_sf.field_shape[-2:])
            # intersect land mask with active mask to set all active grid cells to 1
            active_mask = active_mask * land_mask
            # save the active mask for later use
            np.save(active_mask_path, active_mask)
            logger.info("Active mask created and saved.")
