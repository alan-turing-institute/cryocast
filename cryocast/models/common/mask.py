from pathlib import Path

import numpy as np
from torch import Tensor, from_numpy, nn

from cryocast.types import MaskType


class Mask(nn.Module):
    """Mask out inactive/land cells in model output.

    Loads a mask from disk and applies it multiplicatively. If no mask is requested then
    the input is returned unchanged.
    """

    # buffer in __init__, annotated here to make the type explicit
    mask: Tensor

    def __init__(
        self,
        *,
        mask_type: MaskType | str | None,
        output_shape: tuple[int, int],
        mask_dir: str | Path | None = None,
    ) -> None:
        """Initialise a Mask.

        Args:
            mask_type: MaskType.ACTIVE (active+land), MaskType.LAND (land only), or
                MaskType.NONE/None/"" to disable. Any other value raises ValueError.
            output_shape: Expected (height, width) shape of the tensor this mask will be
                applied to, used to validate the mask loaded from disk.
            mask_dir: Directory holding `active_mask.npy`/`land_mask.npy`, generated for
                SSMIS datasets by `datasets create`. Required when `mask_type` is
                MaskType.ACTIVE or MaskType.LAND.

        """
        super().__init__()
        mask_type = MaskType(mask_type) if mask_type else MaskType.NONE
        if mask_type is MaskType.NONE:
            return

        mask_path = Path(mask_dir) / f"{mask_type}_mask.npy" if mask_dir else None
        if mask_path is None or not mask_path.exists():
            msg = (
                f"{mask_type} mask is requested but no mask was found at {mask_path}. "
                "Masks are generated for SSMIS datasets during `datasets create`."
            )
            raise FileNotFoundError(msg)
        mask = from_numpy(np.load(mask_path)).float()
        if tuple(mask.shape) != tuple(output_shape):
            msg = (
                f"{mask_type} mask shape {tuple(mask.shape)} does not match expected "
                f"output shape {tuple(output_shape)}."
            )
            raise ValueError(msg)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Multiply by the mask if loaded, otherwise return x unchanged."""
        mask = getattr(self, "mask", None)
        if mask is None:
            return x
        return x * mask.to(dtype=x.dtype)
