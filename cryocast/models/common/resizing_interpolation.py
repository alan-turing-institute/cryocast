import logging
from collections.abc import Sequence

from torch import nn

from cryocast.types import TensorNCHW

logger = logging.getLogger(__name__)


class ResizingInterpolation(nn.Module):
    """Resize to an arbitrary output shape using interpolation.

    The output shape must be specified at initialisation time.
    """

    def __init__(
        self,
        output_shape: Sequence[int],
        *,
        align_corners: bool = True,
        antialias: bool = True,
    ) -> None:
        """Initialize the ResizingInterpolate module.

        Args:
            output_shape: the target output size in H x W format.
            align_corners: whether to align the corner pixels of the input and output tensors. Default: True.
            antialias: whether to perform antialiasing during interpolation. Default: True.

        """
        super().__init__()
        self.align_corners = align_corners
        self.output_shape = (output_shape[0], output_shape[1])
        self.antialias = antialias

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return nn.functional.interpolate(
            x,
            size=self.output_shape,
            mode="bilinear",
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
