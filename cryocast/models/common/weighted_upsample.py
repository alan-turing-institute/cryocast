from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn.init import kaiming_normal_


class WeightedUpsample(nn.Module):
    """Learned spatial upsampling via sub-pixel convolution (PixelShuffle).

    Conv2d > PixelShuffle

    The initial Conv2d is ICNR-initialised to mitigate checkerboarding that might
    otherwise occur during early training.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int | None = None,
        upsample_factor: int = 2,
        kernel_size: int = 1,
        padding: int = 0,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ) -> None:
        """Initialise a WeightedUpsample module.

        Args:
            in_channels: the number of input channels.
            out_channels: the number of output channels.
            upsample_factor: the spatial upsampling factor.
            kernel_size: the size of the pre-shuffle convolutional kernel.
            padding: padding for the pre-shuffle convolution.
            padding_mode: padding mode for the pre-shuffle convolution.

        """
        super().__init__()

        # Initial convolution to produce the required channels for PixelShuffle
        out_channels = out_channels if out_channels is not None else in_channels
        initial_conv = nn.Conv2d(
            in_channels,
            out_channels * upsample_factor**2,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

        # ICNR initialisation from https://arxiv.org/abs/1707.02937.
        # Set all sub-channel groups to a common Kaiming-normal kernel so the initial
        # state for PixelShuffle looks like nearest-neighbour upsampling.
        group_weights = initial_conv.weight.new_empty(
            out_channels, *initial_conv.weight.shape[1:]
        )
        kaiming_normal_(group_weights)
        with torch.no_grad():
            initial_conv.weight.copy_(
                group_weights.repeat_interleave(upsample_factor**2, dim=0)
            )

        self.block = nn.Sequential(
            # Convolution: [N, C_in, H, W] -> [N, C_internal, H, W]
            initial_conv,
            # PixelShuffle: [N, C_internal, H, W] -> [N, C_out, H * r, W * r]
            nn.PixelShuffle(upsample_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
