"""Parameter-free channel-count change: a fixed (non-trainable) 1x1 convolution."""

import torch
from torch import Tensor, nn

from cryocast.types import TensorNCHW


class ChannelAdaptor(nn.Module):
    """Deterministically change the number of channels of an NCHW tensor.

    Reducing the number of channels averages across channels, while increasing simply
    duplicates channels. This is implemented as a fixed, linear map applied through a
    non-learnable 1x1 convolution.
    """

    channel_map: Tensor

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialise a ChannelAdaptor."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Output channel idx_out averages input channels [start, end). When out_channels
        # is greater than in_channels, this will duplicate channels. When out_channels
        # is less than in_channels, this will average channels. The map between input
        # and output channels is fixed, and we therefore calculate it once and save it
        # as a buffer.
        channel_map = torch.zeros(out_channels, in_channels)
        for idx_out in range(out_channels):
            start = (idx_out * in_channels) // out_channels
            end = -(-((idx_out + 1) * in_channels) // out_channels)  # ceil division
            channel_map[idx_out, start:end] = 1.0 / (end - start)
        self.register_buffer(
            "channel_map", channel_map.view(out_channels, in_channels, 1, 1)
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Apply the deterministic channel map to an NCHW tensor."""
        if self.out_channels == self.in_channels:
            return x
        return nn.functional.conv2d(x, weight=self.channel_map)
