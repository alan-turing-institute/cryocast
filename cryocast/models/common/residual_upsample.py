"""Upsample block wrapped in DC-AE's parameter-free "Residual Autoencoding" shortcut."""

from typing import Any

from torch import Tensor, nn

from .channel_adaptor import ChannelAdaptor
from .weighted_upsample import WeightedUpsample


class ResidualUpsample(nn.Module):
    """A parametric upsample plus a non-parametric residual shortcut.

    Mirror of :class:`ResidualDownsample` for the decoder side.

    The parametric block is either WeightedUpsample or Upsample + Conv2d.
    The non-parametric shortcut block is a channel adaptor to set the correct number of
    output channels plus a PixelShuffle.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        factor: int,
        pixel_shuffle: bool,
        shortcut: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise a ResidualUpsample.

        Args:
            in_channels: the number of input channels.
            out_channels: the number of output channels.
            factor: the spatial upsampling factor.
            pixel_shuffle: upsample with PixelShuffle if True, else nearest-neighbour
                upsampling followed by convolution.
            shortcut: whether to include the non-parametric residual shortcut.
            **kwargs: forwarded to the parametric block convolutional layer

        """
        super().__init__()
        if pixel_shuffle:
            self.parametric: nn.Module = WeightedUpsample(
                in_channels, out_channels=out_channels, upsample_factor=factor, **kwargs
            )
        else:
            self.parametric = nn.Sequential(
                nn.Upsample(scale_factor=factor, mode="nearest")
                if factor > 1
                else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, **kwargs),
            )
        self.shortcut = (
            nn.Sequential(
                ChannelAdaptor(in_channels, out_channels * factor**2),
                nn.PixelShuffle(factor) if factor > 1 else nn.Identity(),
            )
            if shortcut
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the parametric block plus its residual shortcut to x."""
        if self.shortcut is None:
            return self.parametric(x)
        return self.parametric(x) + self.shortcut(x)
