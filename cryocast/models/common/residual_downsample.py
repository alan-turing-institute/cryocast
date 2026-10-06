"""Downsample block wrapped in DC-AE's parameter-free "Residual Autoencoding" shortcut."""

from typing import Any

from torch import Tensor, nn

from .channel_adaptor import ChannelAdaptor


class ResidualDownsample(nn.Module):
    """A parametric downsample plus a non-parametric residual shortcut.

    Mirror of :class:`ResidualUpsample` for the encoder side.

    The parametric block is one of:
    - Conv2d + (PixelUnshuffle + ChannelAdaptor)
    - strided Conv2d + ChannelAdaptor

    The non-parametric shortcut block is a PixelUnshuffle followed by a channel adaptor
    to set the correct number of output channels.
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
        """Initialise a ResidualDownsample.

        Args:
            in_channels: the number of input channels.
            out_channels: the number of output channels.
            factor: the spatial downsampling factor.
            pixel_shuffle: downsample with PixelUnshuffle if True, else strided convolution.
            shortcut: whether to include the non-parametric residual shortcut.
            **kwargs: forwarded to the parametric block convolutional layer

        """
        super().__init__()
        if pixel_shuffle:
            if factor > 1:
                # Convolve before reshuffling to reduce the intermediate channel count.
                conv_channels = -(-out_channels // factor**2)  # ceil division
                self.parametric: nn.Module = nn.Sequential(
                    nn.Conv2d(in_channels, conv_channels, **kwargs),
                    nn.PixelUnshuffle(factor),
                    ChannelAdaptor(conv_channels * factor**2, out_channels),
                )
            elif factor == 1:
                # If factor is 1, no downsampling is needed
                self.parametric = nn.Conv2d(in_channels, out_channels, **kwargs)
            else:
                msg = f"factor must be >= 1, got {factor}"
                raise ValueError(msg)
        else:
            self.parametric = nn.Conv2d(
                in_channels, out_channels, stride=factor, **kwargs
            )
        self.shortcut = (
            nn.Sequential(
                nn.PixelUnshuffle(factor) if factor > 1 else nn.Identity(),
                ChannelAdaptor(in_channels * factor**2, out_channels),
            )
            if shortcut
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the parametric block plus its residual shortcut to x."""
        if self.shortcut is None:
            return self.parametric(x)
        return self.parametric(x) + self.shortcut(x)
