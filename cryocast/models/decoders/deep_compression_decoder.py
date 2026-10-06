"""Deep Compression decoder following the Deep Compression AutoEncoder architecture.

Reference:
    Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024) [https://arxiv.org/abs/2410.10733]
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from torch import nn

from cryocast.models.common import ResBlock, ResidualUpsample
from cryocast.types import TensorNCHW

from .base_decoder import BaseDecoder

logger = logging.getLogger(__name__)


class DeepCompressionDecoder(BaseDecoder):
    """Decoder following the Deep Compression AutoEncoder (DCAE) architecture.

    Mirror of :class:`DeepCompressionEncoder`.

    - initial convolution from latent channels
    - `len(hid_blocks) - 1` layers of `hid_blocks[i]` ResBlocks then upsample (PixelShuffle or Upsample)
    - final convolution to output channels

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        attention_heads: dict[int, int] | None = None,
        attention_scales: tuple[int, ...] = (5,),
        ffn_factor: int = 1,
        hid_blocks: Sequence[int] = (3, 3, 3),
        hid_channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 3,
        norm: str = "groupnorm",
        patch_size: int = 1,
        periodic: bool = False,
        pixel_shuffle: bool = True,
        stride: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialise a DeepCompressionDecoder."""
        super().__init__(**kwargs)

        attention_heads = attention_heads or {}

        if len(hid_blocks) != len(hid_channels):
            msg = f"hid_blocks and hid_channels must have the same length, got {len(hid_blocks)} and {len(hid_channels)}"
            raise ValueError(msg)
        if patch_size < 1:
            msg = f"patch_size must be >= 1, got {patch_size}"
            raise ValueError(msg)
        if stride < 1:
            msg = f"stride must be >= 1, got {stride}"
            raise ValueError(msg)
        in_channels = self.data_space_in.channels
        out_channels = self.data_space_out.channels

        # Validate the output shape is correct.
        spatial_factor = patch_size * stride ** (len(hid_channels) - 1)
        output_shape = (
            self.data_space_in.shape[0] * spatial_factor,
            self.data_space_in.shape[1] * spatial_factor,
        )
        if output_shape != self.data_space_out.shape:
            msg = (
                f"Stride {stride} and number of layers {len(hid_channels)} will decode "
                f"latents of shape {self.data_space_in.shape} to shape {output_shape} "
                f"but the required output shape is {self.data_space_out.shape}"
            )
            raise ValueError(msg)

        # Set padding and padding mode for convolutions
        padding = kernel_size // 2
        padding_mode: Literal["circular", "zeros"] = "circular" if periodic else "zeros"

        # Construct list of layers
        layers: list[nn.Module] = []
        logger.debug(
            "DeepCompressionDecoder (%s): %d layers, %d -> %d channels",
            self.name,
            len(hid_channels),
            in_channels,
            out_channels,
        )

        # Build in reverse order (deepest first) to mirror the encoder
        for idx in reversed(range(len(hid_blocks))):
            if idx + 1 == len(hid_blocks):
                # Deepest layer: convolve from latent channels
                layers.append(
                    ResidualUpsample(
                        in_channels=in_channels,
                        out_channels=hid_channels[idx],
                        factor=1,
                        pixel_shuffle=False,
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                )

            # Add `num_blocks` residual blocks
            layers.extend(
                ResBlock(
                    hid_channels[idx],
                    attention_heads=attention_heads.get(idx),
                    attention_scales=attention_scales,
                    ffn_factor=ffn_factor,
                    kernel_size=kernel_size,
                    norm=norm,
                    padding_mode=padding_mode,
                    padding=padding,
                )
                for _ in range(hid_blocks[idx])
            )

            if idx > 0:
                # Subsequent layers: upsample by factor `stride`
                layers.append(
                    ResidualUpsample(
                        in_channels=hid_channels[idx],
                        out_channels=hid_channels[idx - 1],
                        factor=stride,
                        pixel_shuffle=pixel_shuffle,
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                )

            else:
                # Shallowest layer: convolve then (optionally unpatchify).
                layers.append(
                    ResidualUpsample(
                        in_channels=hid_channels[idx],
                        out_channels=out_channels,
                        factor=patch_size,
                        pixel_shuffle=pixel_shuffle if patch_size > 1 else False,
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                )

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space with a DCAE decoder.

        Args:
            x: TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        return self.model(x)
