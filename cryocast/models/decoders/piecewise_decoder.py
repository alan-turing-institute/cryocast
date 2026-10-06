from typing import Any

from torch import nn

from cryocast.models.common import (
    CommonConvBlock,
    NormalisedFold,
    Permute,
)
from cryocast.types import TensorNCHW

from .base_decoder import BaseDecoder


class PiecewiseDecoder(BaseDecoder):
    """Piecewise decoder that combines data patches from a latent space to build the output space.

    - Initial convolutional block at input resolution
    - Combine patches into output of size output_height x output_width
    - Final convolutional block at output resolution
    - Normalise then bound the output

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    def __init__(
        self,
        *,
        conv_activation: str = "SiLU",
        conv_kernel_size: int = 3,
        conv_subblocks_initial: int = 3,
        conv_subblocks_final: int = 3,
        use_hann_window: bool = True,
        use_final_normalisation: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise a PiecewiseDecoder."""
        super().__init__(**kwargs)

        # Calculate the number of patches required
        # We set the stride to be half the patch size to ensure overlap, which will
        # capture more of the spatial structure of the data.
        strides = tuple(
            max(1, patch_size // 2) for patch_size in self.data_space_in.shape
        )
        n_patches = (
            (
                self.data_space_out.shape[0]
                + 2 * strides[0]
                - 1 * (self.data_space_in.shape[0] - 1)
                - 1
            )
            // strides[0]
            + 1
        ) * (
            (
                self.data_space_out.shape[1]
                + 2 * strides[1]
                - 1 * (self.data_space_in.shape[1] - 1)
                - 1
            )
            // strides[1]
            + 1
        )
        input_channels_required = self.data_space_out.channels * n_patches

        # Construct the list of layers
        layers: list[nn.Module] = []

        # Optionally add an initial convolutional block at input resolution.
        # This will also set the correct number of channels if needed.
        if conv_subblocks_initial > 0:
            layers.append(
                CommonConvBlock(
                    self.data_space_in.channels,
                    input_channels_required,
                    kernel_size=conv_kernel_size,
                    activation=conv_activation,
                    n_subblocks=conv_subblocks_initial,
                ),
            )
        # Otherwise, if needed, set the number of channels with a 1x1 linear projection
        elif self.data_space_in.channels != input_channels_required:
            layers.append(
                nn.Conv2d(
                    self.data_space_in.channels, input_channels_required, kernel_size=1
                )
            )

        # Unflatten the channel dimension to extract the patches: [N, n_patches, C, patch_h, patch_w]
        layers.append(nn.Unflatten(1, (n_patches, -1)))

        # Flatten the patch dimensions: [N, n_patches, C * patch_area]
        layers.append(nn.Flatten(2, 4))

        # Permute dimensions: [N, C * patch_area, n_patches]
        layers.append(Permute((0, 2, 1)))

        # Fold patches into the output shape: [N, C, output_h, output_w]
        layers.append(
            NormalisedFold(
                output_size=self.data_space_out.shape,
                kernel_size=self.data_space_in.shape,
                stride=strides,
                padding=strides,
                use_hann_window=use_hann_window,
            )
        )

        # Optionally add a final convolutional block at output resolution
        if conv_subblocks_final > 0:
            layers.append(
                CommonConvBlock(
                    self.data_space_out.channels,
                    self.data_space_out.channels,
                    kernel_size=conv_kernel_size,
                    activation=conv_activation,
                    n_subblocks=conv_subblocks_final,
                ),
            )

        # Normalise the folded output before bounding it. We set affine=False to avoid
        # saturation that can cause the output to collapse to a constant prediction.
        if use_final_normalisation:
            layers.append(nn.BatchNorm2d(self.data_space_out.channels, affine=False))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space by combining patches.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        return self.model(x)
