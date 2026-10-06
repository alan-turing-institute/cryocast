from typing import Any

from torch import nn

from cryocast.models.common import CommonConvBlock, Permute
from cryocast.types import DataSpace, TensorNCHW

from .base_encoder import BaseEncoder


class PiecewiseEncoder(BaseEncoder):
    """Piecewise encoder that splits data from an input space into smaller patches in a latent space.

    - Optional initial convolutional block at input resolution
    - Split into patches of size latent_height x latent_width
    - Optional final convolutional block at output resolution

    Input space:
        TensorNTCHW with (batch_size, n_timeslices, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        latent_space: tuple[int, int],
        conv_activation: str = "SiLU",
        conv_kernel_size: int = 3,
        conv_subblocks_initial: int = 3,
        conv_subblocks_final: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialise a PiecewiseEncoder."""
        # Calculate the number of patches required
        # We set the stride to be half the patch size to ensure overlap, which will
        # capture more of the spatial structure of the data.
        strides = tuple(max(1, patch_size // 2) for patch_size in latent_space)
        n_patches = (
            (data_space_in.shape[0] + 2 * strides[0] - 1 * (latent_space[0] - 1) - 1)
            // strides[0]
            + 1
        ) * (
            (data_space_in.shape[1] + 2 * strides[1] - 1 * (latent_space[1] - 1) - 1)
            // strides[1]
            + 1
        )

        # Initialise the base class with the correct number of output channels
        output_channels = data_space_in.channels * n_patches
        super().__init__(
            data_space_in=data_space_in,
            latent_space=latent_space,
            output_channels=output_channels,
            **kwargs,
        )

        # Construct the list of layers
        layers: list[nn.Module] = []

        # Optionally add an initial convolutional block at input resolution.
        if conv_subblocks_initial > 0:
            layers.append(
                CommonConvBlock(
                    self.data_space_in.channels,
                    self.data_space_in.channels,
                    kernel_size=conv_kernel_size,
                    activation=conv_activation,
                    n_subblocks=conv_subblocks_initial,
                ),
            )

        layers += [
            # Unfold into patches of size data_space_out.shape: [N, C * patch_area, n_patches]
            nn.Unfold(
                self.data_space_out.shape,
                stride=strides,
                padding=strides,
            ),
            # Permute dimensions: [N, n_patches, C * patch_area]
            Permute((0, 2, 1)),
            # Unflatten the patches: [N, n_patches, C, patch_h, patch_w]
            nn.Unflatten(2, (-1, *self.data_space_out.shape)),
            # Combine into a single channel dimension: [N, C * n_patches, patch_h, patch_w]
            nn.Flatten(1, 2),
        ]

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

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space by splitting into patches.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
