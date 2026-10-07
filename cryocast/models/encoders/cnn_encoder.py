import logging
from typing import Any

from torch import nn

from cryocast.models.common import ConvBlockDownsample, ResizingInterpolation
from cryocast.types import DataSpace, TensorNCHW

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class CNNEncoder(BaseEncoder):
    """Encoder that uses a convolutional neural net (CNN) to translate data to a latent space.

    - Resize with interpolation (if needed)
    - n_layers of size-reducing convolutional blocks

    Input space:
        TensorNTCHW with (batch_size, n_timeslices, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        activation: str = "ReLU",
        data_space_in: DataSpace,
        kernel_size: int = 3,
        n_layers: int = 3,
        n_subblocks: int = 2,
        scale_factor: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialise a CNNEncoder."""
        # Initialise the base class with the correct number of output channels
        output_channels = data_space_in.channels * scale_factor**n_layers
        super().__init__(
            data_space_in=data_space_in,
            output_channels=output_channels,
            **kwargs,
        )

        # Construct list of layers
        layers: list[nn.Module] = []
        logger.debug("CNNEncoder (%s) with %d layers", self.name, n_layers)

        # If necessary, apply a convolutional resizing to get the correct input dimensions
        initial_required_shape = (
            self.data_space_out.shape[0] * (scale_factor**n_layers),
            self.data_space_out.shape[1] * (scale_factor**n_layers),
        )
        if self.data_space_in.shape != initial_required_shape:
            layers.append(ResizingInterpolation(initial_required_shape))
            logger.debug(
                "- ResizingInterpolation from %s to %s",
                self.data_space_in.shape,
                initial_required_shape,
            )

        # Add n_layers size-reducing convolutional blocks
        n_channels = self.data_space_in.channels
        for _ in range(n_layers):
            layers.append(
                ConvBlockDownsample(
                    n_channels,
                    activation=activation,
                    kernel_size=kernel_size,
                    n_subblocks=n_subblocks,
                    scale_factor=scale_factor,
                )
            )
            logger.debug(
                "- ConvBlockDownsample (%s, %s) with %d channels",
                activation,
                kernel_size,
                n_channels,
            )
            n_channels *= scale_factor

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space with a CNN.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
