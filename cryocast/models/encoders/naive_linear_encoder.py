from typing import Any

from torch import nn

from cryocast.models.common import ResizingInterpolation
from cryocast.types import DataSpace, TensorNCHW

from .base_encoder import BaseEncoder


class NaiveLinearEncoder(BaseEncoder):
    """Naive, linear encoder that takes data in an input space and translates it to a smaller latent space.

    Input space:
        TensorNTCHW with (batch_size, n_timeslices, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)
    """

    def __init__(self, data_space_in: DataSpace, **kwargs: Any) -> None:
        """Initialise a NaiveLinearEncoder."""
        super().__init__(
            output_channels=data_space_in.channels,
            data_space_in=data_space_in,
            **kwargs,
        )

        # Construct list of layers
        layers: list[nn.Module] = []

        # Start by normalising the input across height and width separately for each channel
        layers.append(nn.BatchNorm2d(self.data_space_in.channels))

        # Resize to the desired latent shape if needed
        if self.data_space_in.shape != self.data_space_out.shape:
            layers.append(ResizingInterpolation(self.data_space_out.shape))

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space with a linear transform.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
