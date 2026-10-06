from typing import Any

from torch import nn

from cryocast.types import TensorNCHW

from .base_processor import BaseProcessor


class NullProcessor(BaseProcessor):
    """Null model that simply returns input.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise a NullProcessor.

        Args:
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)
        self.model = nn.Identity()

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: return the most recent timestep in the window unchanged.

        A literal identity would return the whole window; instead a do-nothing model
        under the windowed rollout interface means "predict the most recent known
        timestep", i.e. persistence.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total * n_history_steps, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        """
        return self.model(x)[:, -self.data_space.channels :, :, :]
