from cryocast.models.common import Freezable
from cryocast.types import DataSpace, TensorNCHW, TensorNTCHW


class BaseDecoder(Freezable):
    """Decoder that takes data in a latent space and translates it to a larger output space.

    The decoder returns raw output. The Finaliser turns this into a valid forecast
    state, with range restriction, skip connection, relaxation and mask.

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    def __init__(self, *, data_space_in: DataSpace, data_space_out: DataSpace) -> None:
        """Initialise a BaseDecoder."""
        super().__init__()
        self.data_space_in = data_space_in
        self.data_space_out = data_space_out
        self.name = data_space_out.name

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height,
                latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        msg = "If you are using the default rollout method, you must implement forward."
        raise NotImplementedError(msg)

    def rollout(self, x: TensorNTCHW) -> TensorNTCHW:
        """Decode latent space into output space across multiple timesteps.

        The default implementation calls `self.forward` on all time slices at once, by
        combining the batch and time dimensions and then splitting them again.

        Note that this also increases the effective batch size for any batch
        normalisation layers in the decoder.

        Args:
            x: TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total,
                latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height,
            output_width)

        """
        return self(x.flatten(0, 1)).unflatten(0, x.shape[:2])
