import torch
from torch.utils.checkpoint import checkpoint

from cryocast.models.common import Freezable
from cryocast.models.common.normalisations import verify_no_batchnorm
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

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        data_space_out: DataSpace,
        decode_chunk_size: int | None = None,
    ) -> None:
        """Initialise a BaseDecoder.

        Args:
            data_space_in: The latent space that the decoder reads.
            data_space_out: The output space that the decoder writes.
            decode_chunk_size: If set, the default rollout decodes this number of
                frames (batch size * time slices) at a time, with activation
                checkpointing: the decoder does not keep its activations, and runs
                each chunk again in the backward pass. This decreases memory use for
                long rollouts, but makes training slower. If None, the rollout
                decodes all frames together.

        Raises:
            ValueError: If decode_chunk_size is not positive.

        """
        super().__init__()
        if decode_chunk_size is not None and decode_chunk_size < 1:
            msg = f"decode_chunk_size must be positive, got {decode_chunk_size}."
            raise ValueError(msg)
        self.decode_chunk_size = decode_chunk_size
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

        If `decode_chunk_size` is set, it decodes checkpointed chunks of this number
        of frames.

        Args:
            x: TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total,
                latent_height, latent_width)

        Returns:
            TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height,
            output_width)

        Raises:
            ValueError: If `decode_chunk_size` is set and the decoder has batch
                normalisation layers.

        """
        frames = x.flatten(0, 1)
        if self.decode_chunk_size is None:
            decoded = self(frames)
        else:
            verify_no_batchnorm(self, "decode_chunk_size")
            decoded = torch.cat(
                [
                    checkpoint(self, chunk, use_reentrant=False)
                    for chunk in frames.split(self.decode_chunk_size)
                ]
            )
        return decoded.unflatten(0, x.shape[:2])
