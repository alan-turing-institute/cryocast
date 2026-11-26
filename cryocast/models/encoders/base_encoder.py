from collections.abc import Callable
from functools import cached_property

import torch

from cryocast.models.common import Freezable, checkpointed_chunks
from cryocast.types import DataSpace, TensorNCHW, TensorNTCHW


class BaseEncoder(Freezable):
    """Encoder that takes data in an input space and translates it to a smaller latent space.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_history_steps, latent_channels, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        data_space_in: DataSpace,
        latent_space: tuple[int, int],
        output_channels: int,
        encode_chunk_size: int | None = None,
        latitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
        longitudes_fn: Callable[[], dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialise a BaseEncoder.

        Args:
            data_space_in: The input space that the encoder reads.
            latent_space: The (height, width) of the latent space.
            output_channels: The number of latent channels that the encoder writes.
            encode_chunk_size: If set, the default rollout encodes this number of
                frames (batch size * time slices) at a time, with activation
                checkpointing: the encoder does not keep its activations, and runs
                each chunk again in the backward pass. This decreases memory use
                when the encoder encodes many time slices with gradients, but makes
                training slower. If None, the rollout encodes all frames together.
            latitudes_fn: Function that returns the latitudes of the input space.
            longitudes_fn: Function that returns the longitudes of the input space.

        Raises:
            ValueError: If encode_chunk_size is not positive.

        """
        super().__init__()
        if encode_chunk_size is not None and encode_chunk_size < 1:
            msg = f"encode_chunk_size must be positive, got {encode_chunk_size}."
            raise ValueError(msg)
        self.encode_chunk_size = encode_chunk_size
        self.data_space_in = data_space_in
        self.data_space_out = DataSpace(
            name=f"latent_space_{data_space_in.name}",
            channels=output_channels,
            shape=latent_space,
        )
        self.latitudes_fn = latitudes_fn
        self.longitudes_fn = longitudes_fn
        self.name = data_space_in.name

    @cached_property
    def latitudes(self) -> dict[str, list[float]]:
        return {} if not self.latitudes_fn else self.latitudes_fn()

    @cached_property
    def longitudes(self) -> dict[str, list[float]]:
        return {} if not self.longitudes_fn else self.longitudes_fn()

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        msg = "If you are using the default rollout method, you must implement forward."
        raise NotImplementedError(msg)

    def rollout(self, x: TensorNTCHW) -> TensorNTCHW:
        """Encode input space into latent space across multiple timesteps.

        The default implementation calls `self.forward` on all time slices at once, by
        combining the batch and time dimensions and then splitting them again.

        Note that this also increases the effective batch size for any batch
        normalisation layers in the encoder.

        If `encode_chunk_size` is set, it encodes checkpointed chunks of this number
        of frames.

        Args:
            x: TensorNTCHW with (batch_size, n_timeslices, input_channels, input_height, input_width)

        Returns:
            TensorNTCHW with (batch_size, n_timeslices, latent_channels, latent_height, latent_width)

        Raises:
            ValueError: If `encode_chunk_size` is set and the encoder has batch
                normalisation layers.

        """
        frames = x.flatten(0, 1)
        if self.encode_chunk_size is None:
            encoded = self(frames)
        else:
            encoded = checkpointed_chunks(
                self, frames, self.encode_chunk_size, "encode_chunk_size"
            )
        return encoded.unflatten(0, x.shape[:2])

    def verify_output_channels(
        self, device: torch.device | None = None, dtype: torch.dtype = torch.float32
    ) -> None:
        """Cross-check that `forward` actually produces `data_space_out.channels`.

        Runs a single real forward pass on a zero input at construction time, so a
        wrong or stale channel declaration fails immediately.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                dummy_input = torch.zeros(
                    1,
                    *self.data_space_in.chw,
                    dtype=dtype,
                    device=device or torch.device("cpu"),
                )
                actual_channels = self(dummy_input).shape[1]
        finally:
            self.train(was_training)
        if actual_channels != self.data_space_out.channels:
            msg = (
                f"{type(self).__name__} ('{self.name}') declared "
                f"{self.data_space_out.channels} output channel(s) but forward() "
                f"actually produced {actual_channels}."
            )
            raise ValueError(msg)
