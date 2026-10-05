from typing import Any

from icenet_mp.models.common import Freezable, Mask, RestrictRange, SkipConnection
from icenet_mp.types import (
    DataSpace,
    RangeRestriction,
    SkipConnectionType,
    TensorNCHW,
    TensorNTCHW,
)


class BaseDecoder(Freezable):
    """Decoder that takes data in a latent space and translates it to a larger output space.

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        data_space_in: DataSpace,
        data_space_out: DataSpace,
        mask_dir: str | None = None,
        mask_type: str | None = None,
        restrict_range_max: float | None = None,
        restrict_range_min: float | None = None,
        restrict_range: str | None = None,
        skip_connection: dict[str, Any] | None = None,
        target_channel_offset: int | None = None,
        target_group_channels: int | None = None,
        target_variable_indices: list[int] | None = None,
    ) -> None:
        """Initialise a BaseDecoder."""
        super().__init__()
        self.data_space_in = data_space_in
        self.data_space_out = data_space_out
        self.name = data_space_out.name
        self.target_channel_offset = target_channel_offset
        self.target_group_channels = target_group_channels
        self.target_variable_indices = tuple(target_variable_indices or ())

        # The valid output range, used when finalising outputs
        self.range_min = 0 if restrict_range_min is None else restrict_range_min
        self.range_max = 1 if restrict_range_max is None else restrict_range_max

        # Initialise a skip connection if requested.
        skip_connection_cfg = dict(skip_connection or {})
        method = SkipConnectionType(
            skip_connection_cfg.pop("method", SkipConnectionType.NONE)
        )
        self.skip_connection = (
            SkipConnection(
                output_channels=self.data_space_out.channels,
                method=method,
                **skip_connection_cfg,
            )
            if method != SkipConnectionType.NONE
            else None
        )

        # An additive skip connection treats the model prediction as a residual to apply
        # on top of persistence. This means that the model predictions must be allowed
        # to be negative. We therefore bound to [-max, max] rather than [min, max].
        is_signed_residual = method == SkipConnectionType.ADDITIVE
        residual_bound = max(abs(self.range_min), abs(self.range_max))

        # Bound (none/sigmoid/clamp/tanh) the output into [min, max]
        self.restrict = RestrictRange(
            RangeRestriction(restrict_range or "none"),
            min_val=-residual_bound if is_signed_residual else self.range_min,
            max_val=residual_bound if is_signed_residual else self.range_max,
        )

        # Load the requested mask (ACTIVE/LAND/NONE)
        self.mask = Mask(
            mask_type=mask_type,
            output_shape=self.data_space_out.shape,
            mask_dir=mask_dir,
        )

    def finalise(self, x: TensorNCHW, persistence: TensorNCHW | None) -> TensorNCHW:
        """Apply shared output steps: range restriction, skip connection, and masking.

        Range restriction (none/sigmoid/tanh/clamp) must be applied first, so that all
        cells are at the same scale as the persistence and the mask. Without this, some
        masking methods would convert zeros into non-zero values (e.g. sigmoid(0)=0.5).

        This function is called once by `rollout` after the per-frame `forward`, so
        every decoder gets it automatically without having to call it themselves.

        RangeRestriction choices are: none/sigmoid/tanh/clamp
        """
        # Apply range restriction to bring the output into the required range
        bounded: TensorNCHW = self.restrict(x)

        if self.skip_connection:
            if persistence is None:
                msg = (
                    f"Decoder has skip_connection={self.skip_connection.method.value} "
                    "but no persistence input was provided."
                )
                raise ValueError(msg)
            # Combine the bounded output with the persistence input via skip connection,
            # then clamp before applying the mask to avoid out-of-bounds values.
            bounded = self.skip_connection(bounded, persistence).clamp(
                self.range_min, self.range_max
            )

        # Mask the output to zero out inactive/land cells. This must be applied after
        # range restriction so that masked cells are exactly 0. If range restriction
        # followed masking then e.g. sigmoid would give these cells non-zero values.
        return self.mask(bounded)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space for a single timestep.

        Args:
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height, latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        msg = "If you are using the default rollout method, you must implement forward."
        raise NotImplementedError(msg)

    def rollout(self, x: TensorNTCHW, persistence: TensorNTCHW | None) -> TensorNTCHW:
        """Decode latent space into output space across multiple timesteps.

        The default implementation simply calls `self.forward` on each time slice
        simultaneously by reshaping the input to combine the batch and time dimensions,
        before reshaping back. The shared restrict-skip-mask steps are applied in here,
        so that concrete decoders only need to implement forward().

        Note that this (rollout method) also increases the effective batch size for any
        batch normalisation layers in the encoder.

        Args:
            x: TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)
            persistence: TensorNTCHW with (batch_size, 1, output_channels, output_height, output_width) to add to every forecast step

        Returns:
            TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)

        """
        # Although this should only be called with n_forecast_steps timeslices, we can
        # make the decoder more generic by simply reading the number of timeslices from
        # the input.
        batch_size, n_timeslices = x.shape[0], x.shape[1]

        # Pass the latents through the decoder to return to output space
        unbounded_output_nchw: TensorNCHW = self(x.reshape(-1, *self.data_space_in.chw))

        # Repeat the persistence timeslice until we match the decoder output shape
        if persistence is not None:
            persistence = persistence.expand(-1, n_timeslices, -1, -1, -1).reshape(
                *unbounded_output_nchw.shape
            )

        # Apply the shared finalisation steps (range restriction, skip connection, masking)
        output = self.finalise(unbounded_output_nchw, persistence)

        # Reshape back to [batch, n_timeslices, C_out, H_out, W_out]
        return output.reshape(batch_size, n_timeslices, *self.data_space_out.chw)
