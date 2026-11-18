from typing import Any

import torch

from cryocast.models.common import Freezable, Mask, RestrictRange, SkipConnection
from cryocast.types import (
    DataSpace,
    RangeRestriction,
    SkipAnchorType,
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
    ) -> None:
        """Initialise a BaseDecoder."""
        super().__init__()
        self.data_space_in = data_space_in
        self.data_space_out = data_space_out
        self.name = data_space_out.name

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

        This function is called by `rollout` after `forward`, so every decoder gets it
        automatically without having to call it themselves.

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
            x: TensorNCHW with (batch_size, n_latent_channels_total, latent_height,
                latent_width)

        Returns:
            TensorNCHW with (batch_size, output_channels, output_height, output_width)

        """
        msg = "If you are using the default rollout method, you must implement forward."
        raise NotImplementedError(msg)

    def rollout(
        self,
        x: TensorNTCHW,
        persistence: TensorNTCHW | None,
        *,
        anchor_type: SkipAnchorType = SkipAnchorType.INITIAL,
    ) -> TensorNTCHW:
        """Decode latent space into output space across multiple timesteps.

        The default implementation simply calls `self.forward` on each time slice
        simultaneously by reshaping the input to combine the batch and time dimensions,
        before reshaping back. The shared restrict-skip-mask steps are applied in here,
        so that concrete decoders only need to implement forward().

        Note that this (rollout method) also increases the effective batch size for any
        batch normalisation layers in the encoder.

        The anchor type sets which frame the skip connection combines with each step:

        - initial: every step uses `persistence` as its anchor.
        - previous: the first step uses `persistence` as its anchor. Each later step
            uses the finalised output of the step before it. The `forward` call is still
            batched over all timesteps. Only `finalise` runs once per step.

        The anchor type has no effect when the decoder has no skip connection.

        Args:
            x: TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total,
                latent_height, latent_width)
            persistence: TensorNTCHW with (batch_size, 1, output_channels,
                output_height, output_width) to use as the skip connection anchor
            anchor_type: SkipAnchorType that sets the anchor for each step after the
                first (default: INITIAL)

        Returns:
            TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)

        """
        # Pass the latents through the decoder to return to output space, combining the
        # batch and time dimensions for the decoder and then splitting them again.
        # Reading the shape from the input means that this works for any number of
        # timeslices, not only n_forecast_steps.
        batch_and_time = x.shape[:2]
        unbounded_ntchw: TensorNTCHW = self(x.flatten(0, 1)).unflatten(
            0, batch_and_time
        )

        # If we are anchoring each step on the output of the previous step we need to
        # loop over the timesteps.
        if self.skip_connection is not None and anchor_type == SkipAnchorType.PREVIOUS:
            anchor = None if persistence is None else persistence[:, 0]
            outputs: list[TensorNCHW] = []
            for t in range(unbounded_ntchw.shape[1]):
                anchor = self.finalise(unbounded_ntchw[:, t], anchor)
                outputs.append(anchor)
            return torch.stack(outputs, dim=1)

        # Otherwise, repeat the persistence timeslice across every step and apply the
        # shared finalisation steps (range restriction, skip connection, masking) to all
        # timesteps at once.
        anchor = (
            None
            if persistence is None
            else persistence.expand_as(unbounded_ntchw).flatten(0, 1)
        )
        return self.finalise(unbounded_ntchw.flatten(0, 1), anchor).unflatten(
            0, batch_and_time
        )
