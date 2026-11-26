from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from cryocast.models.common import (
    ExponentialRelaxation,
    Freezable,
    Mask,
    RestrictRange,
    SkipConnection,
)
from cryocast.types import (
    DataSpace,
    RangeRestriction,
    SkipAnchorType,
    SkipConnectionType,
    TensorNCHW,
    TensorNTCHW,
)


class Finaliser(Freezable):
    """Turn raw decoder output into a valid forecast state.

    For each forecast step, the finaliser applies these steps in this order:

    1. Range restriction (none/sigmoid/tanh/clamp) of the raw decoder output.
    2. Skip connection with an anchor, then a clamp into the valid range.
    3. Relaxation towards an equilibrium, if one is given.
    4. The mask, so that masked cells are exactly 0.

    Range restriction must come first, so that all cells are at the same scale as the
    anchor and the equilibrium. The mask must come last. If range restriction came after
    the mask, some methods would convert zeros into non-zero values (for example,
    sigmoid(0) = 0.5).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        data_space: DataSpace,
        mask_dir: str | Path | None = None,
        mask_type: str | None = None,
        restrict_range: str | None = None,
        restrict_range_max: float | None = None,
        restrict_range_min: float | None = None,
        skip_connection: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise a Finaliser.

        Args:
            data_space: The output data space.
            mask_dir: Directory that contains the masks. Required if `mask_type` is set.
            mask_type: Mask to apply: "active", "land", or None for no mask.
            restrict_range: How to bound the raw output: "none", "sigmoid", "tanh" or
                "clamp". None means "none".
            restrict_range_max: Maximum valid value (default 1).
            restrict_range_min: Minimum valid value (default 0).
            skip_connection: Skip connection options. The `method` key is one of
                "none", "additive", "convolutional" or "gated". The other keys go to
                SkipConnection.

        """
        super().__init__()

        # The valid output range
        self.range_min = 0 if restrict_range_min is None else restrict_range_min
        self.range_max = 1 if restrict_range_max is None else restrict_range_max

        # Configure a skip connection if one is requested
        skip_connection_cfg = dict(skip_connection or {})
        method = SkipConnectionType(
            skip_connection_cfg.pop("method", SkipConnectionType.NONE)
        )
        self.skip_connection = (
            SkipConnection(
                output_channels=data_space.channels,
                method=method,
                **skip_connection_cfg,
            )
            if method != SkipConnectionType.NONE
            else None
        )

        # An additive skip connection treats the raw output as a residual to add to the
        # anchor. The residual must be able to be negative, so we bound it to
        # [-max, max] rather than [min, max].
        is_signed_residual = method == SkipConnectionType.ADDITIVE
        residual_bound = max(abs(self.range_min), abs(self.range_max))
        self.restrict = RestrictRange(
            RangeRestriction(restrict_range or "none"),
            min_val=-residual_bound if is_signed_residual else self.range_min,
            max_val=residual_bound if is_signed_residual else self.range_max,
        )

        # Load the requested mask (ACTIVE/LAND/NONE)
        self.mask = Mask(
            mask_type=mask_type, output_shape=data_space.shape, mask_dir=mask_dir
        )

        # Relaxation has no weights, so the owning model sets it from its own options
        self.relaxation: ExponentialRelaxation | None = None

    def forward(
        self,
        x: TensorNCHW,
        *,
        anchor: TensorNCHW | None = None,
        equilibrium: TensorNCHW | None = None,
    ) -> TensorNCHW:
        """Turn raw decoder output for one step into a valid forecast state.

        Args:
            x: TensorNCHW raw decoder output for one step.
            anchor: TensorNCHW skip connection anchor. Required if the finaliser has a
                skip connection.
            equilibrium: TensorNCHW to relax towards, or None for no relaxation.
                Relaxation applies only if `relaxation` is set.

        Raises:
            ValueError: If the finaliser has a skip connection but there is no anchor.

        """
        # Apply range restriction to bring the output into the required range
        bounded: TensorNCHW = self.restrict(x)

        if self.skip_connection:
            if anchor is None:
                msg = (
                    f"Finaliser has skip_connection="
                    f"{self.skip_connection.method.value} but no anchor was provided."
                )
                raise ValueError(msg)
            # Combine the output with the anchor, then clamp into the valid range
            bounded = self.skip_connection(bounded, anchor).clamp(
                self.range_min, self.range_max
            )

        # Relax towards the equilibrium. The result is between two values in the valid
        # range, so it is also in the valid range.
        if self.relaxation is not None and equilibrium is not None:
            bounded = self.relaxation(bounded, equilibrium)

        # Mask last, so that masked cells are exactly 0
        return self.mask(bounded)

    def rollout(
        self,
        x: TensorNTCHW,
        *,
        anchor: TensorNTCHW | None = None,
        anchor_type: SkipAnchorType = SkipAnchorType.INITIAL,
        equilibrium: TensorNTCHW | None = None,
    ) -> TensorNTCHW:
        """Turn raw decoder output for several steps into valid forecast states.

        The anchor type sets which anchor the skip connection uses for each step:

        - initial: every step uses `anchor`.
        - previous: the first step uses `anchor`. Each later step uses the output of
          the step before it, so this runs once per step.

        The anchor type has no effect if the finaliser has no skip connection.

        If `equilibrium` is set, each step is relaxed towards the same step of
        `equilibrium`. With a PREVIOUS anchor, the relaxed step is the next anchor.

        Args:
            x: TensorNTCHW raw decoder output with (batch_size, n_timeslices,
                output_channels, output_height, output_width).
            anchor: TensorNTCHW with (batch_size, 1, output_channels, output_height,
                output_width) to use as the skip connection anchor, or None. With an
                INITIAL anchor type, it can also have n_timeslices steps, to give each
                step its own anchor.
            anchor_type: SkipAnchorType that sets the anchor for each step after the
                first (default: INITIAL).
            equilibrium: TensorNTCHW with the same shape as `x` to relax towards, or
                None for no relaxation. Relaxation applies only if `relaxation` is set.

        Returns:
            TensorNTCHW with the same shape as `x`.

        """
        # With a PREVIOUS anchor, each step depends on the step before it
        if self.skip_connection is not None and anchor_type == SkipAnchorType.PREVIOUS:
            anchor_nchw = None if anchor is None else anchor[:, 0]
            outputs: list[TensorNCHW] = []
            for t in range(x.shape[1]):
                anchor_nchw = self(
                    x[:, t],
                    anchor=anchor_nchw,
                    equilibrium=None if equilibrium is None else equilibrium[:, t],
                )
                outputs.append(anchor_nchw)
            return torch.stack(outputs, dim=1)

        # Otherwise, repeat the anchor across every step and process all steps at once
        batch_and_time = x.shape[:2]
        return self(
            x.flatten(0, 1),
            anchor=None if anchor is None else anchor.expand_as(x).flatten(0, 1),
            equilibrium=None if equilibrium is None else equilibrium.flatten(0, 1),
        ).unflatten(0, batch_and_time)
