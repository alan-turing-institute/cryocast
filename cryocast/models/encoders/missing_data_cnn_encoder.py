"""CNN encoder support for conditioning data with missing values."""

import math
from numbers import Real
from typing import Any

import torch
from torch import nn

from cryocast.types import TensorNCHW

from .cnn_encoder import CNNEncoder


class _IdentityInputAdapter(nn.Conv2d):
    """Start with an identity on observed channels without consuming RNG state."""

    def reset_parameters(self) -> None:
        """Initialize the data channels as identity and availability weight as zero."""
        nn.init.dirac_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MissingDataCNNEncoder(CNNEncoder):
    """CNN encoder that preserves an explicit signal for missing observations.

    Non-finite values and complete time slices filled with an optional missing-date
    sentinel are replaced with zero before convolution. Individual finite values
    equal to the sentinel remain observations in otherwise available slices. A
    per-pixel availability channel is appended. A trainable 1x1 adapter maps the augmented input back to the original channel count so
    the existing CNN architecture and latent-space contract remain unchanged.

    During training, complete time slices can also be dropped at random. Since
    ``BaseEncoder.rollout`` folds batch and time together before calling ``forward``,
    this conditioning dropout is sampled independently for each history time slice.
    """

    conditioning_dropout_probability: float
    missing_fill_value: float | None
    input_adapter: nn.Conv2d

    def __init__(
        self,
        *,
        conditioning_dropout_probability: float = 0.0,
        missing_fill_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the missing-data-aware CNN encoder."""
        if (
            isinstance(conditioning_dropout_probability, bool)
            or not isinstance(conditioning_dropout_probability, Real)
            or not math.isfinite(conditioning_dropout_probability)
            or not 0.0 <= conditioning_dropout_probability <= 1.0
        ):
            msg = "conditioning_dropout_probability must be between 0 and 1 and finite."
            raise ValueError(msg)
        if missing_fill_value is not None and (
            isinstance(missing_fill_value, bool)
            or not isinstance(missing_fill_value, Real)
            or not math.isfinite(missing_fill_value)
        ):
            msg = "missing_fill_value must be a finite real number or None."
            raise ValueError(msg)

        super().__init__(**kwargs)
        self.conditioning_dropout_probability = conditioning_dropout_probability
        self.missing_fill_value = missing_fill_value

        channels = self.data_space_in.channels
        # The availability weight starts at zero. Unlike a random Conv2d followed
        # by zeroing, this also preserves subsequent seeded model initialization.
        self.input_adapter = _IdentityInputAdapter(
            channels + 1, channels, kernel_size=1
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Encode one time slice while retaining an observation-availability signal."""
        available = torch.isfinite(x)
        if self.missing_fill_value is not None:
            # The missing-date loader fills the entire frame. Do not discard an
            # isolated legitimate measurement merely because it equals the sentinel.
            missing_frame = (x == self.missing_fill_value).flatten(1).all(dim=1)
            available = available & ~missing_frame[:, None, None, None]

        availability = available.to(dtype=x.dtype).mean(dim=1, keepdim=True)
        clean = torch.where(available, x, torch.zeros_like(x))

        if self.training and self.conditioning_dropout_probability > 0.0:
            drop_mask = (
                torch.rand(
                    (x.shape[0], 1, 1, 1),
                    device=x.device,
                )
                < self.conditioning_dropout_probability
            )
            clean = clean.masked_fill(drop_mask, 0.0)
            availability = availability.masked_fill(drop_mask, 0.0)

        augmented = torch.cat((clean, availability), dim=1)
        return super().forward(self.input_adapter(augmented))
