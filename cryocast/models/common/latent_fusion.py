import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cryocast.types import TensorNTCHW


class _ZeroScoreHead(nn.Linear):
    """Initialise a neutral attention head without consuming random draws."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class LatentFusion(nn.Module):
    """Fuse multiple encoded data streams while preserving their channel layout.

    ``concat`` reproduces the existing CryoCast behaviour exactly. ``attention``
    learns a scalar importance weight for every input stream, sample and history
    timestep from a spatially pooled channel descriptor, then applies those weights
    before concatenating the streams.

    Attention scores are normalised across streams with a softmax and multiplied by
    the number of streams. The score heads are zero-initialised, so attention mode
    starts as an exact identity relative to concatenation and can learn away from that
    baseline during training without changing processor or decoder dimensions.
    """

    def __init__(
        self,
        input_channels: Sequence[int],
        *,
        mode: str = "concat",
        temperature: float = 1.0,
    ) -> None:
        """Initialise latent fusion for the configured encoder channel counts."""
        super().__init__()
        if not input_channels:
            msg = "input_channels must contain at least one stream."
            raise ValueError(msg)
        if any(
            isinstance(channels, bool)
            or not isinstance(channels, Integral)
            or channels <= 0
            for channels in input_channels
        ):
            msg = "All input channel counts must be positive integers."
            raise ValueError(msg)
        if mode not in {"concat", "attention"}:
            msg = f"Unknown fusion mode {mode!r}; expected 'concat' or 'attention'."
            raise ValueError(msg)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float, Real))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            msg = "temperature must be a finite positive real number."
            raise ValueError(msg)

        self.input_channels = tuple(int(channels) for channels in input_channels)
        self.mode = mode
        self.temperature = float(temperature)
        self.output_channels = sum(self.input_channels)

        self.score_heads = nn.ModuleList()
        if self.mode == "attention":
            for channels in self.input_channels:
                self.score_heads.append(_ZeroScoreHead(channels, 1))

    def _validate_inputs(self, inputs: Sequence[TensorNTCHW]) -> None:
        """Validate stream count, channels and non-channel dimensions."""
        if len(inputs) != len(self.input_channels):
            msg = (
                f"Expected {len(self.input_channels)} input streams, got {len(inputs)}."
            )
            raise ValueError(msg)

        reference_shape = inputs[0].shape
        if len(reference_shape) != 5:  # noqa: PLR2004
            msg = "Latent fusion expects NTCHW tensors with five dimensions."
            raise ValueError(msg)

        for idx, (tensor, expected_channels) in enumerate(
            zip(inputs, self.input_channels, strict=True)
        ):
            if tensor.ndim != 5:  # noqa: PLR2004
                msg = (
                    f"Input {idx} must be NTCHW with five dimensions, got "
                    f"{tensor.ndim}."
                )
                raise ValueError(msg)
            if tensor.shape[2] != expected_channels:
                msg = (
                    f"Input {idx} has {tensor.shape[2]} channels, expected "
                    f"{expected_channels}."
                )
                raise ValueError(msg)
            if (
                tensor.shape[:2] != reference_shape[:2]
                or tensor.shape[3:] != reference_shape[3:]
            ):
                msg = (
                    "All latent streams must have matching batch, time, height and "
                    "width dimensions."
                )
                raise ValueError(msg)

    def attention_weights(self, inputs: Sequence[TensorNTCHW]) -> Tensor:
        """Return per-stream weights with shape ``[batch, time, n_streams]``."""
        self._validate_inputs(inputs)
        first = inputs[0]
        if self.mode == "concat":
            return first.new_ones((*first.shape[:2], len(inputs)))

        # Pool, project and normalise in at least float32 even under autocast.
        # Casting only after a low-precision projection cannot undo overflow.
        heads = [cast("nn.Linear", head) for head in self.score_heads]
        score_dtype = (
            torch.float64
            if first.dtype == torch.float64 or heads[0].weight.dtype == torch.float64
            else torch.float32
        )
        with torch.autocast(device_type=first.device.type, enabled=False):
            logits = [
                F.linear(
                    tensor.to(score_dtype).mean(dim=(-1, -2)),
                    head.weight.to(score_dtype),
                    head.bias.to(score_dtype) if head.bias is not None else None,
                ).squeeze(-1)
                for tensor, head in zip(inputs, heads, strict=True)
            ]
            scores = torch.stack(logits, dim=-1)
            # Centre before temperature scaling so a common large score cannot
            # overflow into inf/inf for a small positive temperature.
            scores = scores - scores.amax(dim=-1, keepdim=True)
            weights = torch.softmax(scores / self.temperature, dim=-1)
        return weights * len(inputs)

    def forward(self, inputs: Sequence[TensorNTCHW]) -> TensorNTCHW:
        """Fuse encoded NTCHW streams into one channel-preserving latent tensor."""
        self._validate_inputs(inputs)
        if self.mode == "concat":
            return torch.cat(list(inputs), dim=2)

        weights = self.attention_weights(inputs)
        weighted_inputs = [
            tensor
            * weights[..., idx]
            .to(tensor.dtype)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            for idx, tensor in enumerate(inputs)
        ]
        return torch.cat(weighted_inputs, dim=2)
