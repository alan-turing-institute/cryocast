from collections.abc import Sequence

import torch
from torch import Tensor, nn

from cryocast.types import TensorNTCHW


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
        if any(channels <= 0 for channels in input_channels):
            msg = "All input channel counts must be greater than zero."
            raise ValueError(msg)
        if mode not in {"concat", "attention"}:
            msg = f"Unknown fusion mode {mode!r}; expected 'concat' or 'attention'."
            raise ValueError(msg)
        if temperature <= 0:
            msg = "temperature must be greater than zero."
            raise ValueError(msg)

        self.input_channels = tuple(int(channels) for channels in input_channels)
        self.mode = mode
        self.temperature = float(temperature)
        self.output_channels = sum(self.input_channels)

        self.score_heads = nn.ModuleList()
        if self.mode == "attention":
            for channels in self.input_channels:
                head = nn.Linear(channels, 1)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                self.score_heads.append(head)

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

        logits = [
            head(tensor.mean(dim=(-1, -2))).squeeze(-1)
            for tensor, head in zip(inputs, self.score_heads, strict=True)
        ]
        weights = torch.softmax(
            torch.stack(logits, dim=-1) / self.temperature,
            dim=-1,
        )
        return weights * len(inputs)

    def forward(self, inputs: Sequence[TensorNTCHW]) -> TensorNTCHW:
        """Fuse encoded NTCHW streams into one channel-preserving latent tensor."""
        self._validate_inputs(inputs)
        if self.mode == "concat":
            return torch.cat(list(inputs), dim=2)

        weights = self.attention_weights(inputs)
        weighted_inputs = [
            tensor * weights[..., idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            for idx, tensor in enumerate(inputs)
        ]
        return torch.cat(weighted_inputs, dim=2)
