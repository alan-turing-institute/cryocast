from collections.abc import Sequence
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from cryocast.types import ProcessorOutput, TensorNTCHW

from .base_processor import BaseProcessor


class MixtureOfExpertsProcessor(BaseProcessor):
    """Blend complete latent forecasts with a learned, per-sample gate.

    The gate uses only the most recent observed latent frame. Each expert runs its
    full forecast from the same history, and one softmax distribution blends all
    leads and channels. This is a dense mixture: every expert is evaluated.

    Only prediction-only experts are supported. Future targets are never passed to
    experts, and custom latent-loss objectives require a separate routing policy.
    """

    def __init__(
        self,
        *,
        experts: Sequence[DictConfig],
        gate_hidden_channels: int = 64,
        **kwargs: Any,
    ) -> None:
        """Instantiate prediction-only experts with a common forecast contract."""
        super().__init__(**kwargs)
        if not experts:
            msg = "MixtureOfExpertsProcessor requires at least one expert."
            raise ValueError(msg)
        if (
            isinstance(gate_hidden_channels, bool)
            or not isinstance(gate_hidden_channels, int)
            or gate_hidden_channels <= 0
        ):
            msg = "gate_hidden_channels must be a positive integer."
            raise ValueError(msg)
        if self.computes_loss_in_latent_space:
            msg = "MixtureOfExpertsProcessor is prediction-only; no latent loss is defined."
            raise ValueError(msg)

        shared_kwargs = {
            "data_space": self.data_space,
            "data_space_target": self.data_space_target,
            "n_forecast_steps": self.n_forecast_steps,
            "n_history_steps": self.n_history_steps,
        }
        # Preserve the target-slice contract where present, without requiring it
        # when combined-latent processors no longer expose an offset.
        if (offset := getattr(self, "target_channel_offset", None)) is not None:
            shared_kwargs["target_channel_offset"] = offset
        instantiated: list[BaseProcessor] = []
        for expert_cfg in experts:
            expert = hydra.utils.instantiate(expert_cfg, **shared_kwargs)
            if not isinstance(expert, BaseProcessor):
                msg = (
                    "Mixture expert must inherit BaseProcessor, got "
                    f"{type(expert).__name__}."
                )
                raise TypeError(msg)
            if expert.computes_loss_in_latent_space:
                msg = (
                    "MixtureOfExpertsProcessor does not support experts that compute "
                    "their own latent-space loss."
                )
                raise ValueError(msg)
            instantiated.append(expert)

        self.experts = nn.ModuleList(instantiated)
        self.gate = nn.Sequential(
            nn.Linear(self.data_space.channels, gate_hidden_channels),
            nn.SiLU(),
            nn.Linear(gate_hidden_channels, len(self.experts)),
        )

    def expert_weights(self, x: TensorNTCHW) -> torch.Tensor:
        """Return per-sample weights, with gate arithmetic in at least float32."""
        if x.ndim != 5 or tuple(x.shape[1:]) != (  # noqa: PLR2004
            self.n_history_steps,
            *self.data_space.chw,
        ):
            msg = (
                f"MoE history has shape {tuple(x.shape)}; expected "
                f"(batch, {self.n_history_steps}, {self.data_space.chw})."
            )
            raise ValueError(msg)
        first = cast("nn.Linear", self.gate[0])
        last = cast("nn.Linear", self.gate[2])
        dtype = torch.promote_types(torch.float32, x.dtype)
        dtype = torch.promote_types(dtype, first.weight.dtype)
        # Promote before pooling and both projections. A float32 softmax cannot
        # repair logits which already overflowed in an autocast/half linear layer.
        with torch.autocast(device_type=x.device.type, enabled=False):
            pooled = x[:, -1].to(dtype=dtype).mean(dim=(-2, -1))
            hidden = F.silu(
                F.linear(
                    pooled,
                    first.weight.to(dtype=dtype),
                    None if first.bias is None else first.bias.to(dtype=dtype),
                )
            )
            logits = F.linear(
                hidden,
                last.weight.to(dtype=dtype),
                None if last.bias is None else last.bias.to(dtype=dtype),
            )
            return torch.softmax(logits, dim=-1)

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        """Run every expert from history only, validate, then blend full forecasts."""
        del y  # Prediction-only experts must not receive future observations.
        weights = self.expert_weights(x)
        expected = (x.shape[0], self.n_forecast_steps, *self.data_space.chw)
        predictions = []
        for index, module in enumerate(self.experts):
            expert = cast("BaseProcessor", module)
            output = expert.rollout(x)
            if output.loss is not None:
                msg = "Mixture experts must return prediction-only ProcessorOutput."
                raise ValueError(msg)
            if tuple(output.prediction.shape) != expected:
                msg = (
                    f"Expert {index} returned shape {tuple(output.prediction.shape)}; "
                    f"expected {expected}."
                )
                raise ValueError(msg)
            predictions.append(output.prediction)

        stacked = torch.stack(predictions, dim=1)
        dtype = torch.promote_types(stacked.dtype, weights.dtype)
        weighted = (
            stacked.to(dtype=dtype)
            * weights.to(dtype=dtype)[:, :, None, None, None, None]
        )
        return ProcessorOutput(prediction=weighted.sum(dim=1).to(dtype=stacked.dtype))
