from collections.abc import Sequence
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from cryocast.types import ProcessorOutput, TensorNTCHW

from .base_processor import BaseProcessor


class MixtureOfExpertsProcessor(BaseProcessor):
    """Blend multiple latent-space forecasting processors with a learned gate.

    Expert weights are predicted per sample from a spatially pooled representation of
    the most recent latent history frame. All experts receive the same input history,
    and their forecast tensors are combined with the learned softmax weights.

    The initial implementation supports experts that return predictions only. Processors
    that own a custom latent-space training loss are rejected because mixing those losses
    would require a separate training objective and routing policy.
    """

    def __init__(
        self,
        *,
        experts: Sequence[DictConfig],
        gate_hidden_channels: int = 64,
        **kwargs: Any,
    ) -> None:
        """Initialise the mixture and instantiate experts with the shared data spaces."""
        super().__init__(**kwargs)
        if not experts:
            msg = "MixtureOfExpertsProcessor requires at least one expert."
            raise ValueError(msg)
        if gate_hidden_channels <= 0:
            msg = "gate_hidden_channels must be greater than 0."
            raise ValueError(msg)

        shared_kwargs = {
            "data_space": self.data_space,
            "data_space_target": self.data_space_target,
            "n_forecast_steps": self.n_forecast_steps,
            "n_history_steps": self.n_history_steps,
            "target_channel_offset": self.target_channel_offset,
        }
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
        """Return per-sample softmax weights for each expert."""
        latest = x[:, -1]
        pooled = latest.mean(dim=(-2, -1))
        return torch.softmax(self.gate(pooled), dim=-1)

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:
        """Run all experts and return their gated weighted forecast."""
        weights = self.expert_weights(x)
        predictions = []
        for module in self.experts:
            expert = cast("BaseProcessor", module)
            output = expert.rollout(x, y)
            if output.loss is not None:
                msg = "Mixture experts must return prediction-only ProcessorOutput."
                raise ValueError(msg)
            predictions.append(output.prediction)

        stacked = torch.stack(predictions, dim=1)
        weighted = stacked * weights[:, :, None, None, None, None]
        return ProcessorOutput(prediction=weighted.sum(dim=1))
