import math

import torch
from torch import nn

from cryocast.types import TensorNCHW


class ExponentialRelaxation(nn.Module):
    def __init__(self, tau: float) -> None:
        """Relax each step of a rollout towards an equilibrium state.

        At each step, the state x moves towards the equilibrium c:

            x_k = (1 - lambda) * x_{k-1} + lambda * c_k, with lambda = 1 - exp(-1 / tau)

        Since 0 < lambda <= 1, the output will always be between x and c.

        If the equilibrium is constant, the anomaly x - c decays exponentially:

            x_k = c + exp(-k / tau) * (x_0 - c)

        Args:
            tau: The folding time of the anomaly, in rollout steps. Must be positive.

        """
        super().__init__()
        if not tau > 0:
            msg = f"tau must be positive, got {tau}."
            raise ValueError(msg)
        self.tau = float(tau)
        self.rate = 1.0 - math.exp(-1.0 / self.tau)

    def forward(self, x: TensorNCHW, equilibrium: TensorNCHW) -> TensorNCHW:
        """Move one step towards the equilibrium state.

        Args:
            x: The state for one step, with shape [B, C, H, W].
            equilibrium: The equilibrium for the same step, with the same shape as x.

        """
        if equilibrium.shape != x.shape:
            msg = (
                f"equilibrium shape {tuple(equilibrium.shape)} does not match "
                f"state shape {tuple(x.shape)}."
            )
            raise ValueError(msg)
        # torch.lerp needs the same dtype for both inputs. Under mixed precision, x can
        # be bf16 and the equilibrium float32. Use the wider dtype: in bf16, a small
        # step towards the equilibrium can round to no change.
        dtype = torch.promote_types(x.dtype, equilibrium.dtype)
        return torch.lerp(x.to(dtype), equilibrium.to(dtype), self.rate)
