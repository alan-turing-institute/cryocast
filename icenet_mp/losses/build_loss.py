"""Instantiate a loss function from its Hydra config."""

from collections.abc import Mapping
from typing import Any

import hydra
from torch import nn

from .lead_time_weighted_loss import LeadTimeWeightedLoss


def build_loss(cfg: Mapping[str, Any]) -> nn.Module:
    """Instantiate a loss function from its config.

    Args:
        cfg: Hydra config for the loss function, which must include a `_target_` key
            specifying the loss class to instantiate. If a `lead_time_exponent` key is
            present, then the loss function will be wrapped in a LeadTimeWeightedLoss
            with that exponent.

    Returns:
        An instantiated loss function, which is a torch.nn.Module.

    """
    # Instantiate the loss function without the lead_time_exponent
    lead_time_exponent = cfg.get("lead_time_exponent")
    loss_fn = hydra.utils.instantiate(
        {k: v for k, v in cfg.items() if k != "lead_time_exponent"}
    )
    if not isinstance(loss_fn, nn.Module):
        msg = (
            f"Loss `_target_` {cfg.get('_target_', '(missing)')!r} created a "
            f"{type(loss_fn).__name__}, expected a torch.nn.Module."
        )
        raise TypeError(msg)
    # Optionally wrap the loss function in a LeadTimeWeightedLoss
    if lead_time_exponent is None:
        return loss_fn
    if isinstance(loss_fn, LeadTimeWeightedLoss):
        msg = (
            "`lead_time_exponent` cannot be combined with a LeadTimeWeightedLoss "
            "`_target_`, as this would wrap the loss twice."
        )
        raise TypeError(msg)
    return LeadTimeWeightedLoss(loss_fn, lead_time_exponent)
