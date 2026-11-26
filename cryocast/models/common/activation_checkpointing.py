from collections.abc import Callable
from typing import Any, TypeVar

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

T = TypeVar("T")


def checkpoint_activations(function: Callable[..., T], *args: Any, enabled: bool) -> T:
    """Call a function, with activation checkpointing if enabled.

    With checkpointing, the function does not keep its activations. It runs again in
    the backward pass. The output and the gradients do not change, but less memory is
    necessary.

    Args:
        function: The function to call.
        *args: The arguments for the function.
        enabled: If True, use activation checkpointing.

    Returns:
        The output of the function.

    """
    if enabled:
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


def checkpointed_chunks(
    module: nn.Module, frames: torch.Tensor, chunk_size: int | None
) -> torch.Tensor:
    """Apply a module to chunks of frames, with activation checkpointing.

    The module must not have batch normalisation. `EncodeProcessDecode` checks this
    when it is constructed.

    Args:
        module: The module to apply to each chunk.
        frames: Tensor with the frames on the first dimension.
        chunk_size: The number of frames in each chunk. If None, apply the module
            once to all frames, without checkpointing.

    Returns:
        The outputs for all chunks, concatenated on the first dimension.

    """
    if chunk_size is None:
        return module(frames)
    return torch.cat(
        [
            checkpoint_activations(module, chunk, enabled=True)
            for chunk in frames.split(chunk_size)
        ]
    )


def verify_no_batchnorm(module: nn.Module, option: str) -> None:
    """Raise a ValueError if the module has batch normalisation.

    Activation checkpointing runs the forward pass again in the backward pass. This
    would update the running statistics of batch normalisation layers a second time.

    Args:
        module: The module to check, with its submodules.
        option: The name of the checkpointing option, for the error message.

    """
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in module.modules()):
        msg = (
            f"{option} cannot be used with batch normalisation in "
            f"{type(module).__name__}. Use a different norm_type, for example groupnorm."
        )
        raise ValueError(msg)
