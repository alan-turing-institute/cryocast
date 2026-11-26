from collections.abc import Callable
from typing import Any, TypeVar

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .normalisations import verify_no_batchnorm

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
    module: nn.Module, frames: torch.Tensor, chunk_size: int | None, option: str
) -> torch.Tensor:
    """Apply a module to chunks of frames, with activation checkpointing.

    Args:
        module: The module to apply to each chunk.
        frames: Tensor with the frames on the first dimension.
        chunk_size: The number of frames in each chunk. If None, apply the module
            once to all frames, without checkpointing.
        option: The name of the chunking option, for the error message.

    Returns:
        The outputs for all chunks, concatenated on the first dimension.

    Raises:
        ValueError: If chunk_size is set and the module has batch normalisation layers.

    """
    if chunk_size is None:
        return module(frames)
    verify_no_batchnorm(module, option)
    return torch.cat(
        [
            checkpoint_activations(module, chunk, enabled=True)
            for chunk in frames.split(chunk_size)
        ]
    )
