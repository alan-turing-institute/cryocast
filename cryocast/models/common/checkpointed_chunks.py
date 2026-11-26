import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .normalisations import verify_no_batchnorm


def checkpointed_chunks(
    module: nn.Module, frames: torch.Tensor, chunk_size: int, option: str
) -> torch.Tensor:
    """Apply a module to chunks of frames, with activation checkpointing.

    The module does not keep its activations. It runs each chunk again in the
    backward pass. The output and the gradients are the same as for one call on all
    frames, but less memory is necessary.

    Args:
        module: The module to apply to each chunk.
        frames: Tensor with the frames on the first dimension.
        chunk_size: The number of frames in each chunk.
        option: The name of the chunking option, for the error message.

    Returns:
        The outputs for all chunks, concatenated on the first dimension.

    Raises:
        ValueError: If the module has batch normalisation layers.

    """
    verify_no_batchnorm(module, option)
    return torch.cat(
        [
            checkpoint(module, chunk, use_reentrant=False)
            for chunk in frames.split(chunk_size)
        ]
    )
