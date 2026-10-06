import torch
from torch import nn

from cryocast.types import TensorNCHW


class Shift(nn.Module):
    def __init__(self, *, scale: bool, offset: bool) -> None:
        """Apply a scale and offset to the input tensor."""
        super().__init__()
        self.scale = (
            nn.Parameter(torch.tensor([1.0]), requires_grad=True) if scale else None
        )
        self.offset = (
            nn.Parameter(torch.tensor([0.0]), requires_grad=True) if offset else None
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Apply a scale and offset to the input tensor."""
        if self.scale is not None:
            x = x * self.scale
        if self.offset is not None:
            x += self.offset
        return x
