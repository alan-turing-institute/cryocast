"""Inverted-bottleneck FFN with a depthwise spatial convolution and a GLU gate."""

from typing import Any

import torch
from torch import Tensor, nn


class GLUMBConv(nn.Module):
    """Inverted-bottleneck FFN with a depthwise spatial convolution and a GLU gate.

    Follows the implementation from the HuggingFace diffusers Sana transformer:
    (https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/sana_transformer.py)

    - 1x1 expanding channel projection
    - SiLU activation
    - depthwise spatial convolution with no cross-channel mixing
    - GLU gate to select spatial features
    - 1x1 projecting channel projection back to the original number of channels
    - (optional) residual connection to the input

    """

    def __init__(
        self,
        channels: int,
        *,
        expand_ratio: float = 1,
        residual_connection: bool = True,
        **conv_kwargs: Any,
    ) -> None:
        """Initialise a GLUMBConv.

        Args:
            channels: the number of input and output channels.
            expand_ratio: the bottleneck expansion factor.
            residual_connection: whether to add the input to the output.
            **conv_kwargs: forwarded to the depthwise spatial convolution
                (e.g. ``kernel_size``, ``padding``, ``padding_mode``).

        """
        super().__init__()
        hidden_channels = int(expand_ratio * channels)
        self.residual_connection = residual_connection

        self.nonlinearity = nn.SiLU()
        # 1x1 channel projection to twice the hidden channels
        self.conv_inverted = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        # Depthwise spatial convolution with no cross-channel mixing
        self.conv_depth = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            groups=hidden_channels * 2,
            **conv_kwargs,
        )
        # 1x1 channel projection back to the original number of channels
        self.conv_point = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the inverted-bottleneck GLU feed-forward block to x."""
        residual = x if self.residual_connection else None
        x = self.conv_inverted(x)
        x = self.nonlinearity(x)
        x = self.conv_depth(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = x * self.nonlinearity(gate)
        x = self.conv_point(x)
        if residual is not None:
            x = x + residual
        return x
