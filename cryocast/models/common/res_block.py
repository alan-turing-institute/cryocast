"""Residual block: normalise -> (optional) attention -> GLU-gated feed-forward network."""

from typing import Any

from torch import Tensor, nn

from .glumb_conv import GLUMBConv
from .lite_mla import LiteMLA
from .normalisations import normalisation_from_name


class ResBlock(nn.Module):
    """Residual block following the EfficientViTBlock design.

    - (optional) attention subblock
        - normalise the input
        - attention to select spatial features
        - add residual connection to the input
    - feed-forward subblock
        - normalise the input
        - GLU-gated inverted-bottleneck convolution expanding channels by `ffn_factor`
        - add residual connection to the input
    """

    def __init__(
        self,
        channels: int,
        *,
        norm: str = "groupnorm",
        attention_heads: int | None = None,
        attention_scales: tuple[int, ...] = (5,),
        ffn_factor: int = 1,
        **conv_kwargs: Any,
    ) -> None:
        """Initialise a ResBlock."""
        super().__init__()

        # Optional attention sub-block
        self.attn_norm: nn.Module | None = (
            normalisation_from_name(norm, channels)
            if attention_heads is not None
            else None
        )
        self.attn: nn.Module | None = (
            None
            if attention_heads is None
            else LiteMLA(
                channels,
                heads=attention_heads,
                scales=attention_scales,
                padding_mode=conv_kwargs.get("padding_mode", "zeros"),
            )
        )

        # Feed-forward sub-block
        self.ffn_norm = normalisation_from_name(norm, channels)
        self.ffn = GLUMBConv(
            channels,
            expand_ratio=ffn_factor,
            residual_connection=False,  # residual connection is already present
            **conv_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.attn is not None and self.attn_norm is not None:
            x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))
