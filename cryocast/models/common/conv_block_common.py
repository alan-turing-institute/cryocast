"""Full convolutional block consisting of multiple stacked ConvNormAct mini-blocks.

The first mini-block changes the number of channels, subsequent blocks keep it constant.

Example usage:

# GroupNorm without dropout, no final layer (2 blocks)
block_gn = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.0,
)

# BatchNorm without dropout, no final layer (2 blocks)
block_bn = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="batchnorm",
    dropout_rate=0.0,
)

# No normalization, no dropout
block_none = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="none",
    dropout_rate=0.0,
)

# GroupNorm with extra final layer (GroupNorm with 3 blocks)
block_final = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.0,
    n_subblocks=3
)

# GroupNorm with dropout (10%) but no final layer
block_dropout = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.1,
)

# GroupNorm with both final layer and 10% dropout (Bottleneck-style)
block_bottleneck = CommonConvBlock(
    in_channels=64,
    out_channels=128,
    norm_type="groupnorm",
    dropout_rate=0.1,
    n_subblocks=3
)
"""

from torch import Tensor, nn

from .conv_norm_act import ConvNormAct


class CommonConvBlock(nn.Module):
    """Full convolutional block consisting of multiple stacked ConvNormAct mini-blocks."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
        kernel_size: int,
        n_subblocks: int = 2,
        norm_type: str = "batchnorm",
    ) -> None:
        """Initialise a CommonConvBlock.

        Args:
            activation: Name of the activation function (from ACTIVATION_FROM_NAME).
            dropout_rate: Dropout probability for each ConvNormAct block.
            in_channels: Input channel size.
            kernel_size: Kernel size for the convolutions.
            n_subblocks: Number of ConvNormAct blocks to stack (default 2).
            norm_type: Type of normalization to apply ("batchnorm", "channelnorm",
                "groupnorm" or "none").
            out_channels: Output channel size.

        """
        super().__init__()

        # Create stacked ConvNormAct blocks
        # First block changes the number of channels, subsequent blocks keep it constant
        self.block = nn.Sequential(
            *(
                ConvNormAct(
                    in_channels if idx_subblock == 0 else out_channels,
                    out_channels,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for idx_subblock in range(n_subblocks)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply stacked ConvNormAct layers to input tensor."""
        return self.block(x)
