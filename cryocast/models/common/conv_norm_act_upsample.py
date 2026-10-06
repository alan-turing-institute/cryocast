from torch import Tensor, nn

from .conv_norm_act import ConvNormAct


class ConvNormActUpsample(nn.Module):
    """Mini block: Upsample > ConvNormAct."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 2,
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
        norm_type: str = "batchnorm",
        padding: int | str = "same",
        upsample_factor: int = 2,
        upsample_mode: str = "nearest",
        stride: int = 1,
    ) -> None:
        """Initialize a ConvNormActUpsample mini-block.

        Args:
            activation: the activation function to use.
            dropout_rate: the dropout probability.
            in_channels: the number of input channels.
            kernel_size: the size of the convolutional kernel.
            norm_type: the type of normalization ("groupnorm", "batchnorm", or "none").
            out_channels: the number of output channels (if None, half of in_channels).
            padding: the padding to use for the convolution.
            stride: the stride to use for the convolution.
            upsample_factor: the factor by which to upsample spatial dimensions.
            upsample_mode: the mode to use for upsampling ("nearest", "bilinear", etc.).

        """
        super().__init__()

        out_channels = in_channels // 2 if out_channels is None else out_channels
        upsample_kwargs = (
            {"align_corners": False}
            if upsample_mode in ("bilinear", "bicubic", "trilinear")
            else {}
        )
        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=upsample_factor, mode=upsample_mode, **upsample_kwargs
            ),
            ConvNormAct(
                in_channels,
                out_channels,
                activation=activation,
                dropout_rate=dropout_rate,
                kernel_size=kernel_size,
                norm_type=norm_type,
                padding=padding,
                stride=stride,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
