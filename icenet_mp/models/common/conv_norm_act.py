from torch import Tensor, nn

from .activations import ACTIVATION_FROM_NAME
from .normalisations import normalisation_from_name


class ConvNormAct(nn.Module):
    """Mini block: Convolution > Normalization > Activation > [optional] Dropout."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
        groups: int = 1,
        norm_type: str = "batchnorm",
        padding: int | str = "same",
        stride: int = 1,
    ) -> None:
        """Initialise a ConvNormAct mini-block.

        Args:
            activation: Name of the activation function (from ACTIVATION_FROM_NAME).
            dropout_rate: Dropout probability. If 0.0, dropout is not applied.
            groups: Number of groups for a grouped convolution (see nn.Conv2d).
            in_channels: Input channel size.
            kernel_size: Kernel size for the convolution.
            norm_type: Type of normalization to apply ("batchnorm", "channelnorm",
                "groupnorm" or "none").
            out_channels: Output channel size.
            padding: the padding to use for the convolution.
            stride: the stride to use for the convolution.

        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
            ),
            normalisation_from_name(norm_type, out_channels),
            ACTIVATION_FROM_NAME[activation](),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvNormAct block to input tensor."""
        return self.block(x)
