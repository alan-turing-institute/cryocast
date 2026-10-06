from torch import Tensor, cat, nn

from cryocast.types import SkipConnectionType

from .conv_norm_act import ConvNormAct


class SkipConnection(nn.Module):
    def __init__(
        self,
        output_channels: int,
        method: SkipConnectionType,
        *,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
    ) -> None:
        """Initialise a SkipConnection module."""
        super().__init__()

        self.method = method
        hidden_channels = hidden_channels or 16 * output_channels

        if self.method == SkipConnectionType.CONVOLUTIONAL:
            self.fusion = nn.Sequential(
                ConvNormAct(
                    2 * output_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    norm_type="groupnorm",
                ),
                nn.Conv2d(
                    hidden_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
            )

        if self.method == SkipConnectionType.GATED:
            self.gate = ConvNormAct(
                2 * output_channels,
                output_channels,
                activation="Sigmoid",
                kernel_size=kernel_size,
                norm_type="groupnorm",
            )

    def forward(self, main: Tensor, skip: Tensor) -> Tensor:
        """Apply a skip connection to the input tensor `main` using the `skip` tensor."""
        # A simple additive skip connection, treating `main` as the residual
        if self.method == SkipConnectionType.ADDITIVE:
            return main + skip

        # A learned fusion of the main and skip tensors. Concatenate the two on the
        # channel dimension, apply a ConvNormAct block then convolve back to the desired
        # output channels.
        if self.method == SkipConnectionType.CONVOLUTIONAL:
            return self.fusion(cat([main, skip], dim=1))

        # Apply a learned per-pixel gate that decides, at every pixel, how much to trust
        # the main and skip tensors.
        if self.method == SkipConnectionType.GATED:
            gate = self.gate(cat([main, skip], dim=1))
            return gate * main + (1 - gate) * skip

        msg = f"Unknown skip connection type: {self.method}"
        raise ValueError(msg)
