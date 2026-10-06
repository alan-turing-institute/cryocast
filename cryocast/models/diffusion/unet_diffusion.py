"""UNetDiffusion: Conditional U-Net for DDPM-based Forecasting.

Description:
    U-Net architecture for use in conditional denoising diffusion probabilistic models (DDPM),
    designed for geophysical forecasting tasks such as sea ice concentration prediction.
    Inputs include noisy predictions, diffusion timestep embeddings, and meteorological
    conditioning inputs. Supports configurable number of forecast days and output classes via constructor parameters.

"""

import math

import torch
from torch import nn

from cryocast.models.common import (
    CommonConvBlock,
    ConvNormActUpsample,
    TimeEmbed,
)
from cryocast.types import TensorNCHW


class UNetDiffusion(nn.Module):
    """U-Net architecture for conditional DDPM-based forecasting.

    Inputs include noisy predictions, time step embeddings, and conditioning inputs.
    Supports configurable depth, filter size, and number of forecast days/classes.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_channels: int,
        output_channels: int,
        *,
        timesteps: int = 1000,
        kernel_size: int = 3,
        start_out_channels: int = 64,
        time_embed_dim: int = 256,
        normalization: str = "groupnorm",
        activation: str = "SiLU",
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize the U-Net diffusion model.

        Args:
            input_channels (int): Number of input conditioning channels (e.g., meteorological variables).
            output_channels (int): Number of output channels for the denoised forecast.
            timesteps (int, optional): Number of diffusion timesteps. Defaults to 1000.
            kernel_size (int, optional): Convolution kernel size used across all layers. Defaults to 3.
            start_out_channels (int, optional): Number of filters in the first convolutional block. Defaults to 64.
            time_embed_dim (int, optional): Dimensionality of the diffusion timestep embedding. Defaults to 256.
            normalization (str, optional): Normalization type to apply ("groupnorm", "batchnorm", "layernorm", etc.).
            activation (str, optional): Activation function to use ("SiLU", "ReLU", "LeakyReLU", etc.).
            dropout_rate (float, optional): Dropout probability applied to convolutional blocks. Defaults to 0.1.

        """
        super().__init__()

        self.timesteps = timesteps
        self.time_embed_dim = time_embed_dim

        if kernel_size <= 0:
            msg = "Kernel size must be greater than 0."
            raise ValueError(msg)

        if start_out_channels <= 0:
            msg = "Start out channels must be greater than 0."
            raise ValueError(msg)

        # Time embedding
        self.time_embed = TimeEmbed(self.time_embed_dim)

        # Channel calculations
        channels = [start_out_channels * 2**i for i in range(4)]

        # Encoder
        # input_channels C*T (n_input_channels * n_input_days)
        # output_channels C*T (n_output_classes * n_forecast_days)
        self.conv1 = CommonConvBlock(
            in_channels=input_channels + output_channels,
            out_channels=channels[0],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
        )
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = CommonConvBlock(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
        )
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = CommonConvBlock(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
        )
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4 = CommonConvBlock(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
        )
        self.maxpool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = CommonConvBlock(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        # Decoder
        self.up6 = ConvNormActUpsample(
            in_channels=channels[3],
            out_channels=channels[2],
            norm_type=normalization,
            activation=activation,
        )
        self.up7 = ConvNormActUpsample(
            in_channels=channels[2],
            out_channels=channels[2],
            norm_type=normalization,
            activation=activation,
        )
        self.up8 = ConvNormActUpsample(
            in_channels=channels[2],
            out_channels=channels[1],
            norm_type=normalization,
            activation=activation,
        )
        self.up9 = ConvNormActUpsample(
            in_channels=channels[1],
            out_channels=channels[0],
            norm_type=normalization,
            activation=activation,
        )

        self.up6b = CommonConvBlock(
            in_channels=channels[3] + self.time_embed_dim,
            out_channels=channels[2],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
        )
        self.up7b = CommonConvBlock(
            in_channels=channels[3] + self.time_embed_dim,
            out_channels=channels[2],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
        )
        self.up8b = CommonConvBlock(
            in_channels=channels[2] + self.time_embed_dim,
            out_channels=channels[1],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
        )
        self.up9b = CommonConvBlock(
            in_channels=channels[1] + self.time_embed_dim,
            out_channels=channels[0],
            kernel_size=kernel_size,
            norm_type=normalization,
            activation=activation,
            n_subblocks=3,
        )

        # Final layer
        self.final_layer = nn.Conv2d(
            channels[0], output_channels, kernel_size=1, padding="same"
        )

    def forward(
        self,
        noise: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the U-Net diffusion model.

        Args:
            noise (torch.Tensor): Noisy forecast tensor of shape [B, H, W, n_classes, n_forecast_days].
            t (torch.Tensor): Diffusion timestep tensor of shape [B].
            conditioning (torch.Tensor): Conditioning input tensor of shape [B, H, W, input_channels].

        Returns:
            torch.Tensor: Predicted denoised forecast of shape [B, H, W, n_classes, n_forecast_days].

        """
        # Time embedding
        t = self._timestep_embedding(t)
        t = self.time_embed(t)

        # Concatenate with conditional input
        noise = torch.cat([noise, conditioning], dim=1)  # [b,(d*c)+input_channels,h,w]

        # Encoder pathway
        bn1 = self.conv1(noise)
        conv1 = self.maxpool1(bn1)
        bn2 = self.conv2(conv1)
        conv2 = self.maxpool2(bn2)
        bn3 = self.conv3(conv2)
        conv3 = self.maxpool3(bn3)
        bn4 = self.conv4(conv3)
        conv4 = self.maxpool4(bn4)

        # Bottleneck
        bn5 = self.conv5(conv4)

        # Decoder with time embedding
        up6 = self.up6(bn5)
        up6 = torch.cat([bn4, up6], dim=1)
        up6 = self._add_time_embedding(up6, t)
        up6 = self.up6b(up6)

        up7 = self.up7(up6)
        up7 = torch.cat([bn3, up7], dim=1)
        up7 = self._add_time_embedding(up7, t)
        up7 = self.up7b(up7)

        up8 = self.up8(up7)
        up8 = torch.cat([bn2, up8], dim=1)
        up8 = self._add_time_embedding(up8, t)
        up8 = self.up8b(up8)

        up9 = self.up9(up8)
        up9 = torch.cat([bn1, up9], dim=1)
        up9 = self._add_time_embedding(up9, t)
        up9 = self.up9b(up9)

        return self.final_layer(up9)  # [b, c_out, h, w]

    def _timestep_embedding(
        self, timesteps: torch.Tensor, dim: int = 256, max_period: int = 10000
    ) -> torch.Tensor:
        """Converts timestep integers into sinusoidal positional embeddings.

        Args:
            timesteps (torch.Tensor): Timestep tensor [B].
            dim (int): Embedding dimension.
            max_period (int): Frequency range.

        Returns:
            torch.Tensor: Embedding tensor of shape [B, dim].

        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding

    def _add_time_embedding(self, x: TensorNCHW, t: torch.Tensor) -> TensorNCHW:
        """Concatenates time embedding across spatial dimensions.

        Args:
            x (torch.Tensor): Feature map tensor [B, C, H, W].
            t (torch.Tensor): Time embedding tensor [B, D].

        Returns:
            torch.Tensor: Time-conditioned feature map [B, C+D, H, W].

        """
        _, _, h, w = x.shape
        t = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

        return torch.cat([x, t], dim=1)
