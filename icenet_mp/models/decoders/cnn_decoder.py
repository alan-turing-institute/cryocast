import logging
from typing import Any

import torch
from torch import nn

from icenet_mp.models.common import ConvBlockUpsample, ResizingInterpolation
from icenet_mp.types import TensorNCHW

from .base_decoder import BaseDecoder

logger = logging.getLogger(__name__)


class CNNDecoder(BaseDecoder):
    """Decoder that uses a convolutional neural net (CNN) to translate latent space back to data space.

    - Increase size with interpolation (if needed)
    - n_layers of size-increasing convolutional blocks
    - Decrease size with interpolation (if needed)
    - Convolve to number of output channels (if needed)

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_timeslices, output_channels, output_height, output_width)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        activation: str = "ReLU",
        kernel_size: int = 3,
        n_layers: int = 3,
        n_subblocks: int = 2,
        norm_type: str = "batchnorm",
        scale_factor: int = 2,
        zero_init_output: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialise a CNNDecoder.

        Args:
            activation: activation function used inside the upsampling blocks.
            kernel_size: kernel size of the convolutional blocks.
            n_layers: number of size-increasing convolutional blocks.
            n_subblocks: ConvNormAct blocks per upsampling block.
            norm_type: normalisation inside the blocks ("groupnorm", "batchnorm", "none").
            scale_factor: spatial upscaling per block.
            zero_init_output: if True, the final output convolution starts with zero
                weights and bias, so an untrained decoder emits exactly zero. Used by
                residual (tendency) models so that the initial prediction is exactly
                the anchor field passed to the skip connection.
            **kwargs: forwarded to ``BaseDecoder`` (spaces, masks, range, skip).

        """
        super().__init__(**kwargs)

        # Calculate the factor by which the scale changes after n_layers
        layer_factor = scale_factor**n_layers

        # Ensure number of channels is divisible by the factor implied by n_layers
        if self.data_space_in.channels % layer_factor:
            msg = (
                f"The number of input channels {self.data_space_in.channels} must be divisible by {layer_factor}. "
                f"Without this, it is not possible to apply {n_layers} convolutions."
            )
            raise ValueError(msg)

        # Calculate the smallest input shape that would produce an output at least as
        # large as the desired output shape. Note that this may not be exact, since we
        # double the size at each layer.
        minimal_input_shape = (
            -(self.data_space_out.shape[0] // -layer_factor),
            -(self.data_space_out.shape[1] // -layer_factor),
        )

        # Construct list of layers
        layers: list[nn.Module] = []
        logger.debug("CNNDecoder (%s) with %d layers", self.name, n_layers)

        # If necessary, resize until we reach the minimal input shape. This ensures that
        # the post-convolution shape will be at least as large as the desired output
        # shape so any further resizing will be a size decrease.
        shape = (
            max(minimal_input_shape[0], self.data_space_in.shape[0]),
            max(minimal_input_shape[1], self.data_space_in.shape[1]),
        )
        if shape != self.data_space_in.shape:
            layers.append(ResizingInterpolation(shape))
            logger.debug(
                "- ResizingInterpolation from %s to %s",
                self.data_space_in.shape,
                shape,
            )

        # Add n_layers size-increasing convolutional blocks
        n_channels = self.data_space_in.channels
        for _ in range(n_layers):
            layers.append(
                ConvBlockUpsample(
                    n_channels,
                    activation=activation,
                    kernel_size=kernel_size,
                    n_subblocks=n_subblocks,
                    norm_type=norm_type,
                    scale_factor=scale_factor,
                )
            )
            logger.debug(
                "- ConvBlockUpsample (%s, %s) with %d channels",
                activation,
                kernel_size,
                n_channels,
            )
            n_channels //= scale_factor
            shape = (shape[0] * scale_factor, shape[1] * scale_factor)

        # If necessary, resize downwards to match the output shape
        if shape != self.data_space_out.shape:
            layers.append(ResizingInterpolation(self.data_space_out.shape))
            logger.debug(
                "- ResizingInterpolation from %s to %s",
                shape,
                self.data_space_out.shape,
            )

        # Run a final convolution that will both ensure that we have the right number of
        # output channels and also the the final operation is a layer that can produce
        # values across the full output range (not just the range of the activation
        # function in the last ConvBlockUpsample).
        layers.append(nn.Conv2d(n_channels, self.data_space_out.channels, 1))
        logger.debug(
            "- Channel convolution from %d to %d",
            n_channels,
            self.data_space_out.channels,
        )

        # Combine the layers sequentially
        self.model = nn.Sequential(*layers)

        # If zero_init_output is requested then zero the weights in the final
        # convolution layer so that the untrained decoder outputs exactly zero.
        if zero_init_output:
            if isinstance(final := self.model[-1], nn.Conv2d):
                with torch.no_grad():
                    final.weight.zero_()
                    if final.bias is not None:
                        final.bias.zero_()
            else:
                msg = "zero_init_output needs the final decoder layer to be a Conv2d."
                raise ValueError(msg)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: decode latent space into output space with a CNN.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.model(x)
