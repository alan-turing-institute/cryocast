"""Processor based on the translator from SimVPv2.

Reference:
    SimVP: Simpler yet Better Video Prediction
    (Gao et al., 2022) [https://arxiv.org/abs/2206.05099]
    SimVPv2: Towards Simple yet Powerful Spatiotemporal Predictive Learning
    (Tan et al., 2022) [https://arxiv.org/abs/2211.12509]

We take the temporal translator from SimVPv2: a flat stack of "gSTA" (Gated
Spatiotemporal Attention) blocks replacing the original SimVP Inception-based U-Net
structure.

Our implementation of GatedAttentionBlock is slightly modified to allow different
numbers of input and output channels.
"""

import logging
from typing import Any

from torch import nn

from cryocast.models.common import GatedAttentionBlock
from cryocast.types import ProcessorOutput, TensorNCHW, TensorNTCHW

from .base_processor import BaseProcessor

log = logging.getLogger(__name__)


class GSTAProcessor(BaseProcessor):
    """Processor based on the translator from SimVPv2.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dilation: int = 3,
        drop_path_prob: float = 0.1,
        hid_channels: int = 256,
        kernel_size: int = 21,
        mlp_drop_prob: float = 0.0,
        mlp_ratio: float = 4.0,
        n_blocks: int = 8,
        **kwargs: Any,
    ) -> None:
        """Initialise a GSTAProcessor.

        Args:
            dilation: Dilation used by gated attention blocks in the translator.
            drop_path_prob: Probability to drop the residual branch of each gSTA block.
            hid_channels: Hidden channel count for the temporal translator.
            kernel_size: Kernel size for gated attention blocks in the translator.
            mlp_drop_prob: Element-wise dropout probability for the translator conv-MLP blocks.
            mlp_ratio: Hidden-channel expansion ratio for the translator conv-MLP blocks.
            n_blocks: Number of gated attention blocks in the temporal translator.
            kwargs: Arguments to BaseProcessor.

        """
        super().__init__(**kwargs)
        if n_blocks < 1:
            msg = f"n_blocks must be at least 1, got {n_blocks}"
            raise ValueError(msg)
        if n_blocks == 1:
            log.warning("GSTAProcessor.n_blocks=1: no hidden channels will be used.")

        # Translator: a flat stack of gated attention blocks
        self.translator = nn.Sequential(
            *(
                GatedAttentionBlock(
                    (
                        self.n_history_steps * self.data_space.channels
                        if idx == 0
                        else hid_channels
                    ),
                    (
                        self.n_forecast_steps * self.data_space.channels
                        if idx == n_blocks - 1
                        else hid_channels
                    ),
                    dilation=dilation,
                    drop_path_prob=drop_path_prob,
                    kernel_size=kernel_size,
                    mlp_drop_prob=mlp_drop_prob,
                    mlp_ratio=mlp_ratio,
                )
                for idx in range(n_blocks)
            )
        )

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        """Rollout the processor over a window of history/forecast timesteps."""
        # Get the initial shape of the input tensor
        n, _, c, h, w = x.shape

        # Fold history timesteps into the channel dimension
        x_nchw = x.flatten(start_dim=1, end_dim=2)

        # Apply the translator to get the forecast timesteps
        prediction_nchw: TensorNCHW = self.translator(x_nchw)

        # Unfold the forecast timesteps from the channel dimension
        prediction = prediction_nchw.reshape(n, -1, c, h, w)

        return ProcessorOutput(prediction=prediction)
