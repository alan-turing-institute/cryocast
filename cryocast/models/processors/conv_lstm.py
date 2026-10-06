from typing import Any

import torch
from torch import Tensor, nn

from cryocast.models.common import ConvLSTMCell
from cryocast.types import ProcessorOutput, TensorNCHW, TensorNTCHW

from .base_processor import BaseProcessor


class ConvLSTMProcessor(BaseProcessor):
    """Forecast latent fields using a stacked convolutional LSTM.

    The processor consumes the history sequence one timestep at a time, retaining a
    spatial hidden/cell state. Forecasts are then produced autoregressively: the most
    recent prediction is fed back as the next ConvLSTM input. Unlike processors that
    flatten the history window into channels, this keeps the temporal recurrence
    explicit while preserving the standard CryoCast NTCHW processor contract.
    """

    def __init__(
        self,
        *,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        n_layers: int = 2,
        dropout: float = 0.0,
        residual: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise a ConvLSTM processor."""
        super().__init__(**kwargs)
        if hidden_channels <= 0:
            msg = "hidden_channels must be greater than 0."
            raise ValueError(msg)
        if n_layers <= 0:
            msg = "n_layers must be greater than 0."
            raise ValueError(msg)
        if not 0.0 <= dropout < 1.0:
            msg = "dropout must be in the range [0, 1)."
            raise ValueError(msg)

        self.hidden_channels = hidden_channels
        self.residual = residual
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    self.data_space.channels if layer_idx == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size,
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout2d(dropout)
        self.output_projection = nn.Conv2d(
            hidden_channels, self.data_space.channels, kernel_size=1
        )

    def _initial_states(self, x: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Create zero hidden/cell states matching an input frame."""
        batch, _, height, width = x.shape
        return [
            (
                x.new_zeros(batch, self.hidden_channels, height, width),
                x.new_zeros(batch, self.hidden_channels, height, width),
            )
            for _ in self.cells
        ]

    def _step(
        self, x: TensorNCHW, states: list[tuple[TensorNCHW, TensorNCHW]]
    ) -> list[tuple[TensorNCHW, TensorNCHW]]:
        """Advance all recurrent layers by one timestep."""
        next_states: list[tuple[TensorNCHW, TensorNCHW]] = []
        layer_input = x
        for layer_idx, (cell, state) in enumerate(zip(self.cells, states, strict=True)):
            next_state = cell(layer_input, state)
            next_states.append(next_state)
            layer_input = next_state[0]
            if layer_idx < len(self.cells) - 1:
                layer_input = self.dropout(layer_input)
        return next_states

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        """Consume history and autoregressively forecast future latent frames."""
        # Initialise hidden states with zeros for every layer
        current: TensorNCHW = x[:, 0]
        states = self._initial_states(current)

        # Consume the history timesteps one at a time to initialise the recurrent state
        for idx_history in range(x.shape[1]):
            current = x[:, idx_history]
            states = self._step(current, states)

        predictions: list[Tensor] = []
        for idx_forecast in range(self.n_forecast_steps):
            # Convolve the most recent hidden state to latent space
            next_frame = self.output_projection(states[-1][0])

            # Optionally combine the prediction with the most recent frame
            if self.residual:
                next_frame = current + next_frame

            # Add the prediction to the list of forecast timesteps
            predictions.append(next_frame)

            # Add the prediction as the next hidden state
            if idx_forecast < self.n_forecast_steps - 1:
                current = next_frame
                states = self._step(current, states)

        # Stack the forecast timesteps into NTCHW
        return ProcessorOutput(prediction=torch.stack(predictions, dim=1))
