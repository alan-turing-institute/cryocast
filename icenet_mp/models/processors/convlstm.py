from typing import Any

import torch
from torch import Tensor, nn

from icenet_mp.types import ProcessorOutput, TensorNTCHW

from .base_processor import BaseProcessor


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatial latent-state evolution."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int) -> None:
        """Initialise a ConvLSTM cell."""
        super().__init__()
        if in_channels <= 0:
            msg = "in_channels must be greater than 0."
            raise ValueError(msg)
        if hidden_channels <= 0:
            msg = "hidden_channels must be greater than 0."
            raise ValueError(msg)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            msg = "kernel_size must be a positive odd integer."
            raise ValueError(msg)

        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Advance hidden and cell state by one timestep."""
        hidden, cell = state
        input_gate, forget_gate, candidate, output_gate = self.gates(
            torch.cat((x, hidden), dim=1)
        ).chunk(4, dim=1)

        input_gate = input_gate.sigmoid()
        forget_gate = forget_gate.sigmoid()
        candidate = candidate.tanh()
        output_gate = output_gate.sigmoid()

        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * next_cell.tanh()
        return next_hidden, next_cell


class ConvLSTMProcessor(BaseProcessor):
    """Forecast latent fields using a stacked convolutional LSTM.

    The processor consumes the history sequence one timestep at a time, retaining a
    spatial hidden/cell state. Forecasts are then produced autoregressively: the most
    recent prediction is fed back as the next ConvLSTM input. Unlike processors that
    flatten the history window into channels, this keeps the temporal recurrence
    explicit while preserving the standard IceNet-MP NTCHW processor contract.
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
        self, x: Tensor, states: list[tuple[Tensor, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        """Advance all recurrent layers by one timestep."""
        next_states: list[tuple[Tensor, Tensor]] = []
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
        if x.ndim != 5:
            msg = f"Expected NTCHW input with 5 dimensions, got shape {tuple(x.shape)}."
            raise ValueError(msg)
        if x.shape[1] != self.n_history_steps:
            msg = (
                f"Expected {self.n_history_steps} history steps, got {x.shape[1]}."
            )
            raise ValueError(msg)
        if x.shape[2] != self.data_space.channels:
            msg = (
                f"Expected {self.data_space.channels} latent channels, got {x.shape[2]}."
            )
            raise ValueError(msg)

        current = x[:, 0]
        states = self._initial_states(current)
        for time_idx in range(self.n_history_steps):
            current = x[:, time_idx]
            states = self._step(current, states)

        predictions: list[Tensor] = []
        for forecast_idx in range(self.n_forecast_steps):
            next_frame = self.output_projection(states[-1][0])
            if self.residual:
                next_frame = current + next_frame
            predictions.append(next_frame)
            current = next_frame
            if forecast_idx < self.n_forecast_steps - 1:
                states = self._step(current, states)

        return ProcessorOutput(prediction=torch.stack(predictions, dim=1))
