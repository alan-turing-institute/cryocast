import torch
from torch import Tensor, nn


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatial latent-state evolution."""

    def __init__(
        self, in_channels: int, hidden_channels: int, kernel_size: int
    ) -> None:
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

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
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
