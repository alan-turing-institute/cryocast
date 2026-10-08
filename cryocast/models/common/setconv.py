"""Set convolution from irregular point observations to a regular spatial grid.

The continuous SetConv input is a set of (position, value) pairs. A Gaussian
kernel produces a sampling-density channel and density-normalised value channels.
A CNN can then operate on the resulting gridded representation.

Locations are provided for every sample and time step, so moving sensors such
as Argo floats do not require a fixed station grid.
"""

import math

import torch
from torch import nn

_COORDINATE_CHANNELS = 2
_GRID_RANK = 3
_TIME_SERIES_RANK = 4


class SetConv(nn.Module):
    """Map irregular two-dimensional observations onto a regular query grid.

    The output consists of a density channel followed by normalized
    kernel-weighted signals. For a query location q and valid input points
    (p_i, v_i), the weights are exp(-||q-p_i||² / (2 * lengthscale²)), the
    density is their sum, and each signal channel is the weighted mean.

    Coordinates must share a *planar distance metric* (preferably a local
    projected CRS, e.g. EASE2). Raw global latitude/longitude degrees do not
    define Euclidean distances and are not appropriate for this kernel.
    """

    def __init__(
        self,
        lengthscale: float,
        *,
        learnable: bool = True,
        chunk_size: int = 2048,
        eps: float = 1e-8,
    ) -> None:
        """Initialize a Gaussian SetConv.

        Args:
            lengthscale: Initial Gaussian kernel length scale in coordinate units.
            learnable: Whether to optimize the log length scale.
            chunk_size: Maximum number of grid queries evaluated at once.
            eps: Density floor for safe normalization.

        """
        super().__init__()
        if not math.isfinite(lengthscale) or lengthscale <= 0:
            msg = "lengthscale must be positive and finite."
            raise ValueError(msg)
        if chunk_size <= 0:
            msg = "chunk_size must be positive."
            raise ValueError(msg)
        if not math.isfinite(eps) or eps <= 0:
            msg = "eps must be positive and finite."
            raise ValueError(msg)

        log_scale = torch.tensor(math.log(lengthscale), dtype=torch.float32)
        if learnable:
            self.log_lengthscale = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_lengthscale", log_scale)
        self.chunk_size = chunk_size
        self.eps = eps

    @staticmethod
    def _validate_shapes(
        values: torch.Tensor,
        positions: torch.Tensor,
        grid_positions: torch.Tensor,
    ) -> None:
        """Reject unsupported point/query tensor geometries."""
        if values.ndim not in (3, _TIME_SERIES_RANK):
            msg = "values must have shape (B, N, C) or (B, T, N, C)."
            raise ValueError(msg)
        if positions.shape != (*values.shape[:-1], _COORDINATE_CHANNELS):
            msg = "positions must match values batch/time/point axes and end in 2."
            raise ValueError(msg)
        if (
            grid_positions.ndim != _GRID_RANK
            or grid_positions.shape[-1] != _COORDINATE_CHANNELS
        ):
            msg = "grid_positions must have shape (H, W, 2)."
            raise ValueError(msg)
        if min(values.shape[0], values.shape[-1], *grid_positions.shape[:2]) <= 0:
            msg = "Batch size, channels and query grid dimensions must be positive."
            raise ValueError(msg)
        if values.ndim == _TIME_SERIES_RANK and values.shape[1] <= 0:
            msg = "Time dimension must be nonempty."
            raise ValueError(msg)

    @staticmethod
    def _make_mask(
        values: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Return the validated batch/time/observation validity mask."""
        expected_shape = values.shape[:3]
        if valid_mask is None:
            return torch.ones(expected_shape, dtype=torch.bool, device=values.device)
        if valid_mask.dtype != torch.bool:
            msg = "valid_mask must have boolean dtype."
            raise TypeError(msg)
        if valid_mask.shape != expected_shape:
            msg = f"valid_mask must have shape {expected_shape}."
            raise ValueError(msg)
        if valid_mask.device != values.device:
            msg = "valid_mask and values must be on the same device."
            raise ValueError(msg)
        return valid_mask

    @staticmethod
    def _validate_finite(
        values: torch.Tensor,
        positions: torch.Tensor,
        grid_positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Allow masked NaNs but refuse non-finite valid measurements."""
        if not torch.isfinite(values[valid_mask]).all():
            msg = "Unmasked observation values must be finite."
            raise ValueError(msg)
        if not torch.isfinite(positions[valid_mask]).all():
            msg = "Unmasked observation positions must be finite."
            raise ValueError(msg)
        if not torch.isfinite(grid_positions).all():
            msg = "All grid_positions must be finite."
            raise ValueError(msg)

    @property
    def lengthscale(self) -> torch.Tensor:
        """Return the positive, optionally trainable Gaussian length scale."""
        return self.log_lengthscale.exp()

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        grid_positions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate point observations at all query grid locations.

        Args:
            values: (B, N, C) or (B, T, N, C) observation values.
            positions: (B, N, 2) or (B, T, N, 2) point positions. These
                may differ between samples and between times.
            grid_positions: (H, W, 2) regular query grid; shared across B/T.
            valid_mask: Optional boolean (B, N) or (B, T, N) mask.
                Masked observations may contain NaN placeholders.

        Returns:
            (B, C + 1, H, W) or (B, T, C + 1, H, W), with density
            in channel 0 and normalized signals in channels 1 onward.

        Raises:
            ValueError: On incompatible shapes or non-finite unmasked data.
            TypeError: On non-floating data or a non-boolean mask.

        """
        self._validate_shapes(values, positions, grid_positions)
        if not all(
            torch.is_floating_point(x) for x in (values, positions, grid_positions)
        ):
            msg = "values, positions and grid_positions must be floating-point tensors."
            raise TypeError(msg)
        if values.device != positions.device:
            msg = "values and positions must be on the same device."
            raise ValueError(msg)
        if values.dtype != positions.dtype:
            msg = "values and positions must have the same dtype."
            raise ValueError(msg)

        has_time = values.ndim == _TIME_SERIES_RANK
        if not has_time:
            values = values.unsqueeze(1)
            positions = positions.unsqueeze(1)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(1)

        batch_size, n_times, n_points, n_channels = values.shape
        valid_mask = self._make_mask(values, valid_mask)
        self._validate_finite(values, positions, grid_positions, valid_mask)

        flat_batch = batch_size * n_times
        flat_mask = valid_mask.reshape(flat_batch, n_points)
        flat_values = torch.where(
            valid_mask[..., None], values, torch.zeros_like(values)
        ).reshape(flat_batch, n_points, n_channels)
        flat_positions = torch.where(
            valid_mask[..., None], positions, torch.zeros_like(positions)
        ).reshape(flat_batch, n_points, 2)

        height, width = grid_positions.shape[:2]
        grid = grid_positions.to(device=positions.device, dtype=positions.dtype)
        queries = grid.reshape(-1, 2)
        lengthscale_squared = self.lengthscale.square()

        chunks: list[torch.Tensor] = []
        for grid_chunk in queries.split(self.chunk_size):
            # (B*T, n_queries_in_chunk, N); never allocate H*W*N at once.
            distances = torch.cdist(
                grid_chunk.unsqueeze(0).expand(flat_batch, -1, -1),
                flat_positions,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            weights = torch.exp(-0.5 * distances.square() / lengthscale_squared)
            weights = torch.where(flat_mask[:, None, :], weights, 0.0)

            density = weights.sum(dim=-1, keepdim=True)
            signal = torch.bmm(weights, flat_values) / density.clamp_min(self.eps)
            signal = torch.where(density > self.eps, signal, 0.0)
            chunks.append(torch.cat((density, signal), dim=-1))

        output = torch.cat(chunks, dim=1)
        output = output.reshape(
            batch_size, n_times, height, width, n_channels + 1
        ).permute(0, 1, 4, 2, 3)
        return output if has_time else output[:, 0]


class SetConvCNN(nn.Module):
    """Embed irregular observations as gridded CNN features.

    The SetConv density channel indicates where observations support the
    interpolated values, allowing the CNN to distinguish data-sparse regions.
    This is an adapter for gridded models; it does not itself perform Argo
    ingestion or reproject geographic coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        lengthscale: float,
        hidden_channels: int = 32,
        learnable: bool = True,
        chunk_size: int = 2048,
    ) -> None:
        """Initialize a SetConv followed by two spatial convolutions."""
        super().__init__()
        if min(in_channels, out_channels, hidden_channels) <= 0:
            msg = "in_channels, out_channels and hidden_channels must be positive."
            raise ValueError(msg)
        self.in_channels = in_channels
        self.setconv = SetConv(lengthscale, learnable=learnable, chunk_size=chunk_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        grid_positions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (B, C_out, H, W) or (B, T, C_out, H, W) features."""
        if values.shape[-1] != self.in_channels:
            msg = (
                f"Expected {self.in_channels} observation channels, "
                f"got {values.shape[-1]}."
            )
            raise ValueError(msg)
        gridded = self.setconv(values, positions, grid_positions, valid_mask)
        if gridded.ndim == _TIME_SERIES_RANK:
            return self.cnn(gridded)
        batch_size, n_times, channels, height, width = gridded.shape
        encoded = self.cnn(gridded.reshape(-1, channels, height, width))
        return encoded.reshape(batch_size, n_times, -1, height, width)
