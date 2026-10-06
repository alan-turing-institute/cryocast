"""Perceiver-style cross-attention for sparse, moving Argo observations."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .argo_torch import (
    TorchSparseSequenceBatch,
    spherical_fourier_features,
)

_LATENT_GRID_NDIM = 2


@dataclass(frozen=True, slots=True)
class SparseCrossAttentionConfig:
    """Configuration for the experimental sparse cross-attention encoder."""

    variable_names: tuple[str, ...] = ("TEMP", "PSAL")
    latent_channels: int = 16
    embedding_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 64
    fourier_frequencies: int = 3
    query_chunk_size: int = 1024
    dropout: float = 0.0
    auxiliary_channels: int = 0
    use_pressure: bool = False
    use_time_offsets: bool = False
    pressure_scale: float = 50.0
    time_scale_hours: float = 24.0


class _CrossAttentionBlock(nn.Module):
    """One pre-norm cross-attention block with a feed-forward stage."""

    def __init__(self, config: SparseCrossAttentionConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.embedding_dim)
        self.token_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.post_attention_norm = nn.LayerNorm(config.embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.embedding_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        query_chunk_size: int,
    ) -> torch.Tensor:
        """Update query tokens while ignoring padded sparse observations."""
        nonempty = token_mask.any(dim=-1)
        if not bool(nonempty.any()):
            return torch.zeros_like(queries)

        updated = torch.zeros_like(queries)
        active_queries = queries[nonempty]
        active_tokens = self.token_norm(tokens[nonempty])
        key_padding_mask = ~token_mask[nonempty]
        chunks: list[torch.Tensor] = []
        for start in range(0, active_queries.shape[1], query_chunk_size):
            query_chunk = active_queries[:, start : start + query_chunk_size]
            attended, _ = self.attention(
                self.query_norm(query_chunk),
                active_tokens,
                active_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            state = query_chunk + self.dropout(attended)
            state = state + self.dropout(
                self.feedforward(self.post_attention_norm(state))
            )
            chunks.append(state)
        updated[nonempty] = torch.cat(chunks, dim=1)
        return updated


class SparseCrossAttentionEncoder(nn.Module):
    """Encode moving sparse Argo observations with Perceiver-style cross-attention.

    Observation tokens combine measurements, unit-sphere Fourier position features,
    and optional pressure, time, or numeric auxiliary channels. Geographic query
    tokens cross-attend to the variable-length observation set. The same query path
    is used for latent-grid encoding and the symmetric reconstruction benchmark.
    """

    latent_query_features: torch.Tensor

    def __init__(
        self,
        config: SparseCrossAttentionConfig,
        latent_latitudes: np.ndarray,
        latent_longitudes: np.ndarray,
    ) -> None:
        """Initialise attention layers and fixed latent-grid positional queries."""
        super().__init__()
        self._validate_config(config)
        latitudes = torch.as_tensor(latent_latitudes, dtype=torch.float32)
        longitudes = torch.as_tensor(latent_longitudes, dtype=torch.float32)
        if latitudes.ndim != _LATENT_GRID_NDIM or longitudes.shape != latitudes.shape:
            msg = "Latent latitude and longitude grids must be aligned 2D arrays."
            raise ValueError(msg)

        self.config = config
        self.latent_shape = (latitudes.shape[0], latitudes.shape[1])
        query_features = spherical_fourier_features(
            latitudes,
            longitudes,
            config.fourier_frequencies,
        ).reshape(-1, self.position_channels)
        self.register_buffer("latent_query_features", query_features)

        token_channels = len(config.variable_names) + self.position_channels
        token_channels += config.auxiliary_channels
        token_channels += int(config.use_pressure) + int(config.use_time_offsets)
        self.token_projection = nn.Linear(
            token_channels,
            config.embedding_dim,
            bias=False,
        )
        self.query_projection = nn.Linear(
            self.position_channels,
            config.embedding_dim,
            bias=False,
        )
        self.query_bias = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.blocks = nn.ModuleList(
            _CrossAttentionBlock(config) for _ in range(config.num_layers)
        )
        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.latent_channels,
            bias=False,
        )

    @staticmethod
    def _validate_config(config: SparseCrossAttentionConfig) -> None:
        if not config.variable_names:
            msg = "At least one input variable is required."
            raise ValueError(msg)
        if (
            min(
                config.latent_channels,
                config.embedding_dim,
                config.num_heads,
                config.num_layers,
                config.feedforward_dim,
                config.query_chunk_size,
            )
            <= 0
        ):
            msg = "Cross-attention dimensions and layer counts must be positive."
            raise ValueError(msg)
        if config.embedding_dim % config.num_heads != 0:
            msg = "embedding_dim must be divisible by num_heads."
            raise ValueError(msg)
        if config.fourier_frequencies < 0 or config.auxiliary_channels < 0:
            msg = "Fourier and auxiliary channel counts cannot be negative."
            raise ValueError(msg)
        if not 0.0 <= config.dropout < 1.0:
            msg = "dropout must be in [0, 1)."
            raise ValueError(msg)
        if config.pressure_scale <= 0 or config.time_scale_hours <= 0:
            msg = "Optional scalar feature scales must be positive."
            raise ValueError(msg)

    @property
    def position_channels(self) -> int:
        """Return the spherical/Fourier positional feature width."""
        return 3 * (1 + 2 * self.config.fourier_frequencies)

    @property
    def name(self) -> str:
        """Return a stable encoder name."""
        return "sparse-cross-attention"

    @property
    def latent_channels(self) -> int:
        """Return the output latent width used by the common reconstruction head."""
        return self.config.latent_channels

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return the ordered observed variables expected by the encoder."""
        return self.config.variable_names

    def forward(self, batch: TorchSparseSequenceBatch) -> torch.Tensor:
        """Return direct latent encoding with shape ``[B,T,C_latent,H,W]``."""
        latent = self._encode_query_features(batch, self.latent_query_features)
        batch_size, n_times = batch.latitudes.shape[:2]
        height, width = self.latent_shape
        return latent.reshape(
            batch_size,
            n_times,
            height,
            width,
            self.config.latent_channels,
        ).permute(0, 1, 4, 2, 3)

    def encode_queries(
        self,
        batch: TorchSparseSequenceBatch,
        query_latitudes: torch.Tensor,
        query_longitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode arbitrary geographic queries to ``[B,T,Q,C_latent]``."""
        if query_latitudes.ndim != 1 or query_longitudes.shape != query_latitudes.shape:
            msg = "Query latitude and longitude must be aligned 1D tensors."
            raise ValueError(msg)
        query_features = spherical_fourier_features(
            query_latitudes.to(
                device=batch.latitudes.device,
                dtype=batch.latitudes.dtype,
            ),
            query_longitudes.to(
                device=batch.longitudes.device,
                dtype=batch.longitudes.dtype,
            ),
            self.config.fourier_frequencies,
        )
        return self._encode_query_features(batch, query_features)

    def _encode_query_features(
        self,
        batch: TorchSparseSequenceBatch,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        tokens, token_mask = self._observation_tokens(batch)
        batch_size, n_times = batch.latitudes.shape[:2]
        queries = self._query_tokens(
            query_features,
            n_samples=batch_size * n_times,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        state = queries
        for block in self.blocks:
            state = block(
                state,
                tokens,
                token_mask,
                query_chunk_size=self.config.query_chunk_size,
            )
        latent = self.output_projection(state)
        return latent.reshape(
            batch_size,
            n_times,
            query_features.shape[0],
            self.config.latent_channels,
        )

    def _observation_tokens(
        self,
        batch: TorchSparseSequenceBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_batch(batch)
        batch_size, n_times, n_points = batch.latitudes.shape
        positional = spherical_fourier_features(
            batch.latitudes,
            batch.longitudes,
            self.config.fourier_frequencies,
        )
        feature_parts = [batch.measurements, positional]
        if self.config.use_pressure and batch.pressure is not None:
            feature_parts.append(
                (batch.pressure / self.config.pressure_scale).unsqueeze(-1)
            )
        if self.config.use_time_offsets and batch.time_offsets_hours is not None:
            feature_parts.append(
                (batch.time_offsets_hours / self.config.time_scale_hours).unsqueeze(-1)
            )
        if batch.auxiliary_measurements is not None:
            feature_parts.append(batch.auxiliary_measurements)
        features = torch.cat(feature_parts, dim=-1)
        features = torch.where(
            batch.mask.unsqueeze(-1),
            torch.nan_to_num(features),
            torch.zeros_like(features),
        )
        tokens = self.token_projection(features).reshape(
            batch_size * n_times,
            n_points,
            self.config.embedding_dim,
        )
        return tokens, batch.mask.reshape(batch_size * n_times, n_points)

    def _validate_batch(self, batch: TorchSparseSequenceBatch) -> None:
        if batch.variable_names != self.config.variable_names:
            msg = "Sparse batch variables must match the encoder configuration."
            raise ValueError(msg)
        actual_auxiliary = (
            0
            if batch.auxiliary_measurements is None
            else batch.auxiliary_measurements.shape[-1]
        )
        if actual_auxiliary != self.config.auxiliary_channels:
            msg = "Sparse batch auxiliary channels do not match the encoder config."
            raise ValueError(msg)
        self._validate_optional_features(batch)

    def _validate_optional_features(self, batch: TorchSparseSequenceBatch) -> None:
        valid = batch.mask
        if self.config.use_pressure:
            if batch.pressure is None:
                msg = "Encoder is configured to use pressure but none was provided."
                raise ValueError(msg)
            if bool(valid.any()) and not bool(
                torch.isfinite(batch.pressure[valid]).all()
            ):
                msg = "Configured pressure values must be finite at valid observations."
                raise ValueError(msg)
        if self.config.use_time_offsets:
            if batch.time_offsets_hours is None:
                msg = (
                    "Encoder is configured to use time offsets but none were provided."
                )
                raise ValueError(msg)
            if bool(valid.any()) and not bool(
                torch.isfinite(batch.time_offsets_hours[valid]).all()
            ):
                msg = "Configured time offsets must be finite at valid observations."
                raise ValueError(msg)
        if (
            batch.auxiliary_measurements is not None
            and bool(valid.any())
            and not bool(torch.isfinite(batch.auxiliary_measurements[valid]).all())
        ):
            msg = "Configured auxiliary values must be finite at valid observations."
            raise ValueError(msg)

    def _query_tokens(
        self,
        query_features: torch.Tensor,
        *,
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        projected = self.query_projection(query_features.to(device=device, dtype=dtype))
        projected = projected.unsqueeze(0).expand(n_samples, -1, -1)
        return projected + self.query_bias.to(device=device, dtype=dtype)
