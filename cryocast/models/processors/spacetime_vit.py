"""Factorised space-time Vision Transformer processor."""

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from cryocast.models.common import PatchEmbedding, TransformerEncoderBlock
from cryocast.types import ProcessorOutput, TensorNTCHW

from .base_processor import BaseProcessor

NTCHW_DIMENSIONS = 5


def _sinusoidal_time_embedding(
    steps: torch.Tensor,
    dim: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a deterministic embedding for relative integer time steps."""
    half_dim = max(dim // 2, 1)
    exponents = torch.arange(
        half_dim,
        device=steps.device,
        dtype=torch.float32,
    )
    if half_dim > 1:
        exponents = exponents / (half_dim - 1)

    frequencies = torch.exp(-math.log(10000.0) * exponents)
    phases = steps.to(dtype=torch.float32).unsqueeze(-1) * frequencies
    embedding = torch.cat((phases.sin(), phases.cos()), dim=-1)
    if embedding.shape[-1] < dim:
        embedding = functional.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding[..., :dim].to(dtype=dtype)


class SpaceTimeVitProcessor(BaseProcessor):
    """Predict all forecast steps jointly with factorised spatial and temporal attention.

    History frames are patch-embedded independently, spatial attention is applied within
    each frame, and a temporal decoder then attends across the history for every spatial
    patch. Forecast queries use deterministic relative lead-time embeddings and a single
    shared learned token, so parameter shapes do not depend on history or forecast
    length. Causal self-attention between forecast queries makes each forecast prefix
    independent of later configured lead times while retaining parallel one-shot
    decoding.

    The processor predicts residuals only for the target latent slice. Non-target
    conditioning channels, including Argo, are carried forward from the latest observed
    frame instead of being treated as prognostic variables.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dropout: float = 0.1,
        emb_dim: int = 256,
        forecast_spatial_depth: int = 2,
        heads: int = 8,
        mlp_dim: int = 1024,
        patch_size: int = 24,
        spatial_depth: int = 2,
        temporal_depth: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialise the factorised space-time transformer."""
        super().__init__(**kwargs)
        self._validate_attention_configuration(
            emb_dim=emb_dim,
            forecast_spatial_depth=forecast_spatial_depth,
            heads=heads,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
        )
        self._configure_data_space(patch_size)

        self.emb_dim = emb_dim

        self.patch_embed = PatchEmbedding(
            self.c_combined,
            patch_size,
            emb_dim,
            self.img_size,
        )
        self.spatial_pos_embed = nn.Parameter(torch.empty(1, self.num_patches, emb_dim))
        self.forecast_token = nn.Parameter(torch.empty(1, 1, 1, emb_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.forecast_token, std=0.02)

        self.input_dropout = nn.Dropout(dropout)
        self.history_spatial = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_dim, heads, mlp_dim, dropout)
                for _ in range(spatial_depth)
            ]
        )
        self.history_norm = nn.LayerNorm(emb_dim)

        self.temporal_decoder = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=emb_dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(temporal_depth)
            ]
        )
        self.temporal_norm = nn.LayerNorm(emb_dim)

        self.forecast_spatial = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_dim, heads, mlp_dim, dropout)
                for _ in range(forecast_spatial_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(emb_dim)

        self.delta_head = nn.Linear(
            emb_dim,
            patch_size * patch_size * self.c_target,
        )
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_head.bias)

        # Preserve the residual head at initialisation while retaining a trainable
        # local smoothing operation for patch boundaries.
        self.delta_smoother = nn.Conv2d(
            self.c_target,
            self.c_target,
            kernel_size=3,
            padding=1,
        )
        nn.init.dirac_(self.delta_smoother.weight)
        if self.delta_smoother.bias is not None:
            nn.init.zeros_(self.delta_smoother.bias)

    def _validate_attention_configuration(
        self,
        *,
        emb_dim: int,
        forecast_spatial_depth: int,
        heads: int,
        spatial_depth: int,
        temporal_depth: int,
    ) -> None:
        """Validate transformer dimensions and configured time horizons."""
        if heads <= 0 or emb_dim % heads:
            msg = f"emb_dim {emb_dim} must be divisible by heads {heads}."
            raise ValueError(msg)
        if spatial_depth < 0 or forecast_spatial_depth < 0:
            msg = "Spatial depths must be non-negative."
            raise ValueError(msg)
        if temporal_depth < 1:
            msg = "temporal_depth must be at least 1."
            raise ValueError(msg)
        if self.n_history_steps < 1 or self.n_forecast_steps < 1:
            msg = "History and forecast steps must both be positive."
            raise ValueError(msg)

    def _configure_data_space(self, patch_size: int) -> None:
        """Validate and cache latent-space dimensions and the target slice."""
        height, width = self.data_space.shape
        if height != width:
            msg = "The height and width of the input must be equal."
            raise ValueError(msg)
        if patch_size <= 0:
            msg = "patch_size must be greater than 0."
            raise ValueError(msg)
        if height % patch_size:
            msg = f"img_size {height} must be divisible by patch_size {patch_size}."
            raise ValueError(msg)

        c_combined = self.data_space.channels
        c_target = self.data_space_target.channels
        target_channel_offset = self.target_channel_offset
        if (
            target_channel_offset is None
            or target_channel_offset < 0
            or target_channel_offset + c_target > c_combined
        ):
            msg = (
                f"target_channel_offset={target_channel_offset} with target channels="
                f"{c_target} does not fit inside combined latent channels={c_combined}."
            )
            raise ValueError(msg)

        self.target_slice_start = target_channel_offset
        self.target_slice_end = target_channel_offset + c_target
        self.c_target = c_target
        self.c_combined = c_combined
        self.img_size = height
        self.patch_size = patch_size
        self.num_patches = (height // patch_size) ** 2

    def _validate_input(self, x: TensorNTCHW) -> None:
        """Check that input dimensions match the configured latent and time spaces."""
        if x.ndim != NTCHW_DIMENSIONS:
            msg = (
                f"Expected a {NTCHW_DIMENSIONS}D NTCHW tensor, "
                f"got shape {tuple(x.shape)}."
            )
            raise ValueError(msg)

        _, times, channels, height, width = x.shape
        expected_tchw = (
            self.n_history_steps,
            self.c_combined,
            self.img_size,
            self.img_size,
        )
        if (times, channels, height, width) != expected_tchw:
            msg = (
                f"Expected input TCHW {expected_tchw}, "
                f"got {(times, channels, height, width)}."
            )
            raise ValueError(msg)

    def _encode_history(self, x: TensorNTCHW) -> torch.Tensor:
        """Patch-embed history and apply per-frame spatial attention."""
        batch, times, channels, height, width = x.shape
        tokens = self.patch_embed(x.reshape(batch * times, channels, height, width))
        tokens = tokens.reshape(batch, times, self.num_patches, self.emb_dim)

        history_steps = torch.arange(
            1 - times,
            1,
            device=x.device,
        )
        history_time = _sinusoidal_time_embedding(
            history_steps,
            self.emb_dim,
            dtype=tokens.dtype,
        )

        tokens = (
            tokens
            + self.spatial_pos_embed[:, None, :, :]
            + history_time[None, :, None, :]
        )
        tokens = self.input_dropout(tokens)
        tokens = self.history_spatial(
            tokens.reshape(batch * times, self.num_patches, self.emb_dim)
        )
        tokens = self.history_norm(tokens)
        return tokens.reshape(batch, times, self.num_patches, self.emb_dim)

    def _decode_forecast_tokens(self, history: torch.Tensor) -> torch.Tensor:
        """Decode every forecast lead jointly from the encoded history."""
        batch, times, _, _ = history.shape
        memory = history.permute(0, 2, 1, 3).reshape(
            batch * self.num_patches,
            times,
            self.emb_dim,
        )

        forecast_steps = torch.arange(
            1,
            self.n_forecast_steps + 1,
            device=history.device,
        )
        forecast_time = _sinusoidal_time_embedding(
            forecast_steps,
            self.emb_dim,
            dtype=history.dtype,
        )

        queries = history[:, -1:, :, :].expand(
            -1,
            self.n_forecast_steps,
            -1,
            -1,
        )
        queries = queries + self.forecast_token + forecast_time[None, :, None, :]
        queries = queries.permute(0, 2, 1, 3).reshape(
            batch * self.num_patches,
            self.n_forecast_steps,
            self.emb_dim,
        )

        forecast_mask = torch.ones(
            (self.n_forecast_steps, self.n_forecast_steps),
            device=history.device,
            dtype=torch.bool,
        ).triu(diagonal=1)
        for decoder_layer in self.temporal_decoder:
            queries = decoder_layer(queries, memory, tgt_mask=forecast_mask)

        queries = self.temporal_norm(queries)
        queries = queries.reshape(
            batch,
            self.num_patches,
            self.n_forecast_steps,
            self.emb_dim,
        ).permute(0, 2, 1, 3)

        queries = self.forecast_spatial(
            queries.reshape(
                batch * self.n_forecast_steps,
                self.num_patches,
                self.emb_dim,
            )
        )
        return self.output_norm(queries)

    def _decode_target_delta(self, tokens: torch.Tensor) -> TensorNTCHW:
        """Decode forecast patch tokens into target-latent residual fields."""
        batch = tokens.shape[0] // self.n_forecast_steps
        patches_per_side = self.img_size // self.patch_size
        delta_patches = self.delta_head(tokens).reshape(
            batch,
            self.n_forecast_steps,
            patches_per_side,
            patches_per_side,
            self.c_target,
            self.patch_size,
            self.patch_size,
        )
        delta = delta_patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch,
            self.n_forecast_steps,
            self.c_target,
            self.img_size,
            self.img_size,
        )
        return self.delta_smoother(
            delta.reshape(
                batch * self.n_forecast_steps,
                self.c_target,
                self.img_size,
                self.img_size,
            )
        ).reshape_as(delta)

    def _build_combined_latent(
        self,
        x: TensorNTCHW,
        target_delta: TensorNTCHW,
    ) -> TensorNTCHW:
        """Insert target residual forecasts into persistent non-target latents."""
        last_frame = x[:, -1]
        target_last = last_frame[
            :,
            self.target_slice_start : self.target_slice_end,
        ]
        target_forecast = target_last.unsqueeze(1) + target_delta
        persistent = last_frame.unsqueeze(1).expand(
            -1,
            self.n_forecast_steps,
            -1,
            -1,
            -1,
        )
        return torch.cat(
            (
                persistent[:, :, : self.target_slice_start],
                target_forecast,
                persistent[:, :, self.target_slice_end :],
            ),
            dim=2,
        )

    def rollout(
        self,
        x: TensorNTCHW,
        y: TensorNTCHW | None = None,
    ) -> ProcessorOutput:
        """Predict the full forecast horizon in one processor call."""
        del y
        self._validate_input(x)
        history = self._encode_history(x)
        forecast_tokens = self._decode_forecast_tokens(history)
        target_delta = self._decode_target_delta(forecast_tokens)
        prediction = self._build_combined_latent(x, target_delta)
        return ProcessorOutput(prediction=prediction)
