import torch
from torch import Tensor, nn


class GatedAttention(nn.Module):
    """SimVPv2 gated spatial-temporal attention (gSTA).

    - a depthwise conv (kernel 2*dilation-1)
    - a depthwise dilated conv (kernel ~= kernel_size // dilation)
    - a gated 1x1 projection.
    """

    def __init__(self, channels: int, *, kernel_size: int, dilation: int) -> None:
        """Initialise a GatedAttention module."""
        super().__init__()
        if dilation < 1:
            msg = f"dilation({dilation}) must be at least 1."
            raise ValueError(msg)
        if kernel_size < dilation:
            msg = f"kernel_size ({kernel_size}) must be >= dilation ({dilation})."
            raise ValueError(msg)

        depthwise_kernel = 2 * dilation - 1
        dilated_kernel = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)

        self.attend = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                depthwise_kernel,
                groups=channels,
                padding=(depthwise_kernel - 1) // 2,
            ),
            nn.Conv2d(
                channels,
                channels,
                dilated_kernel,
                dilation=dilation,
                groups=channels,
                padding=dilation * (dilated_kernel - 1) // 2,
            ),
        )
        self.expand = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.expand(self.attend(x)).chunk(2, dim=1)
        return x + self.proj_out(value * gate.sigmoid())


class GatedAttentionBlock(nn.Module):
    """SimVPv2 gated spatial-temporal attention (gSTA) block.

    - subblock 1
        - normalise the input
        - apply gated attention to select spatial features
        - apply learnable per-channel scale
        - randomly drop a subset of samples for regularisation
        - add a skip connection to the input
    - subblock 2
        - normalise the input
        - apply a conv-MLP to mix channels
        - apply learnable per-channel scale
        - randomly drop a subset of samples for regularisation
        - add a skip connection to the input
    - add a trailing 1x1 projection when in_channels != out_channels
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        mlp_ratio: float,
        drop_path_prob: float,
        mlp_drop_prob: float,
    ) -> None:
        """Initialise a GatedAttentionBlock."""
        super().__init__()
        if drop_path_prob < 0 or drop_path_prob > 1:
            msg = f"drop_path_prob({drop_path_prob}) must be between 0 and 1."
            raise ValueError(msg)
        self.drop_path_prob = drop_path_prob
        if mlp_drop_prob < 0 or mlp_drop_prob > 1:
            msg = f"mlp_drop_prob({mlp_drop_prob}) must be between 0 and 1."
            raise ValueError(msg)

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.attn = GatedAttention(
            in_channels, kernel_size=kernel_size, dilation=dilation
        )
        self.layer_scale_1 = nn.Parameter(torch.full((in_channels,), 1e-2))

        self.norm2 = nn.BatchNorm2d(in_channels)
        hid_channels = int(in_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=1),
            nn.Conv2d(
                hid_channels,
                hid_channels,
                kernel_size=3,
                padding=1,
                groups=hid_channels,
            ),
            nn.GELU(),
            nn.Dropout(mlp_drop_prob),
            nn.Conv2d(hid_channels, in_channels, kernel_size=1),
            nn.Dropout(mlp_drop_prob),
        )
        self.layer_scale_2 = nn.Parameter(torch.full((in_channels,), 1e-2))

        self.projection = (
            None
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def _drop_path(self, x: Tensor) -> Tensor:
        """Stochastic depth: zero this residual branch for a random subset of samples."""
        if self.drop_path_prob == 0.0 or not self.training:
            return x
        if self.drop_path_prob == 1.0:
            return torch.zeros_like(x)
        keep_prob = 1 - self.drop_path_prob
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def _residual(
        self, x: Tensor, norm: nn.Module, block: nn.Module, layer_scale: Tensor
    ) -> Tensor:
        scale = layer_scale[:, None, None]
        return x + self._drop_path(scale * block(norm(x)))

    def forward(self, x: Tensor) -> Tensor:
        x = self._residual(x, self.norm1, self.attn, self.layer_scale_1)
        x = self._residual(x, self.norm2, self.mlp, self.layer_scale_2)
        return x if self.projection is None else self.projection(x)
