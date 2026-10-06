"""Lightweight multi-scale linear attention module for 2D inputs."""

from typing import Literal

import torch
from torch import Tensor, nn


class LiteMLA(nn.Module):
    """Multi-head, multi-scale, linear-complexity self-attention over NCHW tensors.

    Use a ReLU-kernel linear attention mechanism (Katharopoulos et al. 2020) as is done
    in the EfficientViT LiteMLA. This scales as O(N) in the spatial size (H*W) rather
    than O(N^2) in the standard softmax attention.

    Multi-scale aggregation is achieved through applying a depthwise spatial convolution
    to the q/k/v components for each kernel size in ``scales``. These are then passed
    through the ReLU linear attention separately before being concatenated.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        *,
        scales: tuple[int, ...] = (5,),
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        eps: float = 1e-4,
    ) -> None:
        """Initialise a LiteMLA.

        Args:
            channels: the number of input and output channels.
            heads: the number of attention heads.
            scales: the kernel sizes of the depthwise convs to apply to the q/k/v
                projection before attention. Each scale is applied independently, and
                the outputs are concatenated.
            padding_mode: the padding mode to use for the depthwise convs.
            eps: a small value to add to the denominator to avoid division by zero.

        """
        super().__init__()
        if channels % heads != 0:
            msg = f"channels ({channels}) must be divisible by heads ({heads})"
            raise ValueError(msg)
        self.heads = heads
        self.head_dim = channels // heads
        self.eps = eps
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    # Depthwise spatial convolution
                    nn.Conv2d(
                        channels * 3,
                        channels * 3,
                        scale,
                        padding=scale // 2,
                        padding_mode=padding_mode,
                        groups=channels * 3,
                        bias=False,
                    ),
                    # 1x1 channel projection to mix the heads
                    nn.Conv2d(
                        channels * 3, channels * 3, kernel_size=1, groups=3 * heads
                    ),
                )
                for scale in scales
            ]
        )
        self.proj = nn.Conv2d(channels * (1 + len(scales)), channels, kernel_size=1)

    def relu_linear_attention(self, qkv: Tensor) -> Tensor:
        """Apply ReLU linear attention to a single [N, 3*C, H, W] q/k/v projection."""
        n, _, h, w = qkv.shape
        # q, k, v shapes: [N, heads, head_dim, H*W]
        q, k, v = qkv.view(n, 3, self.heads, self.head_dim, h * w).unbind(dim=1)
        q = q.relu()
        k = k.relu()
        # Linear attention: aggregate K^T V and the K normaliser once and apply to q
        # kv shape: [N, heads, head_dim, head_dim]
        kv = torch.einsum("bhdn,bhen->bhde", k, v)
        # k_norm shape: [N, heads, head_dim]
        k_norm = k.sum(dim=-1)
        # numerator shape: [N, heads, head_dim, H*W]
        numerator = torch.einsum("bhdn,bhde->bhen", q, kv)
        # denominator shape: [N, heads, 1, H*W]
        denominator = torch.einsum("bhdn,bhd->bhn", q, k_norm).unsqueeze(2) + self.eps
        return (numerator / denominator).reshape(n, self.heads * self.head_dim, h, w)

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-scale ReLU linear attention to x."""
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv, *(scale_conv(qkv) for scale_conv in self.scale_convs)]
        out = torch.cat([self.relu_linear_attention(q) for q in multi_scale_qkv], dim=1)
        return self.proj(out)
