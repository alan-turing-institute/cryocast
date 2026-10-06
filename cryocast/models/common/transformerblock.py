import torch
from torch import nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        """Initialize a single transformer encoder block with multi-head attention and MLP.

        Args:
            dim (int): Input and output dimension of the block
            heads (int): Number of attention heads
            mlp_dim (int): Hidden dimension of the MLP layer
            dropout (float): Dropout probability

        """
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer encoder block with residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, dim]

        Returns:
            torch.Tensor: Output tensor of shape [B, N, dim]

        """
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]

        return x + self.mlp(self.ln2(x))
