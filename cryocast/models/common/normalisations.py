from torch import Tensor, nn


class ChannelNorm2D(nn.Module):
    """Normalise over the channel dimension of an NCHW or NTCHW tensor."""

    def __init__(self, channels: int) -> None:
        """Initialise a ChannelNorm2D."""
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

    def forward(self, x: Tensor) -> Tensor:
        c_dim = x.ndim - 3  # handle both NCHW and NTCHW cases
        return self.norm(x.movedim(c_dim, -1)).movedim(-1, c_dim)


def normalisation_from_name(norm_type: str, out_channels: int) -> nn.Module:
    """Create a normalization layer based on the specified type."""
    # Batch normalisation
    if norm_type.lower() == "batchnorm":
        return nn.BatchNorm2d(out_channels)

    # Channel normalisation
    if norm_type.lower() == "channelnorm":
        return ChannelNorm2D(out_channels)

    # Group normalisation
    if norm_type.lower() == "groupnorm":
        # Determine the highest integer less than or equal to 8 that divides `channels`
        num_groups = max(num for num in range(1, 9) if out_channels % num == 0)
        return nn.GroupNorm(num_groups, out_channels)

    # No normalisation
    if norm_type.lower() == "none":
        return nn.Identity()

    msg = f"Unknown norm_type: {norm_type}. Choose 'groupnorm', 'batchnorm', or 'none'"
    raise ValueError(msg)
