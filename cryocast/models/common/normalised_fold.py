import torch
from torch import nn


class NormalisedFold(nn.Module):
    """Fold patches into an image while accounting for per-pixel overlaps.

    Optionally applies a 2D Hann window (https://en.wikipedia.org/wiki/Hann_function) to
    each patch before folding so that pixels near patch centres contribute more than
    pixels near patch edges, reducing seam artifacts.
    """

    window: torch.Tensor
    overlap_mask: torch.Tensor

    def __init__(
        self,
        *,
        output_size: tuple[int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        use_hann_window: bool = False,
    ) -> None:
        """Initialise a NormalisedFold."""
        super().__init__()
        self.fold = nn.Fold(
            output_size=output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Compute an overlap mask by folding and unfolding a tensor of ones shaped like
        # the expected output. We disable gradients and mark the mask as a buffer to
        # ensure it is not treated as a learnable parameter.
        with torch.no_grad():
            k_h, k_w = kernel_size
            if use_hann_window:
                # 2D Hann window: outer product of two 1D windows
                window = (
                    torch.hann_window(k_h, periodic=False).unsqueeze(1)
                    * torch.hann_window(k_w, periodic=False).unsqueeze(0)
                ).reshape(1, -1, 1)
            else:
                # Flat window: all ones
                window = torch.ones(1, k_h * k_w, 1)
            self.register_buffer("window", window, persistent=False)

            # Compute per-pixel weight sum (denominator) by folding the window itself.
            unfold = nn.Unfold(
                kernel_size=self.fold.kernel_size,
                stride=self.fold.stride,
                padding=self.fold.padding,
            )
            n_patches = unfold(torch.ones(1, 1, *output_size)).shape[-1]
            weights = window.expand(1, k_h * k_w, n_patches)
            self.register_buffer("overlap_mask", self.fold(weights), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply window weights, fold, then normalize by overlap mask."""
        # Input has shape: [N, C*k_h*k_w, n_patches]
        # We therefore tile the window C times, then broadcast across N and n_patches.
        x = x * self.window.repeat(1, x.shape[1] // self.window.shape[1], 1)
        return self.fold(x) / self.overlap_mask
