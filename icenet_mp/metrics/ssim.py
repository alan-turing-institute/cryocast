import torch
import torch.nn.functional as F

from .base_daily_metric import BaseDailyMetric


class SSIMPerForecastDay(BaseDailyMetric):
    """Structural Similarity Index (SSIM) per forecast lead time.

    Follows the standard SSIM definition from Wang et al. (2004), "Image quality
    assessment: from error visibility to structural similarity", IEEE Transactions on
    Image Processing, vol. 13, no. 4, pp. 600-612. Adapted (channels-first, PyTorch)
    from the Gaussian-filtered dm_pix/cloudcasting implementation at:
    https://github.com/openclimatefix/cloudcasting/blob/main/src/cloudcasting/metrics.py

    Each field is locally compared to the other within a Gaussian-weighted
    `filter_size` x `filter_size` window around every cell, following the standard
    SSIM formulation. Note that the true SSIM is only defined on grayscale; this
    implementation does not perform any colourspace transform, so multi-channel
    inputs are averaged as if each channel were an independent greyscale image.
    """

    kernel: torch.Tensor

    def __init__(
        self,
        max_val: float = 1.0,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        land_mask: torch.Tensor | None = None,
    ) -> None:
        """Initialize the SSIM metric.

        Parameters
        ----------
        max_val: float, optional
            The maximum magnitude that predictions or targets can have (default is 1.0).
        filter_size: int, optional
            Size (in pixels) of the square Gaussian filtering window. Must be a
            positive odd integer (default is 11).
        filter_sigma: float, optional
            The bandwidth of the Gaussian used for filtering (> 0.) (default is 1.5).
        k1: float, optional
            One of the SSIM dampening parameters (> 0.) (default is 0.01).
        k2: float, optional
            One of the SSIM dampening parameters (> 0.) (default is 0.03).
        land_mask: torch.Tensor, optional
            Boolean tensor of shape (H, W), True for ocean cells and False for land.
            When given, land cells are excluded from the final averaged score. Note
            this does not prevent land values from influencing the SSIM of nearby
            ocean cells via the Gaussian filter window, which is applied before
            masking.

        """
        super().__init__(land_mask=land_mask)
        if filter_size < 1 or filter_size % 2 == 0:
            msg = "filter_size must be a positive odd integer."
            raise ValueError(msg)
        self.c1 = (k1 * max_val) ** 2
        self.c2 = (k2 * max_val) ** 2
        self.padding = filter_size // 2

        kernel_1d = self._gaussian_kernel(filter_size, filter_sigma)
        self.register_buffer("kernel", kernel_1d.outer(kernel_1d)[None, None])

    @staticmethod
    def _gaussian_kernel(filter_size: int, filter_sigma: float) -> torch.Tensor:
        """1D Gaussian filter of the given size and standard deviation, normalized to sum to 1."""
        coords = torch.arange(filter_size, dtype=torch.float32) - filter_size // 2
        filt = torch.exp(-0.5 * (coords / filter_sigma) ** 2)
        return filt / filt.sum()

    def _filter(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian filter to each channel of a (N, C, H, W) tensor independently."""
        n_channels = images.shape[1]
        kernel = self.kernel.to(images).expand(n_channels, 1, -1, -1).contiguous()
        return F.conv2d(images, kernel, padding=self.padding, groups=n_channels)

    def _compute_errors(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the per-pixel SSIM map for a (batch, time, channel, height, width) tensor."""
        batch_size, n_steps, n_channels, height, width = preds.shape
        preds_flat = preds.reshape(-1, n_channels, height, width)
        targets_flat = targets.reshape(-1, n_channels, height, width)

        mu_preds = self._filter(preds_flat)
        mu_targets = self._filter(targets_flat)
        mu_preds_sq = mu_preds * mu_preds
        mu_targets_sq = mu_targets * mu_targets
        mu_preds_targets = mu_preds * mu_targets

        sigma_preds_sq = self._filter(preds_flat**2) - mu_preds_sq
        sigma_targets_sq = self._filter(targets_flat**2) - mu_targets_sq
        sigma_preds_targets = self._filter(preds_flat * targets_flat) - mu_preds_targets

        # Clip the variances and covariances to valid values.
        # Variance must be non-negative:
        epsilon = torch.finfo(torch.float32).eps ** 2
        sigma_preds_sq = sigma_preds_sq.clamp(min=epsilon)
        sigma_targets_sq = sigma_targets_sq.clamp(min=epsilon)
        sigma_preds_targets = sigma_preds_targets.sign() * torch.minimum(
            torch.sqrt(sigma_preds_sq * sigma_targets_sq), sigma_preds_targets.abs()
        )

        numerator = (2 * mu_preds_targets + self.c1) * (
            2 * sigma_preds_targets + self.c2
        )
        denominator = (mu_preds_sq + mu_targets_sq + self.c1) * (
            sigma_preds_sq + sigma_targets_sq + self.c2
        )
        ssim_map = numerator / denominator
        return ssim_map.reshape(batch_size, n_steps, n_channels, height, width)
