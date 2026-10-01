"""Adjusted-MSE (AMSE) spectral anti-blur loss on a flat grid.

Adaptation of the spherical-harmonic AMSE loss of Subich et al. (2025),
"Fixing the Double Penalty in Data-Driven Weather Forecasting Through a
Modified Spherical Harmonic Loss Function" (arXiv:2501.19374), to the flat
equal-area EASE2 grids used by icenet-mp. Independent reimplementation of the
published formula; reference implementation (spherical, JAX) at
https://github.com/csubich/graphcast (branch ``amse``).

Pointwise squared-error losses reward damping the amplitude of scales the
model cannot predict exactly (the "double penalty"): per scale group, the
MSE-optimal amplitude ratio equals the coherence with the target, so partially
predictable fine scales are systematically blurred. AMSE removes this
incentive. Writing per scale group k the powers P_k(x), P_k(y) and coherence
Coh_k(x, y) of prediction x and target y,

    MSE(x, y)  = sum_k [ (sqrt(P_k(x)) - sqrt(P_k(y)))^2
                         + 2 * sqrt(P_k(x) * P_k(y)) * (1 - Coh_k(x, y)) ]
    AMSE(x, y) = sum_k [ (sqrt(P_k(x)) - sqrt(P_k(y)))^2
                         + 2 * max(P_k(x), P_k(y)) * (1 - Coh_k(x, y)) ]

so under AMSE the decorrelation penalty can no longer be reduced by shrinking
the predicted amplitude, and the amplitude optimum at any fixed coherence is
P_k(x) = P_k(y): the target's power spectrum is preserved scale by scale.

On the flat grid the orthonormal decomposition is the 2D DFT and the scale
groups are annular bins of the isotropic wavenumber |k| = sqrt(kx^2 + ky^2);
Parseval's identity makes the group decomposition of MSE exact, which is the
only property the construction needs (Subich et al., Sec. 2.2 and 4.2).

Scope: designed and tested for spatial-field predictions (single-stage
encode-process-decode). Using this loss with models whose training loss lives
in a different space (e.g. DDPM v-space) is untested.
"""

import logging
import math
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)

AMSEMode = Literal["hybrid", "pure"]

# Type alias for the cached per-(H, W, device) binning tensors:
# (flat mode->bin index, flat mode weights, indices of non-empty bins,
#  per-bin wavenumber weight gamma_k aligned with those indices)
BinCache = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# Fields must carry at least (batch-like, H, W) dimensions.
MIN_FIELD_NDIM = 3

# Accepted values of ``wavenumber_weight`` (None is normalised to "none").
WAVENUMBER_WEIGHTS = ("none", "fastnet")

# FastNet (arXiv:2509.17601) upweights annular bin k by
# gamma_k = max(N_k * k**sqrt(3), 1), N_k = number of spectral modes in bin k.
FASTNET_EXPONENT = math.sqrt(3.0)


class AMSELoss(nn.Module):
    """Adjusted-MSE (anti-double-penalty) spectral loss on a flat grid.

    Two modes:

    - ``"hybrid"`` (default): the repo's standard Huber loss plus
      ``spectral_weight`` times the AMSE *excess*
      ``sum_k 2 * (max(P_k(x), P_k(y)) - sqrt(P_k(x) * P_k(y))) * (1 - Coh_k)``,
      which is >= 0, zero iff per-bin spectra match or coherence is perfect,
      and reduces to the unmodified Huber control at ``spectral_weight=0``.
    - ``"pure"``: the AMSE formula itself (per-bin amplitude + decorrelation
      terms plus a direct DC term), averaged over fields.

    Both modes sum the annular bins UNWEIGHTED by default, as in the Subich
    reference. ``wavenumber_weight="fastnet"`` instead upweights each bin by
    ``gamma_k = max(N_k * k**sqrt(3), 1)`` (FastNet, arXiv:2509.17601), which
    moves the penalty from the large scales that dominate a steep sea-ice
    spectrum onto the fine scales where the ice edge lives; the total is held
    fixed so ``spectral_weight`` stays comparable across the two settings (see
    ``_weight_bins``). Default OFF: unset, the loss is bit-for-bit unchanged.

    Numerical guards follow the reference implementation: ``eps`` inside the
    geometric-mean square root (the coherence denominator is singular for a
    zero field), coherence clamped at 1 from above (Cauchy-Schwarz can be
    violated by the eps), the mean/DC component handled as a direct squared
    difference (catastrophic cancellation), and all spectral arithmetic in
    float32.
    """

    def __init__(  # noqa: PLR0913 - flat config-driven knobs, all defaulted
        self,
        *,
        mode: AMSEMode = "hybrid",
        spectral_weight: float = 0.1,
        delta: float = 0.5,
        merge_bins_below: int = 4,
        eps: float = 1e-12,
        static_ref_path: str | None = None,
        wavenumber_weight: str | None = None,
    ) -> None:
        """Initialise the AMSE loss.

        Args:
            mode: "hybrid" (Huber + spectral_weight * AMSE excess) or "pure"
                (the AMSE formula alone).
            spectral_weight: weight of the spectral excess term in hybrid mode.
            delta: Huber transition point for the hybrid base term (matches
                the repo's default Huber loss).
            merge_bins_below: annuli with rounded |k| <= this value share one
                bin (the lowest annuli contain too few modes for a stable
                per-sample coherence estimate).
            eps: additive guard inside sqrt(P_k(x) * P_k(y)); keeps gradients
                finite when either field has an empty scale bin.
            static_ref_path: optional .npy path of a static reference field
                subtracted from both fields before the spectral terms.
            wavenumber_weight: per-bin upweighting of the spectral terms.
                None / "none" (default) leaves every annulus at weight 1, i.e.
                the unweighted Subich sum; "fastnet" applies
                gamma_k = max(N_k * k**sqrt(3), 1) (see ``_weight_bins``).

        """
        super().__init__()
        if mode not in ("hybrid", "pure"):
            msg = f"Unknown AMSELoss mode: {mode!r} (expected 'hybrid' or 'pure')"
            raise ValueError(msg)
        if merge_bins_below < 1:
            msg = f"merge_bins_below must be >= 1, got {merge_bins_below}"
            raise ValueError(msg)
        weight_mode = (wavenumber_weight or "none").lower()
        if weight_mode not in WAVENUMBER_WEIGHTS:
            msg = (
                f"Unknown AMSELoss wavenumber_weight: {wavenumber_weight!r} "
                f"(expected None or one of {WAVENUMBER_WEIGHTS})"
            )
            raise ValueError(msg)
        self.mode = mode
        self.spectral_weight = spectral_weight
        self.delta = delta
        self.merge_bins_below = merge_bins_below
        self.eps = eps
        self.wavenumber_weight = weight_mode
        # Optional anomaly transform: a static reference field (e.g. the
        # training-period mean SIC) subtracted from BOTH prediction and target
        # before the spectral terms. Removes the shared static structure
        # (coastline/climatological ring) that otherwise dominates band power
        # and inflates coherence, diluting the anti-blur excess (measured
        # 2026-07-27: static share 63-86% of band power on full_south).
        self.static_ref_path = static_ref_path
        self._static_ref: torch.Tensor | None = None
        # Lazily-built cache of binning tensors, keyed by (H, W, device).
        # A plain dict (not buffers) so it never enters the state dict.
        self._bin_cache: dict[tuple[int, int, str], BinCache] = {}

    def _binning(self, height: int, width: int, device: torch.device) -> BinCache:
        """Return (bin index, mode weight, non-empty bin ids, gamma_k) for H x W."""
        key = (height, width, str(device))
        cached = self._bin_cache.get(key)
        if cached is not None:
            return cached
        ky = torch.fft.fftfreq(height, device=device) * height
        kx = torch.fft.rfftfreq(width, device=device) * width
        kmag = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        bins = torch.round(kmag).long()
        # Merge the sparse low-|k| annuli (excluding DC) into one bin.
        bins = torch.where(
            (bins >= 1) & (bins < self.merge_bins_below), self.merge_bins_below, bins
        )
        # rfft halves the x axis: interior columns stand for +-kx pairs and
        # count twice; column 0 and (for even W) the Nyquist column appear once.
        weights = torch.full(kmag.shape, 2.0, device=device)
        weights[:, 0] = 1.0
        if width % 2 == 0:
            weights[:, -1] = 1.0
        # The DC mode is excluded from the annuli (handled as a direct term).
        weights[0, 0] = 0.0
        bins_flat = bins.reshape(-1)
        weights_flat = weights.reshape(-1)
        counts = torch.zeros(int(bins_flat.max()) + 1, device=device)
        counts.scatter_add_(0, bins_flat, (weights_flat > 0).float())
        present = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
        # FastNet wavenumber weight gamma_k = max(N_k * k**sqrt(3), 1), cached
        # alongside the bin tensors and used only when wavenumber_weight is on.
        # N_k is the number of modes of the FULL 2D DFT falling in bin k, so it
        # is the sum of the SAME rfft double-count weights used for the powers
        # (no second binning): interior columns count 2, column 0 and the even-W
        # Nyquist column count 1, and DC carries weight 0. k is the bin's
        # representative isotropic wavenumber, i.e. the bin index, which for the
        # merged low bin is merge_bins_below.
        mode_counts = torch.zeros_like(counts)
        mode_counts.scatter_add_(0, bins_flat, weights_flat)
        wavenumber = torch.arange(counts.numel(), device=device, dtype=counts.dtype)
        gamma = torch.clamp(mode_counts * wavenumber**FASTNET_EXPONENT, min=1.0)
        gamma = gamma.index_select(0, present)
        if self.wavenumber_weight != "none":
            # Once per grid, and only when the option is on: a run-log
            # fingerprint proving the flag actually reached the loss.
            logger.info(
                "AMSE wavenumber weighting %r engaged on %dx%d: %d active bins, "
                "gamma in [%.4g, %.4g]",
                self.wavenumber_weight,
                height,
                width,
                int(present.numel()),
                float(gamma.min()),
                float(gamma.max()),
            )
        result = (bins_flat, weights_flat, present, gamma)
        self._bin_cache[key] = result
        return result

    def _weight_bins(
        self, contributions: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Apply the FastNet per-bin upweighting to [n_fields, n_bins] terms.

        With ``wavenumber_weight="none"`` the input is returned untouched, so
        the default path is bit-for-bit the unweighted Subich sum.

        With ``"fastnet"`` each bin is multiplied by
        ``gamma_k = max(N_k * k**sqrt(3), 1)`` and the result is divided by a
        DETACHED scalar chosen so the batch total is unchanged:

            sum_(i,k) gamma_k * c_ik / Z == sum_(i,k) c_ik,
            Z = sum_(i,k) gamma_k * c_ik / sum_(i,k) c_ik.

        Rationale: raw gamma_k grows like k**2.7 (N_k ~ 2*pi*k), spanning ~7
        orders of magnitude across the bins of a 432x432 grid, so the raw
        weighted sum would change the SCALE of the spectral term by orders of
        magnitude. That silently rescales the effective learning rate of the
        spectral branch and makes ``spectral_weight`` mean something different
        from its value in the unweighted arm - which would confound the
        factorial comparison the flag exists to serve. Normalising gamma to
        mean 1 over the active bins does NOT fix this: gamma is strongly
        anti-correlated with where the penalty sits (the penalty is concentrated
        at low |k| where gamma is smallest), so the total would still move by a
        large factor. Preserving the total makes the change purely a
        REDISTRIBUTION of the penalty across scales - exactly the intended
        intervention - and keeps the arm trainable at the unweighted arm's lr.
        Z is detached, so gradients are those of a fixed per-bin reweighting
        (a constant rescale of an ordinary weighted sum), not of a ratio.

        The DC term is deliberately NOT weighted. It is not an annulus (the DC
        mode carries binning weight 0 and never reaches a bin), and the FastNet
        formula itself assigns it gamma_0 = max(0 * 0**sqrt(3), 1) = 1; it is
        the domain-mean bias, which must keep full weight.
        """
        if self.wavenumber_weight == "none":
            return contributions
        gamma = self._binning(height, width, contributions.device)[3]
        weighted = contributions * gamma
        # Both sums are non-negative (every per-bin term is clamped at 0), so
        # the ratio is a weighted mean of gamma >= 1: bounded and well posed.
        # The eps pair makes Z = 1 for the exact-match case (0 / 0).
        scale = (weighted.detach().sum() + self.eps) / (
            contributions.detach().sum() + self.eps
        )
        return weighted / scale

    def binned_spectra(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-field, per-bin spectral statistics of the mean-removed fields.

        Returns:
            Tuple ``(P, T, C, dc)`` where ``P[i, b]`` / ``T[i, b]`` are the
            prediction/target powers of field ``i`` in |k|-bin ``b``,
            ``C[i, b]`` is the real cross-spectrum, and ``dc[i]`` is the
            squared difference of the field means. Powers are normalised so
            that ``sum_b (P + T - 2C) + dc`` equals the per-field MSE
            (Parseval's identity).

        """
        if prediction.shape != target.shape:
            msg = (
                f"prediction shape {tuple(prediction.shape)} "
                f"!= target shape {tuple(target.shape)}"
            )
            raise ValueError(msg)
        if prediction.ndim < MIN_FIELD_NDIM:
            msg = (
                f"expected at least {MIN_FIELD_NDIM} dimensions, got {prediction.ndim}"
            )
            raise ValueError(msg)
        height, width = prediction.shape[-2], prediction.shape[-1]
        fields_p = prediction.reshape(-1, height, width).float()
        fields_t = target.reshape(-1, height, width).float()
        if self.static_ref_path is not None:
            if self._static_ref is None:
                ref = torch.from_numpy(np.load(self.static_ref_path)).float()
                if not torch.isfinite(ref).all():
                    # A silently all-NaN reference field reached a production GPU job
                    # on 2026-07-27 and NaN'd every loss while the model itself was
                    # healthy. Fail loudly at load time instead.
                    n_bad = int((~torch.isfinite(ref)).sum())
                    msg = (
                        f"static_ref at {self.static_ref_path} has {n_bad} non-finite "
                        f"cells of {ref.numel()}; refusing to train on it."
                    )
                    raise ValueError(msg)
                if ref.shape != (height, width):
                    msg = (
                        f"static_ref shape {tuple(ref.shape)} does not match "
                        f"field shape ({height}, {width})"
                    )
                    raise ValueError(msg)
                self._static_ref = ref
            ref = self._static_ref.to(device=fields_p.device, dtype=fields_p.dtype)
            fields_p = fields_p - ref
            fields_t = fields_t - ref
        mean_p = fields_p.mean(dim=(-2, -1))
        mean_t = fields_t.mean(dim=(-2, -1))
        dc = (mean_p - mean_t) ** 2
        spec_p = torch.fft.rfft2(fields_p - mean_p[:, None, None], norm="ortho")
        spec_t = torch.fft.rfft2(fields_t - mean_t[:, None, None], norm="ortho")
        bins_flat, weights_flat, present, _ = self._binning(
            height, width, fields_p.device
        )
        n_fields = spec_p.shape[0]
        n_bins_total = int(bins_flat.max()) + 1
        scale = 1.0 / (height * width)
        index = bins_flat.unsqueeze(0).expand(n_fields, -1)

        def accumulate(values: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(n_fields, n_bins_total, device=values.device)
            out.scatter_add_(
                1, index, (values * weights_flat).reshape(n_fields, -1) * scale
            )
            return out.index_select(1, present)

        # Avoid using .abs() and .conj() which are causing crashes on the DGX Spark
        # ("invalid value for --gpu-architecture") due to NVRTC JIT issues.
        power_p = accumulate((spec_p.real**2 + spec_p.imag**2).reshape(n_fields, -1))
        power_t = accumulate((spec_t.real**2 + spec_t.imag**2).reshape(n_fields, -1))
        cross = accumulate(
            (spec_p.real * spec_t.real + spec_p.imag * spec_t.imag).reshape(
                n_fields, -1
            )
        )
        return power_p, power_t, cross, dc

    def spectral_excess(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-field AMSE excess ``AMSE - MSE`` (the pure anti-blur surcharge).

        Non-negative; zero iff, in every |k|-bin, the two power spectra match
        or the coherence is perfect. Shrinking predicted amplitude below the
        target's cannot reduce this term.
        """
        power_p, power_t, cross, _ = self.binned_spectra(prediction, target)
        geo_mean = torch.sqrt(self.eps + power_p * power_t)
        coherence = torch.clamp(cross / geo_mean, max=1.0)
        excess_weight = (torch.maximum(power_p, power_t) - geo_mean).clamp_min(0.0)
        contributions = 2.0 * excess_weight * (1.0 - coherence)
        height, width = prediction.shape[-2], prediction.shape[-1]
        return self._weight_bins(contributions, height, width).sum(dim=1)

    def pure_amse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-field AMSE (amplitude + adjusted decorrelation terms + DC term)."""
        power_p, power_t, cross, dc = self.binned_spectra(prediction, target)
        geo_mean = torch.sqrt(self.eps + power_p * power_t)
        coherence = torch.clamp(cross / geo_mean, max=1.0)
        amplitude = (power_p + power_t - 2.0 * geo_mean).clamp_min(0.0)
        decorrelation = 2.0 * torch.maximum(power_p, power_t) * (1.0 - coherence)
        height, width = prediction.shape[-2], prediction.shape[-1]
        contributions = self._weight_bins(amplitude + decorrelation, height, width)
        return contributions.sum(dim=1) + dc

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the scalar loss for prediction/target of shape [..., H, W]."""
        if self.mode == "pure":
            return self.pure_amse(prediction, target).mean()
        base = functional.huber_loss(
            prediction.float(), target.float(), delta=self.delta
        )
        return (
            base
            + self.spectral_weight * self.spectral_excess(prediction, target).mean()
        )
