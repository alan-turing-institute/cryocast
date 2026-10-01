import math

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional

from icenet_mp.losses.amse_loss import AMSELoss, AMSEMode


def make_fields(
    seed: int = 0, shape: tuple[int, ...] = (2, 2, 1, 32, 32)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a seeded (prediction, target) pair of random fields."""
    generator = torch.Generator().manual_seed(seed)
    prediction = torch.rand(*shape, generator=generator)
    target = torch.rand(*shape, generator=generator)
    return prediction, target


def blur(field: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Box-blur the trailing spatial dimensions of an [N, T, C, H, W] tensor."""
    n, t, c, h, w = field.shape
    flat = field.reshape(n * t * c, 1, h, w)
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size**2
    blurred = functional.conv2d(flat, kernel, padding=kernel_size // 2)
    return blurred.reshape(n, t, c, h, w)


# Every behaviour that must hold with and without the FastNet upweighting.
parametrize_weight = pytest.mark.parametrize(
    "wavenumber_weight", [None, "fastnet"], ids=["unweighted", "fastnet"]
)


class TestAMSELoss:
    @parametrize_weight
    def test_config_instantiation(self, wavenumber_weight: str | None) -> None:
        config = OmegaConf.create(
            {
                "_target_": "icenet_mp.losses.amse_loss.AMSELoss",
                "mode": "hybrid",
                "spectral_weight": 0.1,
                "wavenumber_weight": wavenumber_weight,
            }
        )
        loss_fn = hydra.utils.instantiate(config)
        assert isinstance(loss_fn, AMSELoss)
        assert loss_fn.wavenumber_weight == (wavenumber_weight or "none")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            AMSELoss(mode="not-a-mode")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="merge_bins_below"):
            AMSELoss(merge_bins_below=0)

    @parametrize_weight
    @pytest.mark.parametrize("mode", ["hybrid", "pure"], ids=["hybrid", "pure"])
    def test_identity_is_zero(
        self, mode: AMSEMode, wavenumber_weight: str | None
    ) -> None:
        """A perfect match is zero (and the 0/0 in the gamma normaliser is not NaN)."""
        prediction, _ = make_fields()
        loss_fn = AMSELoss(mode=mode, wavenumber_weight=wavenumber_weight)
        loss = loss_fn(prediction, prediction)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_scalar_output(self) -> None:
        prediction, target = make_fields()
        loss = AMSELoss()(prediction, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    @pytest.mark.parametrize(
        ("seed", "shape"),
        [(1, (2, 2, 1, 32, 32)), (2, (2, 1, 1, 31, 33))],
        ids=["even-grid", "odd-grid"],
    )
    def test_parseval_identity(self, seed: int, shape: tuple[int, ...]) -> None:
        """Binned spectra must reconstruct the spatial MSE exactly (Parseval).

        The odd grid checks the rfft factor-2 bookkeeping for odd widths.
        """
        prediction, target = make_fields(seed=seed, shape=shape)
        loss_fn = AMSELoss()
        power_p, power_t, cross, dc = loss_fn.binned_spectra(prediction, target)
        spectral_mse = (power_p + power_t - 2.0 * cross).sum(dim=1) + dc
        spatial_mse = (
            ((prediction - target) ** 2).reshape(spectral_mse.shape[0], -1).mean(dim=1)
        )
        assert torch.allclose(spectral_mse, spatial_mse, rtol=1e-4, atol=1e-7)

    @parametrize_weight
    def test_blur_is_penalized_and_monotone(
        self, wavenumber_weight: str | None
    ) -> None:
        """The spectral excess is positive for blurred predictions and grows with blur width."""
        _, target = make_fields(seed=3)
        loss_fn = AMSELoss(wavenumber_weight=wavenumber_weight)
        excess_light = loss_fn.spectral_excess(blur(target, 3), target).mean()
        excess_heavy = loss_fn.spectral_excess(blur(target, 7), target).mean()
        assert excess_light.item() > 0.0
        assert excess_heavy.item() > excess_light.item()

    def test_amplitude_gradient_points_up(self) -> None:
        """For a uniformly damped prediction the spectral gradient pushes amplitude up."""
        _, target = make_fields(seed=4)
        scale = torch.tensor(0.5, requires_grad=True)
        excess = AMSELoss().spectral_excess(scale * target, target).mean()
        (gradient,) = torch.autograd.grad(excess, scale)
        assert gradient.item() < 0.0

    @parametrize_weight
    @pytest.mark.parametrize("mode", ["hybrid", "pure"], ids=["hybrid", "pure"])
    @pytest.mark.parametrize(
        "case",
        ["random", "zero", "masked", "identical"],
        ids=["random", "zero", "masked", "identical"],
    )
    def test_loss_and_gradients_stay_finite(
        self, mode: AMSEMode, case: str, wavenumber_weight: str | None
    ) -> None:
        """Degenerate fields (all-zero, masked to an active region, identical) stay finite."""
        prediction, target = make_fields(seed=16)
        if case == "zero":
            prediction = torch.zeros_like(target)
        elif case == "identical":
            prediction = target.clone()
        elif case == "masked":
            mask = torch.zeros(32, 32)
            mask[8:24, 8:24] = 1.0
            prediction, target = prediction * mask, target * mask
        prediction = prediction.detach().requires_grad_()
        loss = AMSELoss(mode=mode, wavenumber_weight=wavenumber_weight)(
            prediction, target
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()

    @parametrize_weight
    def test_spectral_part_is_symmetric(self, wavenumber_weight: str | None) -> None:
        prediction, target = make_fields(seed=7)
        loss_fn = AMSELoss(mode="pure", wavenumber_weight=wavenumber_weight)
        assert torch.allclose(loss_fn(prediction, target), loss_fn(target, prediction))

    @parametrize_weight
    def test_hybrid_reduces_to_huber_at_zero_weight(
        self, wavenumber_weight: str | None
    ) -> None:
        prediction, target = make_fields(seed=8)
        hybrid = AMSELoss(
            mode="hybrid", spectral_weight=0.0, wavenumber_weight=wavenumber_weight
        )(prediction, target)
        huber = functional.huber_loss(prediction, target, delta=0.5)
        assert torch.allclose(hybrid, huber)

    @parametrize_weight
    def test_low_precision_input(self, wavenumber_weight: str | None) -> None:
        prediction, target = make_fields(seed=9)
        loss = AMSELoss(wavenumber_weight=wavenumber_weight)(
            prediction.to(torch.bfloat16), target.to(torch.bfloat16)
        )
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)


def penalty_by_bin(
    loss_fn: AMSELoss, prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (bin wavenumbers, batch-summed anti-blur contribution per bin)."""
    power_p, power_t, cross, _ = loss_fn.binned_spectra(prediction, target)
    geo_mean = torch.sqrt(loss_fn.eps + power_p * power_t)
    coherence = torch.clamp(cross / geo_mean, max=1.0)
    excess = (torch.maximum(power_p, power_t) - geo_mean).clamp_min(0.0)
    contributions = 2.0 * excess * (1.0 - coherence)
    height, width = prediction.shape[-2], prediction.shape[-1]
    weighted = loss_fn._weight_bins(contributions, height, width)
    _, _, present, _ = loss_fn._binning(height, width, prediction.device)
    return present.float(), weighted.sum(dim=0)


def share_above(
    wavenumbers: torch.Tensor, contributions: torch.Tensor, cut: float
) -> float:
    """Fraction (%) of the total penalty carried by bins with |k| > cut."""
    total = float(contributions.sum())
    return 100.0 * float(contributions[wavenumbers > cut].sum()) / total


class TestAMSELossWavenumberWeight:
    """The optional FastNet upweighting gamma_k = max(N_k * k**sqrt(3), 1)."""

    # Values produced by the implementation BEFORE wavenumber_weight existed
    # (fields = make_fields(seed=11), CPU, float32). Any drift in the default
    # path moves frozen anchors and existing runs, so it must fail here.
    GOLDEN_HYBRID_VALUE = 7.9457767308e-02
    GOLDEN_HYBRID_GRAD_NORM = 5.4061640985e-03
    GOLDEN_PURE_VALUE = 1.8319147825e-01
    GOLDEN_PURE_GRAD_NORM = 1.6285512596e-02
    GOLDEN_EXCESS = (
        1.9828230143e-02,
        1.6821861267e-02,
        1.6672907397e-02,
        1.7617737874e-02,
    )

    def test_default_is_off(self) -> None:
        assert AMSELoss().wavenumber_weight == "none"

    def test_unknown_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="wavenumber_weight"):
            AMSELoss(wavenumber_weight="fastnetish")

    @pytest.mark.parametrize(
        "off",
        [None, "none", "None", "NONE"],
        ids=["python-none", "lower", "title", "upper"],
    )
    @pytest.mark.parametrize("mode", ["hybrid", "pure"], ids=["hybrid", "pure"])
    def test_off_is_bitwise_identical_to_default(
        self, off: str | None, mode: AMSEMode
    ) -> None:
        """OFF must not perturb the existing loss by even one ulp."""
        prediction, target = make_fields(seed=10)
        prediction = prediction.requires_grad_()
        reference = AMSELoss(mode=mode)(prediction, target)
        (reference_grad,) = torch.autograd.grad(reference, prediction)
        candidate = AMSELoss(mode=mode, wavenumber_weight=off)(prediction, target)
        (candidate_grad,) = torch.autograd.grad(candidate, prediction)
        assert torch.equal(reference, candidate)
        assert torch.equal(reference_grad, candidate_grad)

    def test_off_matches_pre_change_reference_values(self) -> None:
        """Golden values captured from the implementation before this option."""
        prediction, target = make_fields(seed=11)
        prediction = prediction.requires_grad_()

        hybrid = AMSELoss(mode="hybrid", spectral_weight=0.1)(prediction, target)
        (hybrid_grad,) = torch.autograd.grad(hybrid, prediction)
        assert hybrid.item() == pytest.approx(self.GOLDEN_HYBRID_VALUE, rel=1e-6)
        assert hybrid_grad.norm().item() == pytest.approx(
            self.GOLDEN_HYBRID_GRAD_NORM, rel=1e-6
        )

        pure = AMSELoss(mode="pure")(prediction, target)
        (pure_grad,) = torch.autograd.grad(pure, prediction)
        assert pure.item() == pytest.approx(self.GOLDEN_PURE_VALUE, rel=1e-6)
        assert pure_grad.norm().item() == pytest.approx(
            self.GOLDEN_PURE_GRAD_NORM, rel=1e-6
        )

        excess = AMSELoss().spectral_excess(prediction.detach(), target)
        assert excess.tolist() == pytest.approx(list(self.GOLDEN_EXCESS), rel=1e-6)

    @pytest.mark.parametrize(
        ("height", "width"),
        [(32, 32), (31, 33), (48, 64)],
        ids=["32x32", "31x33", "48x64"],
    )
    def test_gamma_matches_the_published_formula(self, height: int, width: int) -> None:
        """gamma_k == max(N_k * k**sqrt(3), 1), N_k counted on the FULL 2D DFT."""
        loss_fn = AMSELoss()
        _, _, present, gamma = loss_fn._binning(height, width, torch.device("cpu"))
        ky = np.fft.fftfreq(height) * height
        kx = np.fft.fftfreq(width) * width
        kmag = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        bins = np.round(kmag).astype(int)
        bins = np.where(
            (bins >= 1) & (bins < loss_fn.merge_bins_below),
            loss_fn.merge_bins_below,
            bins,
        )
        n_bins = int(bins.max()) + 1
        mode_counts = np.zeros(n_bins)
        for i in range(height):
            for j in range(width):
                if i == 0 and j == 0:
                    continue  # the DC mode belongs to no annulus
                mode_counts[bins[i, j]] += 1.0
        expected = np.maximum(mode_counts * np.arange(n_bins) ** math.sqrt(3.0), 1.0)[
            present.numpy()
        ]
        assert gamma.numpy() == pytest.approx(expected, rel=1e-6)
        # The whole plane is accounted for, DC excluded.
        assert mode_counts.sum() == height * width - 1

    def test_gamma_moves_the_penalty_to_fine_scales(self) -> None:
        """The declared purpose: the high-|k| share of the penalty must rise."""
        _, target = make_fields(seed=12, shape=(2, 2, 1, 64, 64))
        prediction = blur(target, 5)
        wavenumbers, unweighted = penalty_by_bin(AMSELoss(), prediction, target)
        _, weighted = penalty_by_bin(
            AMSELoss(wavenumber_weight="fastnet"), prediction, target
        )
        for cut in (4, 10, 20, 30):
            assert share_above(wavenumbers, weighted, cut) > share_above(
                wavenumbers, unweighted, cut
            )

        # ... and the median wavenumber of the penalty moves up.
        def median_k(contributions: torch.Tensor) -> float:
            cumulative = torch.cumsum(contributions / contributions.sum(), dim=0)
            return float(wavenumbers[int(torch.searchsorted(cumulative, 0.5))])

        assert median_k(weighted) > median_k(unweighted)

    def test_gamma_preserves_the_batch_total(self) -> None:
        """Scale-preserving normalisation: same magnitude, different shape."""
        _, target = make_fields(seed=13, shape=(3, 2, 1, 32, 32))
        prediction = blur(target, 3)
        off = AMSELoss().spectral_excess(prediction, target)
        on = AMSELoss(wavenumber_weight="fastnet").spectral_excess(prediction, target)
        assert float(on.sum()) == pytest.approx(float(off.sum()), rel=1e-5)
        # The redistribution is real: individual fields do move.
        assert not torch.allclose(on, off, rtol=1e-3)

    def test_gamma_changes_the_gradient_not_the_scalar_value(self) -> None:
        """Same loss number, different descent direction (the point of the flag)."""
        _, target = make_fields(seed=14, shape=(2, 2, 1, 32, 32))
        blurred = blur(target, 3)
        grads = []
        for weight in (None, "fastnet"):
            prediction = blurred.clone().requires_grad_()
            loss = AMSELoss(spectral_weight=1.0, wavenumber_weight=weight)(
                prediction, target
            )
            (gradient,) = torch.autograd.grad(loss, prediction)
            grads.append((loss.detach().item(), gradient))
        assert grads[0][0] == pytest.approx(grads[1][0], rel=1e-5)
        cosine = float(
            functional.cosine_similarity(
                grads[0][1].flatten(), grads[1][1].flatten(), dim=0
            )
        )
        assert cosine < 0.999
        # The spectral branch stays the same order of magnitude, so the arm is
        # trainable at the unweighted twin's learning rate.
        ratio = float(grads[1][1].norm() / grads[0][1].norm())
        assert 0.1 < ratio < 10.0

    def test_gamma_leaves_the_dc_term_unweighted(self) -> None:
        """A pure mean offset has no annular content: pure AMSE == the DC term."""
        _, target = make_fields(seed=18)
        prediction = target + 0.25
        for weight in (None, "fastnet"):
            loss = AMSELoss(mode="pure", wavenumber_weight=weight)(prediction, target)
            assert loss.item() == pytest.approx(0.25**2, rel=1e-5)
