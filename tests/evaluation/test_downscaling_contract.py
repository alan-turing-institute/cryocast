"""Independent numerical and matching checks for downscaling comparisons."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from cryocast.evaluation.downscaling import (
    _high_frequency_fractions,
    compare_downscaler,
)

if TYPE_CHECKING:
    from cryocast.models import Downscaler


class _ComparisonModel(nn.Module):
    """Expose fixed predictions to isolate evaluation arithmetic from training."""

    source_group_name = "source"
    source_variable_index = 0
    n_forecast_steps = 1
    validity_mask = None

    def __init__(self, prediction: Tensor | None = None) -> None:
        """Use the source itself unless a learned forecast is supplied."""
        super().__init__()
        self.prediction = prediction

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """Return a controlled learned forecast."""
        return inputs["source"] if self.prediction is None else self.prediction

    def interpolation_baseline(self, source: Tensor) -> Tensor:
        """Use a fixed reference, not another implementation of the tested metric."""
        return source


def _wave(*, high: bool = False) -> Tensor:
    x = torch.arange(8, dtype=torch.float32)
    wave = torch.cos(2 * torch.pi * x / 8) if not high else (-1.0) ** x
    return (0.5 + 0.2 * wave).expand(8, 8).reshape(1, 1, 1, 8, 8).clone()


@pytest.mark.parametrize("shape", [(7, 8), (8, 9), (8, 8)])
@pytest.mark.parametrize("cutoff", [0.25, 0.5, 0.75])
def test_real_fft_power_matches_independent_full_complex_fft(
    shape: tuple[int, int], cutoff: float
) -> None:
    """Interior real-FFT columns carry the power of both conjugate frequencies."""
    rng = np.random.default_rng(595)
    array = rng.normal(size=shape)
    mask = rng.random(size=shape) > 0.2
    centered = np.where(mask, array - array[mask].mean(), 0.0)
    full_power = np.abs(np.fft.fft2(centered, norm="ortho")) ** 2
    # Compare the inclusive cutoff before taking a square root: SIMD sqrt and
    # scalar sqrt can round an exact boundary bin to different sides.
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    high_frequency = fy * fy + fx * fx >= cutoff * cutoff / 2.0
    expected = full_power[high_frequency].sum() / full_power.sum()
    actual = _high_frequency_fractions(
        torch.from_numpy(array).reshape(1, 1, 1, *shape),
        torch.from_numpy(mask).reshape(1, 1, 1, *shape),
        cutoff=cutoff,
    )
    assert actual.item() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_constant_frames_keep_the_same_spectral_denominator() -> None:
    """A constant prediction is a valid smooth field, not an omitted sample."""
    source = torch.full((1, 1, 1, 8, 8), 0.5)
    target = _wave()
    result = compare_downscaler(
        cast("Downscaler", _ComparisonModel()), [{"source": source, "target": target}]
    )
    assert result.spectral_frames == 1
    assert result.learned.high_frequency_fraction == 0.0
    assert np.isfinite(result.learned.mae)


def test_spectral_error_is_paired_before_averaging() -> None:
    """Opposite errors on two dates must not cancel into a perfect result."""
    target = torch.cat([_wave(), _wave(high=True)])
    learned = target.flip(0)
    result = compare_downscaler(
        cast("Downscaler", _ComparisonModel(learned)),
        [{"source": target, "target": target}],
    )
    assert result.interpolation.high_frequency_fraction_error == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.learned.high_frequency_fraction_error == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_learned_prediction_is_an_evaluation_failure(bad: float) -> None:
    """An invalid prediction must not become an output JSON NaN or dropped frame."""
    target = _wave()
    learned = target.clone()
    learned[..., 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        compare_downscaler(
            cast("Downscaler", _ComparisonModel(learned)),
            [{"source": target, "target": target}],
        )


def test_max_batches_does_not_read_an_extra_dataset_batch() -> None:
    """A bounded smoke check must not trigger an additional download/read."""

    def batches() -> Iterator[dict[str, Tensor]]:
        yield {"source": _wave(), "target": _wave()}
        msg = "Read beyond max_batches"
        raise AssertionError(msg)

    result = compare_downscaler(
        cast("Downscaler", _ComparisonModel()), batches(), max_batches=1
    )
    assert result.batches == 1


def test_unequal_batches_preserve_cell_and_frame_weighting() -> None:
    """Pooled errors are weighted by values and spectral errors by paired frames."""
    source = torch.cat([_wave(), _wave(high=True), _wave()])
    target = torch.cat([_wave(high=True), _wave(), _wave()])
    model = cast("Downscaler", _ComparisonModel())
    pooled = compare_downscaler(model, [{"source": source, "target": target}])
    split = compare_downscaler(
        model,
        [
            {"source": source[:2], "target": target[:2]},
            {"source": source[2:], "target": target[2:]},
        ],
    )
    errors = source.double() - target.double()
    assert pooled.learned.mae == pytest.approx(errors.abs().mean().item())
    assert pooled.learned.rmse == pytest.approx(errors.square().mean().sqrt().item())
    assert pooled.learned == split.learned
    assert pooled.spectral_frames == split.spectral_frames == 3


@pytest.mark.parametrize("shape", [(8, 8), (8, 12), (16, 16), (16, 32)])
@pytest.mark.parametrize(
    ("cutoff", "expected"),
    [
        (float(np.nextafter(0.5, 0.0)), 1.0),
        (0.5, 1.0),
        (float(np.nextafter(0.5, 1.0)), 0.0),
    ],
)
def test_spectral_cutoff_includes_exact_boundary_without_tolerance(
    shape: tuple[int, int], cutoff: float, expected: float
) -> None:
    """A diagonal quarter-frequency wave sits exactly on the default boundary."""
    y = torch.arange(shape[0], dtype=torch.float64)[:, None]
    x = torch.arange(shape[1], dtype=torch.float64)[None, :]
    values = torch.cos((torch.pi / 2.0) * (x + y)).reshape(1, 1, 1, *shape)
    fraction = _high_frequency_fractions(
        values, torch.ones_like(values, dtype=torch.bool), cutoff=cutoff
    )
    assert fraction.item() == pytest.approx(expected, abs=1e-12)
