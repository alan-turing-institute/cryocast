"""Controlled comparison of learned and interpolated downscaling outputs."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor

from cryocast.models import Downscaler


@dataclass(frozen=True, slots=True)
class DownscalingMethodSummary:
    """Aggregate error and spatial-detail metrics for one downscaling method."""

    mae: float
    rmse: float
    gradient_rmse: float
    high_frequency_fraction: float
    high_frequency_fraction_error: float


@dataclass(frozen=True, slots=True)
class DownscalingComparison:
    """Side-by-side interpolation and learned-downscaling evaluation."""

    interpolation: DownscalingMethodSummary
    learned: DownscalingMethodSummary
    target_high_frequency_fraction: float
    valid_values: int
    valid_gradient_edges: int
    spectral_frames: int
    batches: int
    mae_improvement_percent: float | None
    rmse_improvement_percent: float | None
    gradient_rmse_improvement_percent: float | None
    high_frequency_error_improvement_percent: float | None


@dataclass(slots=True)
class _MethodAccumulator:
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    valid_values: int = 0
    gradient_squared_error_sum: float = 0.0
    valid_gradient_edges: int = 0
    high_frequency_fraction_sum: float = 0.0
    spectral_frames: int = 0


@dataclass(slots=True)
class _TargetSpectrumAccumulator:
    high_frequency_fraction_sum: float = 0.0
    spectral_frames: int = 0


def _validate_shapes(
    prediction: Tensor,
    baseline: Tensor,
    target: Tensor,
    valid: Tensor,
) -> None:
    if prediction.shape != target.shape or baseline.shape != target.shape:
        msg = (
            "Prediction, interpolation baseline, and target must have matching "
            f"shapes, got {tuple(prediction.shape)}, {tuple(baseline.shape)}, "
            f"and {tuple(target.shape)}."
        )
        raise ValueError(msg)
    if valid.shape != target.shape or valid.dtype is not torch.bool:
        msg = "Validity mask must be boolean and match the target shape."
        raise ValueError(msg)
    if target.ndim != 5:  # noqa: PLR2004
        msg = f"Expected NTCHW tensors, got target shape {tuple(target.shape)}."
        raise ValueError(msg)


def _gradient_squared_error(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
) -> tuple[float, int]:
    """Return squared finite-difference error over valid horizontal/vertical edges."""
    squared_sum = torch.zeros((), device=prediction.device, dtype=torch.float64)
    count = 0

    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
    if bool(valid_dx.any()):
        diff = (pred_dx - target_dx)[valid_dx].double()
        squared_sum += torch.sum(diff * diff)
        count += int(valid_dx.sum().item())

    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
    if bool(valid_dy.any()):
        diff = (pred_dy - target_dy)[valid_dy].double()
        squared_sum += torch.sum(diff * diff)
        count += int(valid_dy.sum().item())

    return float(squared_sum.item()), count


def _high_frequency_fractions(
    values: Tensor,
    valid: Tensor,
    *,
    cutoff: float,
) -> Tensor:
    """Return high-frequency power fractions for valid NTCHW frames.

    Invalid cells are zeroed after subtracting each frame's valid-cell mean. The same
    validity mask is applied to target, interpolation, and learned fields, so mask-edge
    spectral leakage is shared across all three comparisons.
    """
    if not 0.0 < cutoff < 1.0:
        msg = "high_frequency_cutoff must be between 0 and 1."
        raise ValueError(msg)

    height, width = values.shape[-2:]
    frames = values.reshape(-1, height, width)
    masks = valid.reshape(-1, height, width)

    freq_y = torch.fft.fftfreq(height, device=values.device)
    freq_x = torch.fft.rfftfreq(width, device=values.device)
    radius = torch.sqrt(freq_y[:, None] ** 2 + freq_x[None, :] ** 2)
    max_radius = sqrt(0.5**2 + 0.5**2)
    high_frequency = radius >= cutoff * max_radius

    fractions: list[Tensor] = []
    for frame, mask in zip(frames, masks, strict=True):
        n_valid = int(mask.sum().item())
        if n_valid < 2:  # noqa: PLR2004
            continue
        frame_mean = frame[mask].mean()
        centered = torch.where(mask, frame - frame_mean, torch.zeros_like(frame))
        spectrum = torch.fft.rfft2(centered, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        total_power = power.sum()
        if not bool(torch.isfinite(total_power)) or float(total_power.item()) <= 0.0:
            continue
        fractions.append(power[high_frequency].sum() / total_power)

    if not fractions:
        return torch.empty(0, dtype=values.dtype, device=values.device)
    return torch.stack(fractions)


def _update_method(
    accumulator: _MethodAccumulator,
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    high_frequency_cutoff: float,
) -> None:
    errors = (prediction - target)[valid].double()
    accumulator.abs_error_sum += float(torch.abs(errors).sum().item())
    accumulator.squared_error_sum += float(torch.square(errors).sum().item())
    accumulator.valid_values += errors.numel()

    gradient_squared_error, gradient_count = _gradient_squared_error(
        prediction,
        target,
        valid,
    )
    accumulator.gradient_squared_error_sum += gradient_squared_error
    accumulator.valid_gradient_edges += gradient_count

    high_frequency = _high_frequency_fractions(
        prediction,
        valid,
        cutoff=high_frequency_cutoff,
    )
    accumulator.high_frequency_fraction_sum += float(
        high_frequency.double().sum().item()
    )
    accumulator.spectral_frames += high_frequency.numel()


def _summarise_method(
    accumulator: _MethodAccumulator,
    *,
    target_high_frequency_fraction: float,
) -> DownscalingMethodSummary:
    if accumulator.valid_values <= 0:
        msg = "No valid target cells were available for evaluation."
        raise ValueError(msg)
    if accumulator.valid_gradient_edges <= 0:
        msg = "No valid adjacent target cells were available for gradient evaluation."
        raise ValueError(msg)
    if accumulator.spectral_frames <= 0:
        msg = "No valid target frames were available for spectral evaluation."
        raise ValueError(msg)

    high_frequency_fraction = (
        accumulator.high_frequency_fraction_sum / accumulator.spectral_frames
    )
    return DownscalingMethodSummary(
        mae=accumulator.abs_error_sum / accumulator.valid_values,
        rmse=sqrt(accumulator.squared_error_sum / accumulator.valid_values),
        gradient_rmse=sqrt(
            accumulator.gradient_squared_error_sum / accumulator.valid_gradient_edges
        ),
        high_frequency_fraction=high_frequency_fraction,
        high_frequency_fraction_error=abs(
            high_frequency_fraction - target_high_frequency_fraction
        ),
    )


def _improvement_percent(baseline: float, learned: float) -> float | None:
    if baseline <= 0.0:
        return None
    return 100.0 * (baseline - learned) / baseline


def compare_downscaler(
    model: Downscaler,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: str | torch.device = "cpu",
    high_frequency_cutoff: float = 0.5,
    max_batches: int | None = None,
) -> DownscalingComparison:
    """Compare learned downscaling with geographic interpolation on identical batches.

    Both methods consume exactly the same low-resolution SIC inputs and are evaluated
    against the same high-resolution target cells.
    """
    if max_batches is not None and max_batches <= 0:
        msg = "max_batches must be greater than zero when provided."
        raise ValueError(msg)

    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    model.eval()

    interpolation_accumulator = _MethodAccumulator()
    learned_accumulator = _MethodAccumulator()
    target_spectrum = _TargetSpectrumAccumulator()
    batch_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            if model.source_group_name not in batch or "target" not in batch:
                msg = (
                    "Each evaluation batch must contain the configured source group "
                    f"{model.source_group_name!r} and a 'target' tensor."
                )
                raise KeyError(msg)

            source = torch.as_tensor(
                batch[model.source_group_name],
                dtype=torch.float32,
                device=resolved_device,
            )
            target = torch.as_tensor(
                batch["target"],
                dtype=torch.float32,
                device=resolved_device,
            )
            learned = model({model.source_group_name: source})
            source_sic = source[
                :,
                -model.n_forecast_steps :,
                model.source_variable_index : model.source_variable_index + 1,
                :,
                :,
            ]
            interpolation = model.interpolation_baseline(source_sic)

            valid = torch.isfinite(target)
            if model.validity_mask is not None:
                mask = model.validity_mask.to(device=resolved_device)
                valid &= mask.view(1, 1, 1, *model.output_space.shape)

            _validate_shapes(learned, interpolation, target, valid)
            _update_method(
                interpolation_accumulator,
                interpolation,
                target,
                valid,
                high_frequency_cutoff=high_frequency_cutoff,
            )
            _update_method(
                learned_accumulator,
                learned,
                target,
                valid,
                high_frequency_cutoff=high_frequency_cutoff,
            )
            target_high_frequency = _high_frequency_fractions(
                target,
                valid,
                cutoff=high_frequency_cutoff,
            )
            target_spectrum.high_frequency_fraction_sum += float(
                target_high_frequency.double().sum().item()
            )
            target_spectrum.spectral_frames += target_high_frequency.numel()
            batch_count += 1

    if batch_count == 0:
        msg = "No evaluation batches were consumed."
        raise ValueError(msg)
    if target_spectrum.spectral_frames <= 0:
        msg = "No valid target frames were available for spectral evaluation."
        raise ValueError(msg)

    target_high_frequency_fraction = (
        target_spectrum.high_frequency_fraction_sum / target_spectrum.spectral_frames
    )
    interpolation_summary = _summarise_method(
        interpolation_accumulator,
        target_high_frequency_fraction=target_high_frequency_fraction,
    )
    learned_summary = _summarise_method(
        learned_accumulator,
        target_high_frequency_fraction=target_high_frequency_fraction,
    )

    return DownscalingComparison(
        interpolation=interpolation_summary,
        learned=learned_summary,
        target_high_frequency_fraction=target_high_frequency_fraction,
        valid_values=learned_accumulator.valid_values,
        valid_gradient_edges=learned_accumulator.valid_gradient_edges,
        spectral_frames=target_spectrum.spectral_frames,
        batches=batch_count,
        mae_improvement_percent=_improvement_percent(
            interpolation_summary.mae,
            learned_summary.mae,
        ),
        rmse_improvement_percent=_improvement_percent(
            interpolation_summary.rmse,
            learned_summary.rmse,
        ),
        gradient_rmse_improvement_percent=_improvement_percent(
            interpolation_summary.gradient_rmse,
            learned_summary.gradient_rmse,
        ),
        high_frequency_error_improvement_percent=_improvement_percent(
            interpolation_summary.high_frequency_fraction_error,
            learned_summary.high_frequency_fraction_error,
        ),
    )
