"""Auditable common-mask forecast metrics and frozen continuation schedules."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SEEDS = (575, 576, 577)
NORMS = ("batchnorm", "groupnorm")
INITIAL_STEPS = 96
EXTRA_STEPS = 192
VALIDATE_EVERY = 48
RESTART_LR = 1e-3
FINAL_LR = 1e-5
LEADS = 14
PIXEL_AREA_KM2 = 625.0
SIC_THRESHOLD = 0.15


def require(condition: object, message: str) -> None:
    """Reject missing or unmatched evidence explicitly."""
    if not condition:
        raise ValueError(message)


def continuation_lr(index: int, steps: int = EXTRA_STEPS) -> float:
    """Predeclared cosine restart, inclusive of its first and final updates."""
    require(steps > 1 and 0 <= index < steps, "Invalid continuation update index")
    return FINAL_LR + 0.5 * (RESTART_LR - FINAL_LR) * (
        1.0 + math.cos(math.pi * index / (steps - 1))
    )


def training_order(seed: int, count: int, steps: int = EXTRA_STEPS) -> np.ndarray:
    """Use an arm-independent, non-repeating permutation of training windows."""
    require(count >= steps > 0, "Insufficient distinct training windows")
    return np.random.default_rng(594_000 + seed).permutation(count)[:steps]


def per_lead_metrics(
    prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray
) -> list[dict[str, Any]]:
    """Compute raw sufficient statistics on one matched TCHW forecast window."""
    require(prediction.shape == truth.shape == valid.shape, "Metric axes differ")
    require(prediction.ndim == 4, "Expected TCHW arrays")  # noqa: PLR2004
    require(valid.dtype == np.bool_, "Mask must be boolean")
    require(np.isfinite(truth[valid]).all(), "Truth must be finite on valid support")
    require(np.isfinite(prediction[valid]).all(), "Nonfinite forecast at valid truth")
    output = []
    for lead, (pred, target, mask) in enumerate(
        zip(prediction, truth, valid, strict=True), 1
    ):
        count = int(mask.sum())
        require(count > 0, "Every scored lead requires valid cells")
        difference = pred.astype(np.float64)[mask] - target.astype(np.float64)[mask]
        signed_extent = int(((pred > SIC_THRESHOLD) & mask).sum()) - int(
            ((target > SIC_THRESHOLD) & mask).sum()
        )
        output.append(
            {
                "lead": lead,
                "count": count,
                "absolute_sum": float(np.abs(difference).sum()),
                "squared_sum": float(np.square(difference).sum()),
                "signed_sie_km2": signed_extent * PIXEL_AREA_KM2,
                "absolute_sie_km2": abs(signed_extent) * PIXEL_AREA_KM2,
            }
        )
    return output
