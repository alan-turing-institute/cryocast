"""Evaluation helpers for controlled model comparisons."""

from .downscaling import (
    DownscalingComparison,
    DownscalingMethodSummary,
    compare_downscaler,
)

__all__ = [
    "DownscalingComparison",
    "DownscalingMethodSummary",
    "compare_downscaler",
]
