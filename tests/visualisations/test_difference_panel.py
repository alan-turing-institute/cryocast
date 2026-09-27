"""Tests for icenet_mp/visualisations/difference_calculator.py.

Covers the difference and standardised-difference computations for a
ground-truth/prediction pair.
"""

import numpy as np
import pytest

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import DiffMode
from icenet_mp.visualisations.difference_panel import (
    DifferencePanel,
)


class TestComputeDifference:
    ground_truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = np.array([[0.5, 2.5], [2.0, 5.0]])

    def test_signed(self) -> None:
        """Signed difference is ground_truth - prediction."""
        result = DifferencePanel(
            DiffMode.SIGNED, self.ground_truth, self.prediction
        ).difference

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])

    def test_absolute(self) -> None:
        """Absolute difference is |ground_truth - prediction|."""
        result = DifferencePanel(
            DiffMode.ABSOLUTE, self.ground_truth, self.prediction
        ).difference

        np.testing.assert_allclose(result, [[0.5, 0.5], [1.0, 1.0]])

    def test_smape(self) -> None:
        """SMAPE difference normalises the absolute error by the mean magnitude."""
        result = DifferencePanel(
            DiffMode.SMAPE, self.ground_truth, self.prediction
        ).difference

        expected = np.array(
            [
                [0.5 / 0.75, 0.5 / 2.25],
                [1.0 / 2.5, 1.0 / 4.5],
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_smape_avoids_division_by_zero(self) -> None:
        """A near-zero denominator is clipped rather than dividing by zero."""
        ground_truth = np.array([0.0])
        prediction = np.array([0.0])

        result = DifferencePanel(DiffMode.SMAPE, ground_truth, prediction).difference

        assert np.isfinite(result).all()

    def test_rejects_shape_mismatch(self) -> None:
        """Reject ground truth/prediction arrays with mismatched shapes."""
        with pytest.raises(InvalidArrayError, match="matching shapes"):
            DifferencePanel(
                DiffMode.SIGNED,
                self.ground_truth,
                np.zeros((3, 3)),
            )

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised difference mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown difference mode: dummy"):
            _ = DifferencePanel(
                "dummy",  # type: ignore[arg-type]
                self.ground_truth,
                self.prediction,
            ).difference

    def test_falls_back_to_raw_difference_without_uncertainty(self) -> None:
        """Without uncertainty, `difference` is the raw diff_mode-based value."""
        result = DifferencePanel(
            DiffMode.SIGNED, self.ground_truth, self.prediction
        ).difference

        np.testing.assert_allclose(result, [[0.5, -0.5], [1.0, -1.0]])


class TestComputeStandardisedDifference:
    def test_basic(self) -> None:
        """Compute signed prediction error in units of uncertainty."""
        ground_truth = np.array([[0.5, 0.8], [0.4, 0.2]], dtype=np.float32)
        prediction = np.array([[0.4, 0.6], [0.3, 0.5]], dtype=np.float32)
        uncertainty = np.array([[0.1, 0.2], [0.05, 0.1]], dtype=np.float32)

        result = DifferencePanel(
            DiffMode.SIGNED, ground_truth, prediction, uncertainty
        ).standardised_difference

        np.testing.assert_allclose(result, [[1.0, 1.0], [2.0, -3.0]])

    def test_masks_invalid_uncertainty(self) -> None:
        """Mask values where uncertainty is zero, negative or non-finite."""
        ground_truth = np.ones((2, 2), dtype=np.float32)
        prediction = np.zeros((2, 2), dtype=np.float32)
        uncertainty = np.array([[0.5, 0.0], [-1.0, np.nan]], dtype=np.float32)

        result = DifferencePanel(
            DiffMode.SIGNED, ground_truth, prediction, uncertainty
        ).standardised_difference

        assert result[0, 0] == pytest.approx(2.0)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 0])
        assert np.isnan(result[1, 1])

    def test_rejects_shape_mismatch(self) -> None:
        """Reject input arrays with mismatched shapes."""
        with pytest.raises(InvalidArrayError, match="matching shapes"):
            DifferencePanel(
                DiffMode.SIGNED,
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((3, 3), dtype=np.float32),
            )

    def test_rejects_1d_arrays(self) -> None:
        """Reject 1D ground truth/prediction/uncertainty arrays."""
        with pytest.raises(InvalidArrayError, match="Expected 2D"):
            DifferencePanel(
                DiffMode.SIGNED,
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            )

    def test_accepts_3d_video_arrays(self) -> None:
        """Compute a standardised difference over a [T, H, W] video stack."""
        ground_truth = np.ones((2, 2, 2), dtype=np.float32)
        prediction = np.zeros((2, 2, 2), dtype=np.float32)
        uncertainty = np.full((2, 2, 2), 0.5, dtype=np.float32)

        result = DifferencePanel(
            DiffMode.SIGNED, ground_truth, prediction, uncertainty
        ).standardised_difference

        assert result.shape == (2, 2, 2)
        np.testing.assert_allclose(result, 2.0)

    def test_requires_uncertainty(self) -> None:
        """Accessing standardised_difference without uncertainty raises."""
        ground_truth = np.ones((2, 2), dtype=np.float32)
        prediction = np.zeros((2, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="requires `uncertainty` to be provided"):
            _ = DifferencePanel(
                DiffMode.SIGNED, ground_truth, prediction
            ).standardised_difference

    def test_difference_uses_standardised_when_uncertainty_given(self) -> None:
        """`difference` reflects the standardised value once uncertainty is bound."""
        ground_truth = np.ones((2, 2), dtype=np.float32)
        prediction = np.zeros((2, 2), dtype=np.float32)
        uncertainty = np.full((2, 2), 0.5, dtype=np.float32)

        calc = DifferencePanel(DiffMode.SIGNED, ground_truth, prediction, uncertainty)

        np.testing.assert_allclose(calc.difference, calc.standardised_difference)
        np.testing.assert_allclose(calc.difference, 2.0)


class TestMakeDiffColourmap:
    def test_signed(self) -> None:
        """A signed sample yields a symmetric range around its largest extreme."""
        ground_truth = np.array([-2.0, 3.0, 0.5])
        prediction = np.zeros(3)

        scale = DifferencePanel(DiffMode.SIGNED, ground_truth, prediction).colour_scale

        assert scale.vmin == pytest.approx(-3.0)
        assert scale.vmax == pytest.approx(3.0)
        assert scale.cmap == "RdBu_r"

    def test_signed_below_one_still_uses_unit_floor(self) -> None:
        """A small-magnitude sample still gets at least a +/-1 symmetric range."""
        ground_truth = np.array([0.1])
        prediction = np.zeros(1)

        scale = DifferencePanel(DiffMode.SIGNED, ground_truth, prediction).colour_scale

        assert scale.vmin == pytest.approx(-1.0)
        assert scale.vmax == pytest.approx(1.0)

    def test_absolute(self) -> None:
        """Absolute mode sets vmax from the difference array's max."""
        ground_truth = np.array([0.1, 0.9, 0.4])
        prediction = np.zeros(3)

        scale = DifferencePanel(
            DiffMode.ABSOLUTE, ground_truth, prediction
        ).colour_scale

        assert scale.vmin == pytest.approx(0.0)
        assert scale.vmax == pytest.approx(0.9)
        assert scale.cmap == "magma"

    def test_smape(self) -> None:
        """SMAPE mode behaves like absolute mode."""
        ground_truth = np.array([0.2, 0.6])
        prediction = np.zeros(2)

        scale = DifferencePanel(DiffMode.SMAPE, ground_truth, prediction).colour_scale

        assert scale.vmin == pytest.approx(0.0)
        assert scale.cmap == "magma"

    def test_vmax_floor_avoids_zero_width_range(self) -> None:
        """An all-zero difference array still yields a strictly positive vmax."""
        ground_truth = np.zeros(3)
        prediction = np.zeros(3)

        scale = DifferencePanel(
            DiffMode.ABSOLUTE, ground_truth, prediction
        ).colour_scale

        assert scale.vmax == pytest.approx(1e-6)

    def test_invalid_mode_raises(self) -> None:
        """An unrecognised mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown difference mode: dummy"):
            _ = DifferencePanel(
                "dummy",  # type: ignore[arg-type]
                np.zeros(2),
                np.zeros(2),
            ).colour_scale

    def test_colours_standardised_difference_when_uncertainty_given(self) -> None:
        """colour_scale scales against the standardised difference, not the raw one."""
        ground_truth = np.array([[10.0]])
        prediction = np.array([[0.0]])
        uncertainty = np.array([[2.0]])

        scale = DifferencePanel(
            DiffMode.SIGNED, ground_truth, prediction, uncertainty
        ).colour_scale

        assert scale.vmax == pytest.approx(5.0)
