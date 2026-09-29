from functools import cached_property

import numpy as np

from icenet_mp.exceptions import InvalidArrayError
from icenet_mp.types import ArrayHW, ArrayTHW, ColourScale, DiffMode
from icenet_mp.utils import safe_nanmax, safe_nanmin


class DifferencePanel:
    """Compute and style the difference between a ground-truth/prediction pair."""

    VALID_NDIMS = (2, 3)

    def __init__(
        self,
        diff_mode: DiffMode,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        uncertainty: ArrayHW | ArrayTHW | None = None,
    ) -> None:
        """Initialise a DifferencePanel for one ground-truth/prediction pair.

        Args:
            diff_mode: Which method to use to compute differences.
            ground_truth: The ground truth array. 2D `[H, W]` (image) or 3D
                `[T, H, W]` (video).
            prediction: Predicted values, matching the shape of `ground_truth`.
            uncertainty: Optional standard uncertainty for the observations, matching
                the shape of `ground_truth`. Required to access `standardised_difference`.

        Raises:
            InvalidArrayError: If `ground_truth` and `prediction` don't have matching
                shapes, or (when `uncertainty` is given) if any array isn't 2D or 3D,
                or the three arrays' shapes don't all match.

        """
        if ground_truth.shape != prediction.shape:
            msg = (
                "Ground truth and prediction must have matching shapes; "
                f"got {ground_truth.shape} and {prediction.shape}."
            )
            raise InvalidArrayError(msg)

        if uncertainty is not None:
            arrays = (ground_truth, prediction, uncertainty)
            if any(array.ndim not in DifferencePanel.VALID_NDIMS for array in arrays):
                shapes = tuple(array.shape for array in arrays)
                msg = (
                    "Expected 2D [H, W] or 3D [T, H, W] ground truth, prediction and "
                    f"uncertainty arrays, got {shapes}."
                )
                raise InvalidArrayError(msg)
            if not (ground_truth.shape == prediction.shape == uncertainty.shape):
                msg = (
                    "Ground truth, prediction and uncertainty must have matching "
                    f"shapes; got {ground_truth.shape}, {prediction.shape} and "
                    f"{uncertainty.shape}."
                )
                raise InvalidArrayError(msg)

        self.diff_mode = diff_mode
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.uncertainty = uncertainty

    @cached_property
    def _diff_mode_difference(self) -> np.ndarray:
        """Difference between ground truth and prediction per the configured diff_mode.

        Raises:
            ValueError: If the difference mode is invalid.

        """
        if self.diff_mode == DiffMode.SIGNED:
            return self.ground_truth - self.prediction
        if self.diff_mode == DiffMode.ABSOLUTE:
            return np.abs(self.ground_truth - self.prediction)
        if self.diff_mode == DiffMode.SMAPE:
            denom = np.clip(
                (np.abs(self.ground_truth) + np.abs(self.prediction)) / 2.0,
                1e-6,
                None,
            )
            return np.abs(self.prediction - self.ground_truth) / denom
        msg = f"Unknown difference mode: {self.diff_mode}"
        raise ValueError(msg)

    @cached_property
    def standardised_difference(self) -> np.ndarray:
        """Return prediction error in units of observational standard uncertainty.

        Divides `_diff_mode_difference` (whichever convention `diff_mode` selects)
        by `uncertainty`, so this is only a signed z-score when `diff_mode` is
        `DiffMode.SIGNED`; for `ABSOLUTE`/`SMAPE` it is non-negative. Locations
        with non-finite or non-positive uncertainty are returned as NaN because a
        z value is undefined there.

        Returns:
            Standardised difference array, matching `ground_truth`'s shape.

        Raises:
            ValueError: If `uncertainty` wasn't provided at initialisation time.

        """
        if self.uncertainty is None:
            msg = "standardised_difference requires `uncertainty` to be provided."
            raise ValueError(msg)

        result = np.full(self.ground_truth.shape, np.nan, dtype=float)
        valid = np.isfinite(self.uncertainty) & (self.uncertainty > 0)
        np.divide(
            self._diff_mode_difference,
            self.uncertainty,
            out=result,
            where=valid,
        )
        return result

    @cached_property
    def difference(self) -> np.ndarray:
        """The difference to display: standardised against uncertainty when given.

        Falls back to the raw, diff_mode-based difference when no `uncertainty`
        was provided at construction.

        Returns:
            Difference array, matching `ground_truth`'s shape.

        """
        if self.uncertainty is not None:
            return self.standardised_difference
        return self._diff_mode_difference

    @cached_property
    def colour_scale(self) -> ColourScale:
        """Construct a ColourScale for visualising `difference`.

        This generates a ColourScale object that contains the appropriate
        colour limits and colourmap for visualising differences between
        datasets. The behaviour of the colour mapping depends on the
        difference mode bound at construction.

        Returns:
            ColourScale: Colour limits and colourmap for the difference panel.

        """
        sample = self.difference

        if self.diff_mode == DiffMode.SIGNED:
            # Force symmetric limits around zero so 0 is the literal midpoint
            vmin_data = safe_nanmin(sample, default=-1.0)
            vmax_data = safe_nanmax(sample, default=1.0)
            max_abs = max(1.0, abs(vmin_data), abs(vmax_data))

            return ColourScale(
                vmin=-max_abs,
                vmax=max_abs,
                cmap="RdBu_r",
            )

        # ABSOLUTE or SMAPE: positive-only scale (`difference` above already
        # validates diff_mode, so no other mode can reach this point)
        vmax = max(1e-6, safe_nanmax(sample, default=0.0))
        return ColourScale(
            vmin=0.0,
            vmax=vmax,
            cmap="magma",
        )
