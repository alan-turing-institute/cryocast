"""Tests for the synthetic shape generators (moving circle and grow-shrink blob)."""

import numpy as np

from cryocast.synthetic.shapes import (
    GrowShrinkCircleConfig,
    MovingCircleConfig,
    _apply_morphology,
    generate_grow_shrink_frames,
    generate_moving_circle_frames,
)


class TestMovingCircle:
    """The translating circle is a rigid body: constant area, deterministic path."""

    def test_shape_and_dtype(self) -> None:
        """Frames have the configured [T, H, W] shape and float32 dtype."""
        frames = generate_moving_circle_frames(
            MovingCircleConfig(height=20, width=24, n_timesteps=7)
        )
        assert frames.shape == (7, 20, 24)
        assert frames.dtype == np.float32

    def test_values_are_binary(self) -> None:
        """Every pixel is either the foreground or the background value."""
        frames = generate_moving_circle_frames(MovingCircleConfig(n_timesteps=5))
        assert set(np.unique(frames)).issubset({0.0, 1.0})

    def test_area_is_constant(self) -> None:
        """A rigid circle keeps the same foreground area at every timestep."""
        frames = generate_moving_circle_frames(MovingCircleConfig(n_timesteps=10))
        areas = frames.sum(axis=(1, 2))
        assert np.allclose(areas, areas[0])

    def test_circle_stays_inside_the_grid(self) -> None:
        """Bouncing off the edges keeps the whole circle within the frame."""
        frames = generate_moving_circle_frames(
            MovingCircleConfig(n_timesteps=40, velocity=(3.0, 2.0))
        )
        # The circle never wraps: if it left the grid its area would drop below the
        # full disc area on the frame where part of it is clipped.
        areas = frames.sum(axis=(1, 2))
        assert np.all(areas == areas[0])

    def test_deterministic(self) -> None:
        """The same config always produces identical frames."""
        config = MovingCircleConfig(n_timesteps=8, velocity=(1.5, 2.5))
        assert np.array_equal(
            generate_moving_circle_frames(config),
            generate_moving_circle_frames(config),
        )


class TestGrowShrink:
    """The grow-shrink blob stays put but pulses in size via morphology."""

    def test_shape_and_dtype(self) -> None:
        """Frames have the configured [T, H, W] shape and float32 dtype."""
        frames = generate_grow_shrink_frames(
            GrowShrinkCircleConfig(height=40, width=40, n_timesteps=12)
        )
        assert frames.shape == (12, 40, 40)
        assert frames.dtype == np.float32

    def test_area_pulses(self) -> None:
        """The foreground area both grows above and shrinks below the seed area."""
        frames = generate_grow_shrink_frames(
            GrowShrinkCircleConfig(n_timesteps=20, period=20.0, growth=4, shrinkage=4)
        )
        areas = frames.sum(axis=(1, 2))
        seed_area = areas[0]  # t=0 -> sin(0)=0 -> no morphology applied
        assert areas.max() > seed_area
        assert areas.min() < seed_area

    def test_grows_then_shrinks_with_the_cycle(self) -> None:
        """A phase-0 sine peaks at a quarter period and troughs at three-quarters."""
        period = 20
        frames = generate_grow_shrink_frames(
            GrowShrinkCircleConfig(n_timesteps=period, period=float(period))
        )
        areas = frames.sum(axis=(1, 2))
        assert areas[period // 4] > areas[0]
        assert areas[3 * period // 4] < areas[0]

    def test_never_vanishes(self) -> None:
        """With shrinkage bounded below the seed radius the blob is never empty."""
        frames = generate_grow_shrink_frames(
            GrowShrinkCircleConfig(
                n_timesteps=30, period=15.0, base_radius=6.0, growth=3, shrinkage=3
            )
        )
        assert frames.sum(axis=(1, 2)).min() > 0

    def test_center_of_mass_is_stationary(self) -> None:
        """Growing and shrinking in place does not move the blob's centre."""
        frames = generate_grow_shrink_frames(
            GrowShrinkCircleConfig(height=48, width=48, n_timesteps=20, period=20.0)
        )
        centres = []
        for frame in frames:
            rows, cols = np.nonzero(frame)
            centres.append((rows.mean(), cols.mean()))
        centres_arr = np.array(centres)
        assert np.allclose(centres_arr, centres_arr[0], atol=1.0)

    def test_deterministic(self) -> None:
        """The same config always produces identical frames."""
        config = GrowShrinkCircleConfig(n_timesteps=10, period=8.0, phase=0.25)
        assert np.array_equal(
            generate_grow_shrink_frames(config),
            generate_grow_shrink_frames(config),
        )


class TestMorphology:
    """The pure-numpy dilation/erosion underpinning the grow-shrink dynamics."""

    def _disc(self, size: int = 21, radius: float = 5.0) -> np.ndarray:
        rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        centre = size // 2
        return (rows - centre) ** 2 + (cols - centre) ** 2 <= radius**2

    def test_zero_iterations_is_identity(self) -> None:
        """Applying zero morphology iterations returns the seed unchanged."""
        seed = self._disc()
        assert np.array_equal(_apply_morphology(seed, 0), seed)

    def test_dilation_grows_area(self) -> None:
        """Positive iterations dilate: the area strictly increases."""
        seed = self._disc()
        assert _apply_morphology(seed, 3).sum() > seed.sum()

    def test_erosion_shrinks_area(self) -> None:
        """Negative iterations erode: the area strictly decreases."""
        seed = self._disc()
        assert _apply_morphology(seed, -2).sum() < seed.sum()

    def test_erosion_against_the_border(self) -> None:
        """A blob touching the grid edge erodes there (out-of-bounds is background)."""
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, :] = True  # a full row on the top edge
        eroded = _apply_morphology(mask, -1)
        assert eroded.sum() < mask.sum()
