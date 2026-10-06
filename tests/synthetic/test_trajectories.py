"""Tests for composing single trajectories into a multi-trajectory dataset."""

import datetime

import numpy as np
import pytest

from cryocast.synthetic.shapes import GrowShrinkCircleConfig, MovingCircleConfig
from cryocast.synthetic.trajectories import (
    default_grow_shrink_configs,
    default_trajectory_configs,
    generate_frames,
    generate_multi_trajectory_dataset,
)


class TestDefaultMovingConfigs:
    """`default_trajectory_configs` yields distinct, resolution-scaled trajectories."""

    def test_count_and_type(self) -> None:
        """It returns exactly `n_trajectories` moving-circle configs."""
        configs = default_trajectory_configs(n_trajectories=6)
        assert len(configs) == 6
        assert all(isinstance(c, MovingCircleConfig) for c in configs)

    def test_start_positions_and_velocities_differ(self) -> None:
        """Distinct start/velocity combinations force the model to generalise."""
        configs = default_trajectory_configs(n_trajectories=5)
        starts = {c.start_position for c in configs}
        velocities = {c.velocity for c in configs}
        assert len(starts) == len(configs)
        assert len(velocities) > 1

    def test_radius_scales_with_grid(self) -> None:
        """Radius scales with grid size relative to the 32x32 reference."""
        small = default_trajectory_configs(height=32, width=32)[0]
        large = default_trajectory_configs(height=64, width=64)[0]
        assert large.radius == pytest.approx(2 * small.radius)


class TestDefaultGrowShrinkConfigs:
    """`default_grow_shrink_configs` yields distinct, non-vanishing pulsing blobs."""

    def test_count_and_type(self) -> None:
        """It returns exactly `n_trajectories` grow-shrink configs."""
        configs = default_grow_shrink_configs(n_trajectories=7)
        assert len(configs) == 7
        assert all(isinstance(c, GrowShrinkCircleConfig) for c in configs)

    def test_shrinkage_stays_below_seed_radius(self) -> None:
        """Erosion is capped so a shrinking blob can never disappear."""
        for config in default_grow_shrink_configs(n_trajectories=8):
            assert config.shrinkage < config.base_radius

    def test_phases_differ(self) -> None:
        """Different starting phases give the model varied grow/shrink states."""
        phases = {c.phase for c in default_grow_shrink_configs(n_trajectories=6)}
        assert len(phases) > 1

    def test_generated_blobs_never_vanish(self) -> None:
        """Every frame of every default trajectory retains some foreground."""
        for config in default_grow_shrink_configs(
            height=48, width=48, n_timesteps=20, n_trajectories=5
        ):
            areas = generate_frames(config).sum(axis=(1, 2))
            assert areas.min() > 0


class TestGenerateFrames:
    """`generate_frames` dispatches on the config type."""

    def test_dispatches_moving(self) -> None:
        """A MovingCircleConfig produces a constant-area sequence."""
        frames = generate_frames(MovingCircleConfig(n_timesteps=5))
        areas = frames.sum(axis=(1, 2))
        assert np.all(areas == areas[0])

    def test_dispatches_grow_shrink(self) -> None:
        """A GrowShrinkCircleConfig produces a varying-area sequence."""
        frames = generate_frames(GrowShrinkCircleConfig(n_timesteps=20, period=20.0))
        areas = frames.sum(axis=(1, 2))
        assert areas.max() > areas.min()

    def test_unknown_config_raises(self) -> None:
        """An unrecognised config type is rejected."""
        with pytest.raises(TypeError, match="Unknown trajectory config type"):
            generate_frames(object())  # type: ignore[arg-type]


class TestMultiTrajectoryDataset:
    """Independent trajectories are stitched together with gap days between them."""

    def test_total_length_and_spans(self) -> None:
        """Total days = sum of trajectory lengths + gap days between each pair."""
        gap = 2
        n_timesteps = 10
        n_trajectories = 4
        configs = default_trajectory_configs(
            n_trajectories=n_trajectories, n_timesteps=n_timesteps
        )
        start_dates = [
            datetime.datetime(2021, 1, 1)
            + datetime.timedelta(days=idx * (n_timesteps + gap))
            for idx in range(n_trajectories)
        ]
        dataset = generate_multi_trajectory_dataset(configs, start_dates)
        expected_days = n_trajectories * n_timesteps + gap * (n_trajectories - 1)
        assert dataset.frames.shape[0] == expected_days
        assert len(dataset.dates) == expected_days
        assert len(dataset.spans) == n_trajectories
        assert len(dataset.missing_dates) == gap * (n_trajectories - 1)

    def test_missing_dates_are_the_gaps(self) -> None:
        """Gap days are marked missing and fall between (not inside) trajectories."""
        gap = 2
        n_timesteps = 8
        n_trajectories = 3
        configs = default_trajectory_configs(
            n_trajectories=n_trajectories, n_timesteps=n_timesteps
        )
        start_dates = [
            datetime.datetime(2021, 1, 1)
            + datetime.timedelta(days=idx * (n_timesteps + gap))
            for idx in range(n_trajectories)
        ]
        dataset = generate_multi_trajectory_dataset(configs, start_dates)
        span_days = {
            span.start_date + datetime.timedelta(days=offset)
            for span in dataset.spans
            for offset in range(n_timesteps)
        }
        assert not span_days.intersection(dataset.missing_dates)

    def test_works_for_grow_shrink(self) -> None:
        """The same stitching works for grow-shrink trajectories."""
        gap = 2
        n_timesteps = 8
        n_trajectories = 3
        configs = default_grow_shrink_configs(
            height=48, width=48, n_trajectories=n_trajectories, n_timesteps=n_timesteps
        )
        start_dates = [
            datetime.datetime(2021, 1, 1)
            + datetime.timedelta(days=idx * (n_timesteps + gap))
            for idx in range(n_trajectories)
        ]
        dataset = generate_multi_trajectory_dataset(configs, start_dates)
        assert dataset.frames.shape == (
            n_trajectories * n_timesteps + (n_trajectories - 1) * gap,
            48,
            48,
        )
        assert len(dataset.spans) == n_trajectories
