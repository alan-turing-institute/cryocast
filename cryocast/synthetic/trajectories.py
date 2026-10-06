"""Compose several independent moving-circle trajectories into one dataset.

A single long trajectory sliced into overlapping windows gives a model many
near-identical views of one physical instance, not genuine examples to generalise
from. Concatenating several independent trajectories -- each with a different start
position and velocity, separated by gap days so no window can bridge two of them --
lets whole trajectories be held out for validation/test, so the check actually
exercises whether a model learned the general translate-and-bounce rule rather than
memorising one specific path. The per-trajectory task stays exactly as simple.
"""

import datetime
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cryocast.types import ArrayTHW

from .shapes import (
    GrowShrinkCircleConfig,
    MovingCircleConfig,
    generate_grow_shrink_frames,
    generate_moving_circle_frames,
)

TrajectoryConfig = MovingCircleConfig | GrowShrinkCircleConfig


def generate_frames(config: TrajectoryConfig) -> ArrayTHW:
    """Generate the [T, H, W] frames for a single trajectory of either dynamics."""
    if isinstance(config, MovingCircleConfig):
        return generate_moving_circle_frames(config)
    if isinstance(config, GrowShrinkCircleConfig):
        return generate_grow_shrink_frames(config)
    msg = f"Unknown trajectory config type: {type(config).__name__}"
    raise TypeError(msg)


@dataclass(frozen=True)
class TrajectorySpan:
    """The calendar date range (inclusive) occupied by one trajectory."""

    start_date: datetime.datetime
    end_date: datetime.datetime


@dataclass(frozen=True)
class MultiTrajectoryDataset:
    frames: ArrayTHW
    dates: list[datetime.datetime]
    missing_dates: list[datetime.datetime]
    spans: list[TrajectorySpan]


def daily_dates(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> list[datetime.datetime]:
    """Return consecutive daily dates between `start_date` and `end_date`."""
    return [
        start_date + datetime.timedelta(days=step)
        for step in range((end_date - start_date).days + 1)
    ]


def default_trajectory_configs(
    *,
    height: int = 32,
    width: int = 32,
    n_timesteps: int = 20,
    radius: float | None = None,
    n_trajectories: int = 8,
) -> tuple[MovingCircleConfig, ...]:
    """`n_trajectories` bouncing-circle trajectories with distinct start/velocity.

    Start positions are spread evenly around a ring centred on the grid; velocity
    directions are roughly tangential, alternating a +45-degree offset every other
    trajectory, with two alternating speeds. Varying the (start position, velocity)
    combination -- while keeping the circle, grid, and update rule identical -- means
    a model must learn the general translate-and-bounce rule rather than memorise one
    specific path. `radius`/velocity magnitude scale with grid size (relative to a
    32x32 reference) unless `radius` is given explicitly, so the task looks the same
    at any resolution.
    """
    scale = height / 32
    radius = radius if radius is not None else 5.0 * scale
    centre_row, centre_col = height / 2, width / 2
    start_radius = min(height, width) / 2 - radius - 2 * scale
    speeds = (2.0 * scale, 3.0 * scale)

    trajectories = []
    for i in range(n_trajectories):
        angle = 2 * np.pi * i / n_trajectories
        start_position = (
            centre_row + start_radius * np.sin(angle),
            centre_col + start_radius * np.cos(angle),
        )
        speed = speeds[i % len(speeds)]
        velocity_angle = angle + np.pi / 2 + (np.pi / 4 if i % 2 else 0.0)
        velocity = (speed * np.sin(velocity_angle), speed * np.cos(velocity_angle))
        trajectories.append(
            MovingCircleConfig(
                height=height,
                width=width,
                n_timesteps=n_timesteps,
                radius=radius,
                start_position=start_position,
                velocity=velocity,
            )
        )
    return tuple(trajectories)


def default_grow_shrink_configs(
    *,
    height: int = 32,
    width: int = 32,
    n_timesteps: int = 20,
    n_trajectories: int = 8,
) -> tuple[GrowShrinkCircleConfig, ...]:
    """`n_trajectories` grow-shrink blobs with distinct centre, rate, and phase.

    Each blob stays put but pulses in size via morphological dilation/erosion. Centres
    are spread evenly around a ring (kept clear of the edges by the largest grown
    radius), and the growth amplitude, cycle period, and starting phase all vary per
    trajectory, so a model must learn the general grow-then-shrink rule rather than
    memorise one pulse. `base_radius`/growth scale with grid size (relative to a 32x32
    reference) so the task looks the same at any resolution.
    """
    scale = height / 32
    base_radius = 8.0 * scale
    # A dilation/erosion iteration changes the blob radius by roughly one pixel, so
    # expressing the amplitudes as fractions of the seed radius keeps the deepest
    # erosion safely inside the seed (never vanishes) and the largest dilation inside
    # the frame at any resolution. growth peaks at ~0.75R, shrinkage at ~0.5R.
    max_growth = round(0.75 * base_radius)
    centre_row, centre_col = height / 2, width / 2
    start_radius = max(
        0.0, min(height, width) / 2 - base_radius - max_growth - 2 * scale
    )

    configs = []
    for i in range(n_trajectories):
        angle = 2 * np.pi * i / n_trajectories
        center = (
            centre_row + start_radius * np.sin(angle),
            centre_col + start_radius * np.cos(angle),
        )
        # Vary the growth amplitude a little per trajectory (0.55R-0.75R).
        growth = round((0.55 + 0.10 * (i % 3)) * base_radius)
        # Erosion stays at most half the seed radius, so a shrinking blob never vanishes.
        shrinkage = min(growth, round(0.5 * base_radius))
        period = n_timesteps * (0.7 + 0.3 * (i % 2))
        phase = i / n_trajectories
        configs.append(
            GrowShrinkCircleConfig(
                height=height,
                width=width,
                n_timesteps=n_timesteps,
                base_radius=base_radius,
                growth=growth,
                shrinkage=shrinkage,
                period=period,
                phase=phase,
                center=center,
            )
        )
    return tuple(configs)


def generate_default_dataset(
    *,
    dynamics: str,
    grid_size: int,
    start_dates: Sequence[datetime.datetime],
) -> MultiTrajectoryDataset:
    """Generate the standard multi-trajectory dataset for the requested dynamics."""
    trajectories: tuple[TrajectoryConfig, ...]
    if dynamics == "moving":
        trajectories = default_trajectory_configs(
            height=grid_size, width=grid_size, n_trajectories=len(start_dates)
        )
    elif dynamics == "grow-shrink":
        trajectories = default_grow_shrink_configs(
            height=grid_size, width=grid_size, n_trajectories=len(start_dates)
        )
    else:
        msg = f"Unknown dynamics {dynamics!r}; expected 'moving' or 'grow-shrink'."
        raise ValueError(msg)
    return generate_multi_trajectory_dataset(trajectories, start_dates)


def generate_multi_trajectory_dataset(
    trajectories: Sequence[TrajectoryConfig],
    start_dates: Sequence[datetime.datetime],
) -> MultiTrajectoryDataset:
    """Concatenate independent trajectories end-to-end, with gap days between them."""
    frame_chunks: list[ArrayTHW] = []
    spans: list[TrajectorySpan] = []
    missing_dates: list[datetime.datetime] = []

    last_end_date: datetime.datetime | None = None
    for trajectory, start_date in zip(trajectories, start_dates, strict=True):
        if last_end_date is not None:
            if start_date <= last_end_date:
                msg = (
                    f"start_dates must be strictly increasing, but {start_date} "
                    f"follows {last_end_date}"
                )
                raise ValueError(msg)
            gap_dates = daily_dates(
                last_end_date + datetime.timedelta(days=1),
                start_date - datetime.timedelta(days=1),
            )
            missing_dates.extend(gap_dates)
            frame_chunks.append(
                np.zeros(
                    (len(gap_dates), trajectory.height, trajectory.width),
                    dtype=np.float32,
                )
            )

        frame_chunks.append(generate_frames(trajectory))
        end_date = start_date + datetime.timedelta(days=trajectory.n_timesteps - 1)
        spans.append(
            TrajectorySpan(
                start_date=start_date,
                end_date=end_date,
            )
        )
        last_end_date = spans[-1].end_date

    return MultiTrajectoryDataset(
        frames=np.concatenate(frame_chunks, axis=0),
        dates=daily_dates(
            min(span.start_date for span in spans),
            max(span.end_date for span in spans),
        ),
        missing_dates=missing_dates,
        spans=spans,
    )
