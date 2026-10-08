"""Tests for profile-aware sparse Argo benchmark splitting."""

import numpy as np

from cryocast.research.argo_profile_split import (
    profile_group_count,
    profiles_overlap,
    split_observations,
)
from cryocast.research.argo_sparse import SparseObservations


def _profile_observations() -> SparseObservations:
    return SparseObservations(
        latitudes=np.asarray([70.0, 70.0, 72.0, 72.0, 74.0, 74.0]),
        longitudes=np.asarray([10.0, 10.0, 20.0, 20.0, 30.0, 30.0]),
        measurements=np.asarray(
            [
                [1.0, 30.0],
                [2.0, 31.0],
                [3.0, 32.0],
                [4.0, 33.0],
                [5.0, 34.0],
                [6.0, 35.0],
            ]
        ),
        variable_names=("TEMP", "PSAL"),
        metadata={
            "PLATFORM_NUMBER": ("A", "A", "B", "B", "C", "C"),
            "CYCLE_NUMBER": (1, 1, 2, 2, 3, 3),
        },
    )


def test_profile_split_keeps_complete_profiles_on_one_side() -> None:
    """Keep all pressure levels from a profile on the same split side."""
    observations = _profile_observations()

    split = split_observations(observations, holdout_fraction=0.34, seed=477)

    assert profiles_overlap(split.observed, split.held_out) == set()
    assert split.observed.count + split.held_out.count == observations.count
    assert profile_group_count(observations) == 3


def test_profile_split_is_reproducible() -> None:
    """Use the fixed seed to reproduce the same complete-profile split."""
    observations = _profile_observations()

    first = split_observations(observations, holdout_fraction=0.34, seed=477)
    second = split_observations(observations, holdout_fraction=0.34, seed=477)

    np.testing.assert_array_equal(
        first.observed.measurements, second.observed.measurements
    )
    np.testing.assert_array_equal(
        first.held_out.measurements, second.held_out.measurements
    )


def test_profile_split_falls_back_when_profile_ids_are_unavailable() -> None:
    """Retain the deterministic row split for fixtures without profile metadata."""
    observations = _profile_observations()
    without_metadata = SparseObservations(
        latitudes=observations.latitudes,
        longitudes=observations.longitudes,
        measurements=observations.measurements,
        variable_names=observations.variable_names,
    )

    split = split_observations(without_metadata, holdout_fraction=0.34, seed=477)

    assert split.observed.count > 0
    assert split.held_out.count > 0
    assert profile_group_count(without_metadata) is None
    assert profiles_overlap(split.observed, split.held_out) is None
