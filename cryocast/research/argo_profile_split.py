"""Profile-aware splits for the sparse Argo reconstruction benchmark.

Argo rows from one float profile are strongly correlated across pressure levels. A
row-wise held-out split can therefore put levels from the same profile on both sides
of the reconstruction task. This module keeps complete ``PLATFORM_NUMBER`` /
``CYCLE_NUMBER`` groups together when those raw metadata fields are available.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from .argo_sparse import (
    ObservationSplit,
    SparseObservations,
)
from .argo_sparse import (
    split_observations as _row_split,
)

_PROFILE_METADATA_FIELDS = ("PLATFORM_NUMBER", "CYCLE_NUMBER")
_MIN_PROFILE_GROUPS = 2


def _is_missing(value: object) -> bool:
    """Return whether an Argo profile identifier is missing."""
    if value is None:
        return True
    if isinstance(value, Real):
        return math.isnan(float(value))
    return False


def profile_group_keys(
    observations: SparseObservations,
) -> tuple[tuple[str, str], ...] | None:
    """Return per-row profile keys, or ``None`` when reliable IDs are unavailable."""
    metadata = observations.metadata
    if metadata is None or any(
        field not in metadata for field in _PROFILE_METADATA_FIELDS
    ):
        return None

    platform_values = metadata[_PROFILE_METADATA_FIELDS[0]]
    cycle_values = metadata[_PROFILE_METADATA_FIELDS[1]]
    keys: list[tuple[str, str]] = []
    for platform, cycle in zip(platform_values, cycle_values, strict=True):
        if _is_missing(platform) or _is_missing(cycle):
            return None
        keys.append((str(platform), str(cycle)))
    return tuple(keys)


def profile_group_count(observations: SparseObservations) -> int | None:
    """Return the number of complete profiles represented by a sparse sample."""
    keys = profile_group_keys(observations)
    return None if keys is None else len(set(keys))


def split_observations(
    observations: SparseObservations,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> ObservationSplit:
    """Split complete Argo profiles when possible, otherwise use the row baseline.

    The fallback keeps the generic research scaffold usable for synthetic fixtures and
    raw sources that do not expose profile identifiers. Real Argo benchmark runs retain
    ``PLATFORM_NUMBER`` and ``CYCLE_NUMBER`` explicitly so evaluation does not leak
    pressure levels from one profile across observed and held-out sets.
    """
    keys = profile_group_keys(observations)
    if keys is None or len(set(keys)) < _MIN_PROFILE_GROUPS:
        return _row_split(
            observations,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
    if not 0.0 < holdout_fraction < 1.0:
        msg = "holdout_fraction must be between 0 and 1."
        raise ValueError(msg)

    groups = tuple(dict.fromkeys(keys))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    held_out_group_count = round(len(groups) * holdout_fraction)
    held_out_group_count = max(1, min(len(groups) - 1, held_out_group_count))
    held_out_groups = {groups[int(index)] for index in order[:held_out_group_count]}

    held_out_indices = np.asarray(
        [index for index, key in enumerate(keys) if key in held_out_groups],
        dtype=np.int64,
    )
    observed_indices = np.asarray(
        [index for index, key in enumerate(keys) if key not in held_out_groups],
        dtype=np.int64,
    )
    return ObservationSplit(
        observed=observations.take(observed_indices),
        held_out=observations.take(held_out_indices),
    )


def profiles_overlap(
    first: SparseObservations,
    second: SparseObservations,
) -> set[tuple[str, str]] | None:
    """Return profile keys shared by two subsets, or ``None`` without profile IDs."""
    first_keys = profile_group_keys(first)
    second_keys = profile_group_keys(second)
    if first_keys is None or second_keys is None:
        return None
    return set(first_keys) & set(second_keys)
