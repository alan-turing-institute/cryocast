import logging
from collections.abc import Sequence
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import ClassVar, TypeVar, cast

import numpy as np
from anemoi.datasets import open_dataset
from anemoi.datasets.usage.dataset import Dataset as AnemoiDataset
from torch.utils.data import Dataset

from icenet_mp.types import ArrayCHW, ArrayTCHW, DataSpace, Hemisphere

ArrayType = TypeVar("ArrayType", ArrayCHW, ArrayTCHW)

logger = logging.getLogger(__name__)


class SingleDataset(Dataset):
    """A dataset containing one or more timeslices of data from a single source."""

    # Cache anemoi reads at class level to reduce disk I/O
    anemoi_cache: ClassVar[dict[tuple[Path, ...], AnemoiDataset]] = {}

    def __init__(
        self,
        name: str,
        input_files: Sequence[Path],
        *,
        date_ranges: Sequence[dict[str, str | None]] = [{"start": None, "end": None}],
        group: str | None = None,
        normalise: bool = True,
        variables: Sequence[str] = (),
    ) -> None:
        """A dataset for use by IceNet-MP.

        The underlying Anemoi dataset has shape [T; C; ensembles; position].
        We reshape this to CHW before returning.

        Args:
            name: The name of the dataset, used to identify it in plots and logs.
            input_files: The paths to the Anemoi dataset files.
            date_ranges: The ranges of dates to include in the dataset. Each range is a
                dict with "start" and "end" keys, which can be None to indicate
                open-ended ranges.
            group: The group name for the dataset, used to identify it in plots and
                logs. If None, the group name defaults to the dataset name.
            normalise: Whether to normalise the data to [0, 1] for each channel.
            variables: The names of the variables to include in the dataset. If empty,
                all variables are included.

        """
        super().__init__()
        self.hemisphere: Hemisphere = (
            Hemisphere.NORTH
            if any("north" in str(input_file).lower() for input_file in input_files)
            else Hemisphere.SOUTH
        )
        self.name = name
        self.group = name if group is None else group
        self._date_ranges = self.normalise_date_ranges(date_ranges)
        self._input_files = tuple(sorted(input_files))
        self._normalise = normalise
        self._norm_offset: np.ndarray | None = None
        self._norm_scale: np.ndarray | None = None
        self._variables = set(variables)

    @classmethod
    def load_dataset(cls, input_files: tuple[Path, ...]) -> AnemoiDataset:
        if input_files not in cls.anemoi_cache:
            cls.anemoi_cache[input_files] = open_dataset(input_files)
        return cls.anemoi_cache[input_files]

    @cached_property
    def _date2idx(self) -> dict[np.datetime64, int]:
        """Map date to global index in the dataset."""
        return {date: idx for idx, date in enumerate(self.dates)}

    @cached_property
    def _idx2anemoi(self) -> dict[int, tuple[int, int]]:
        """Map global index to a location in an Anemoi dataset."""
        idx2anemoi = {}
        for idx_ds, dataset in enumerate(self.dataslices):
            for idx_date, date in enumerate(dataset.dates):
                idx_global = self._date2idx.get(self.normalise_date(date), None)
                if idx_global is not None:
                    idx2anemoi[idx_global] = (idx_ds, idx_date)
        return idx2anemoi

    @staticmethod
    def normalise_date(np_datetime: np.datetime64) -> np.datetime64:
        """Normalise a datetime to noon."""
        dt: datetime = np_datetime.astype("datetime64[ms]").astype(datetime)
        return np.datetime64(dt.replace(hour=12, minute=0, second=0, microsecond=0))

    @staticmethod
    def normalise_date_ranges(
        date_ranges: Sequence[dict[str, str | None]],
    ) -> list[dict[str, str | None]]:
        """Sort the ranges, then merge overlapping or touching ones into single spans.

        Ranges that overlap/touch must merge into one Anemoi subset, or data fetching step comes back
        short and fails to reshape. Two ranges merge when the later one starts on/before the day
        after the earlier one finishes; the merged span is the union of all ranges.

        A "None" bound is open, an open-ended previous range always overlap with the later ranges, an
        open-started later range always overlap with previous ranges.

        Each merge is logged so a user notices that their config ranges were altered and verifies.
        """
        ranges = sorted(
            date_ranges, key=lambda dr: "" if dr["start"] is None else dr["start"]
        )
        if len(ranges) <= 1:
            return [dict(date_range) for date_range in ranges]

        # Assume data always has a daily frequency; for future updates: read from source metadata
        frequency = np.timedelta64(1, "D")

        def _ranges_overlap_or_touch(
            prev: dict[str, str | None], nxt: dict[str, str | None]
        ) -> bool:
            """Whether nxt overlaps or is consecutive with prev, so they should merge.

            Ranges are sorted by start. They merge when nxt starts on or before the
            day after prev ends. An open end on prev (None = after any date) covers
            everything later, and an open start on nxt (None = before any date) sits
            inside prev, so in either case they always merge.
            """
            if prev["end"] is None or nxt["start"] is None:
                return True
            return bool(
                np.datetime64(nxt["start"]) <= np.datetime64(prev["end"]) + frequency
            )

        def _later_end(a: str | None, b: str | None) -> str | None:
            """Return the later of two end dates; None (open) is after any date."""
            if a is None or b is None:
                return None
            return a if np.datetime64(a) >= np.datetime64(b) else b

        merged: list[dict[str, str | None]] = []
        for date_range in ranges:
            if merged and _ranges_overlap_or_touch(merged[-1], date_range):
                prev = merged.pop()
                fused: dict[str, str | None] = {
                    "start": prev["start"],
                    "end": _later_end(prev["end"], date_range["end"]),
                }
                logger.warning(
                    "Merged overlapping or touching date ranges %s and %s into %s. "
                    "Please check config file range setting if unintended",
                    prev,
                    date_range,
                    fused,
                )
                merged.append(fused)
            else:
                merged.append(dict(date_range))
        return merged

    @cached_property
    def dataslices(self) -> list[AnemoiDataset]:
        """Get all slices of contiguous dates from the underlying Anemoi dataset."""
        return [
            self.load_dataset(self._input_files)._subset(
                name=self.name,
                start=date_range["start"],
                end=date_range["end"],
                **({"select": self._variables} if self._variables else {}),
            )
            for date_range in self._date_ranges
        ]

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Return all available dates in the dataset, removing any that are missing."""
        return sorted(
            {
                self.normalise_date(date)
                for ds in self.dataslices
                for date in np.delete(ds.dates, list(ds.missing))
            }
        )

    @cached_property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.dates[-1]

    @cached_property
    def frequency(self) -> np.timedelta64:
        """Return the frequency of the dataset.

        Use frequency of the underlying Anemoi dataset, which is read from metadata,
        rather than the frequency of the dataslices which are recalculated on-the-fly
        by Anemoi through diff-ing the first two dates in the dataslice, which is
        incorrect for datasets with missing dates.
        """
        return np.timedelta64(self.load_dataset(self._input_files).frequency)

    @cached_property
    def latitudes(self) -> list[float]:
        """Return the latitudes of the dataset."""
        return self.dataslices[0].latitudes.tolist()

    @cached_property
    def longitudes(self) -> list[float]:
        """Return the longitudes of the dataset."""
        return self.dataslices[0].longitudes.tolist()

    @cached_property
    def space(self) -> DataSpace:
        """Return the data space for this dataset."""
        return DataSpace(
            channels=self.dataslices[0].shape[1],
            name=self.name,
            shape=self.dataslices[0].field_shape,
        )

    @cached_property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.dates[0]

    @cached_property
    def variable_names(self) -> list[str]:
        """Return the variable names for this dataset."""
        return self.dataslices[0].variables

    @cached_property
    def statistics(self) -> dict[str, np.ndarray]:
        """Return per-channel statistics from the underlying dataset.

        Keys are 'mean', 'stdev', 'maximum', 'minimum', each with shape [C].
        """
        return self.dataslices[0].statistics

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        return len(self.dates)

    def __getitem__(self, idx: int) -> ArrayCHW:
        """Return normalised data for a single timestep in [C, H, W] format."""
        return self.normalise(self._chw_raw(idx))

    def _chw_raw(self, idx: int) -> ArrayCHW:
        """Return unnormalised data for a single timestep in [C, H, W] format."""
        try:
            idx_ds, idx_date = self._idx2anemoi[idx]
            return self.dataslices[idx_ds][idx_date].reshape(self.space.chw)
        except KeyError as exc:
            msg = f"Index {idx} out of range for dataset of length {len(self)}."
            raise IndexError(msg) from exc

    def get_tchw(self, dates: Sequence[np.datetime64]) -> ArrayTCHW:
        """Return normalised data for an arbitrary sequence of timesteps in [T, C, H, W] format."""
        return self.normalise(
            np.stack(
                [self._chw_raw(self.to_index(target_date)) for target_date in dates],
                axis=0,
            )
        )

    def get_tchw_slice(
        self, start_date: np.datetime64, n_steps: int, *, check: bool = True
    ) -> ArrayTCHW:
        """Return the data for consecutive timesteps in [T, C, H, W] format.

        Since contiguous dates must be in a single dataslice, we simply identify which
        one this is and read from it.

        If `check` is True then we check that we're not crossing the boundary between
        dataslices, adding a small amount of overhead.

        Args:
            start_date: The date of the first timestep to return.
            n_steps: The number of consecutive timesteps to return.
            check: Whether to check that the requested slice is valid. If False, this
                   method may return meaningless or incorrect data if the requested
                   slice is invalid

        """
        start_date = self.normalise_date(start_date)
        try:
            idx_global_start = self._date2idx[start_date]
            idx_ds_start, idx_date_start = self._idx2anemoi[idx_global_start]
            if check:
                idx_global_end = idx_global_start + n_steps - 1
                idx_ds_end, _ = self._idx2anemoi[idx_global_end]
                if idx_ds_start != idx_ds_end:
                    msg = (
                        f"Requested slice of {n_steps} steps following {start_date} "
                        f"crosses the boundary between dataslices {idx_ds_start} and "
                        f"{idx_ds_end}."
                    )
                    raise ValueError(msg)
            return self.normalise(
                self.dataslices[idx_ds_start][
                    idx_date_start : idx_date_start + n_steps
                ].reshape(n_steps, *self.space.chw)
            )
        except KeyError as exc:
            msg = (
                f"Requested slice of {n_steps} steps following {start_date} "
                f"is out of range for dataset with dates from {self.start_date} to "
                f"{self.end_date}"
            )
            raise ValueError(msg) from exc

    def normalise(self, data: ArrayType) -> ArrayType:
        """Normalise the data to [0, 1] for each channel if configured to do so.

        Note that this can be applied to both ArrayCHW and ArrayTCHW.
        """
        if not self._normalise:
            return data
        if self._norm_offset is None:
            self._norm_offset = (
                self.statistics["minimum"].reshape(-1, 1, 1).astype(data.dtype)
            )
        if self._norm_scale is None:
            self._norm_scale = (
                (1.0 / (self.statistics["maximum"] - self.statistics["minimum"]))
                .reshape(-1, 1, 1)
                .astype(data.dtype)
            )
        return cast("ArrayType", (data - self._norm_offset) * self._norm_scale)

    def subset(
        self,
        *,
        date_ranges: Sequence[dict[str, str | None]] | None = None,
        normalise: bool | None = None,
        variables: Sequence[str] | None = None,
    ) -> "SingleDataset":
        return SingleDataset(
            name=self.name,
            input_files=self._input_files,
            date_ranges=date_ranges or self._date_ranges,
            group=self.group,
            normalise=self._normalise if normalise is None else normalise,
            variables=variables or list(self._variables),
        )

    def to_index(self, date: np.datetime64) -> int:
        """Return the index of a given date in the dataset."""
        date = self.normalise_date(date)
        try:
            return self._date2idx[date]
        except KeyError as exc:
            msg = (
                f"Date {np.datetime_as_string(date, unit='D')} not found in the "
                f"dataset {self.start_date} to {self.end_date} every {self.frequency}"
            )
            raise IndexError(msg) from exc
