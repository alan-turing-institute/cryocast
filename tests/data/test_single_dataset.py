import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from icenet_mp.data.single_dataset import SingleDataset
from icenet_mp.types import DataSpace


class TestSingleDataset:
    def test_name(self) -> None:
        dataset = SingleDataset(name="test_dataset", input_files=[])
        assert dataset.name == "test_dataset"

    def test_dates(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        assert all(date in dataset.dates for date in dates_as_np)

    def test_end_date(
        self,
        mock_dataset: Path,
        dates_as_str: tuple[str, ...],
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[{"start": None, "end": dates_as_str[1]}],
        )
        assert dataset.start_date == dates_as_np[0]
        assert dataset.end_date == dates_as_np[1]

    def test_date_ranges(
        self,
        mock_dataset: Path,
        dates_as_str: tuple[str, ...],
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[
                {"start": dates_as_str[0], "end": dates_as_str[1]},
                {"start": dates_as_str[-2], "end": dates_as_str[-1]},
            ],
        )
        assert dates_as_np[2] not in dataset.dates
        assert len(dataset.dataslices) == 2
        assert len(dataset) == 4

    def test_missing_dates(
        self,
        mock_dataset_missing_dates: Path,
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        """Test that missing dates are excluded from SingleDataset.dates."""
        dataset = SingleDataset(
            name="test_missing", input_files=[mock_dataset_missing_dates]
        )
        missing_indices = {1, 3}
        expected_dates = [
            date for idx, date in enumerate(dates_as_np) if idx not in missing_indices
        ]
        assert len(expected_dates) == 3
        assert dataset.dates == expected_dates
        assert dates_as_np[1] not in dataset.dates  # 2020-01-02 should be missing
        assert dates_as_np[3] not in dataset.dates  # 2020-01-04 should be missing

    def test_frequency_ignores_missing_dates(
        self, mock_dataset_missing_dates: Path, dates_as_str: tuple[str, ...]
    ) -> None:
        """Frequency should come from the dataset's declared metadata.

        Subset recalculates frequency by diffing the first two dates of whichever
        date-range slice is requested, so we need to ignore this.
        """
        dataset = SingleDataset(
            name="test_missing",
            input_files=[mock_dataset_missing_dates],
            date_ranges=[
                # This range has a 2-day cadence
                {"start": None, "end": dates_as_str[4]},
            ],
        )
        assert dataset.frequency == np.timedelta64(1, "D")

    def test_start_date(
        self,
        mock_dataset: Path,
        dates_as_str: tuple[str, ...],
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[{"start": dates_as_str[1], "end": None}],
        )
        assert dataset.start_date == dates_as_np[1]
        assert dataset.end_date == dates_as_np[-1]

    def test_datetime_normalization(
        self, mock_dataset_non_normalized_times: Path
    ) -> None:
        """Test that datetime normalization is applied to all dates."""
        dataset = SingleDataset(
            name="test_normalized",
            input_files=[mock_dataset_non_normalized_times],
        )
        for date in dataset.dates:
            dt: datetime = date.astype("datetime64[us]").astype(datetime)
            assert dt.hour == 12
            assert dt.minute == 0
            assert dt.second == 0
            assert dt.microsecond == 0
        expected_dates = [
            np.datetime64("2020-01-01T12:00:00"),
            np.datetime64("2020-01-02T12:00:00"),
            np.datetime64("2020-01-03T12:00:00"),
            np.datetime64("2020-01-04T12:00:00"),
            np.datetime64("2020-01-05T12:00:00"),
        ]
        assert dataset.dates == expected_dates

    def test_getitem(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        data_array = dataset[0]
        assert isinstance(data_array, np.ndarray)
        assert data_array.shape == (3, 2, 2)
        # Check exception for out of range
        with pytest.raises(
            IndexError, match="Index 10 out of range for dataset of length 5"
        ):
            dataset[10]

    def test_get_tchw(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        data_array = dataset.get_tchw(dates_as_np)
        assert isinstance(data_array, np.ndarray)
        assert data_array.shape == (5, 3, 2, 2)
        # Check exception for out of range
        with pytest.raises(
            IndexError, match="Date 1970-01-01 not found in the dataset"
        ):
            dataset.get_tchw([np.datetime64("1970-01-01"), np.datetime64("1970-01-02")])

    def test_get_tchw_slice(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        """get_tchw_slice returns the correct shape and the same data as get_tchw."""
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        result = dataset.get_tchw_slice(dates_as_np[0], 3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3, 2, 2)
        np.testing.assert_array_equal(result, dataset.get_tchw(list(dates_as_np[:3])))

    def test_get_tchw_slice_check_raises(
        self,
        mock_dataset: Path,
        dates_as_str: tuple[str, ...],
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        # Requesting 3 steps from "2020-01-01" spans two dataslices:
        # (2020-01-01 to 2020-01-02) and (2020-01-03 to 2020-01-04)
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[
                {"start": dates_as_str[0], "end": dates_as_str[1]},
                {"start": dates_as_str[3], "end": dates_as_str[4]},
            ],
        )
        with pytest.raises(ValueError, match="crosses the boundary between dataslices"):
            dataset.get_tchw_slice(dates_as_np[0], 3)

    def test_get_tchw_slice_check_false(
        self,
        mock_dataset: Path,
        dates_as_str: tuple[str, ...],
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        # Requesting 3 steps from "2020-01-01" spans two dataslices:
        # (2020-01-01 to 2020-01-02) and (2020-01-03 to 2020-01-04)
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[
                {"start": dates_as_str[0], "end": dates_as_str[1]},
                {"start": dates_as_str[3], "end": dates_as_str[4]},
            ],
        )
        with pytest.raises(ValueError, match="cannot reshape array"):
            dataset.get_tchw_slice(dates_as_np[0], 3, check=False)

    def test_touching_ranges_merge_into_one_dataslice(
        self,
        mock_dataset: Path,
        dates_as_str: tuple[str, ...],
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        # a small test of merging touching ranges, described in issue #279.
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            date_ranges=[
                {"start": dates_as_str[0], "end": dates_as_str[1]},
                {"start": dates_as_str[2], "end": dates_as_str[4]},
            ],
        )
        assert len(dataset.dataslices) == 1
        assert len(dataset) == 5
        result = dataset.get_tchw_slice(dates_as_np[0], 3, check=False)
        assert result.shape == (3, 3, 2, 2)

    def test_get_tchw_with_missing_dates(
        self,
        mock_dataset_missing_dates: Path,
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        """Test that get_tchw works correctly when dates are missing."""
        dataset = SingleDataset(
            name="test_missing", input_files=[mock_dataset_missing_dates]
        )
        missing_indices = {1, 3}
        expected_dates = [
            date for idx, date in enumerate(dates_as_np) if idx not in missing_indices
        ]
        assert len(expected_dates) == 3
        result = dataset.get_tchw(expected_dates)
        assert result.shape == (3, 1, 2, 2)
        with pytest.raises(
            IndexError, match="Date 2020-01-02 not found in the dataset"
        ):
            dataset.get_tchw([dates_as_np[1]])

    def test_len(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        assert len(dataset) == 5

    def test_len_with_missing_dates(self, mock_dataset_missing_dates: Path) -> None:
        """Test that dataset length reflects missing dates."""
        dataset = SingleDataset(
            name="test_missing", input_files=[mock_dataset_missing_dates]
        )
        assert len(dataset) == 3

    def test_space(self, mock_dataset: Path) -> None:
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        assert isinstance(dataset.space, DataSpace)
        assert dataset.space.channels == 3
        assert dataset.space.shape == (2, 2)

    def test_subset(self, mock_dataset: Path) -> None:
        """Test the select_variables classmethod."""
        original_dataset = SingleDataset(
            name="mock_dataset", input_files=[mock_dataset]
        )
        assert original_dataset.space.channels == 3
        subset_dataset = original_dataset.subset(variables=["ice_conc"])
        assert subset_dataset.space.channels == 1
        assert subset_dataset.name == "mock_dataset"
        data_array = subset_dataset[0]
        assert data_array.shape == (1, 2, 2)

    def test_subset_variable_order_follows_storage_not_request_order(
        self, mock_dataset: Path
    ) -> None:
        """`subset` selects the requested variables while keeping on-disk order.

        `mock_dataset` stores variables as (ice_conc, ice_thickness, temperature).
        Variables are requested here in the opposite order, but the subset's
        `variable_names` still comes back in storage order: `variables` is turned
        into a `set` internally so anemoi does not reorder the underlying data, so
        any caller that needs a specific channel by index must read it back from
        `variable_names` rather than assuming it matches the requested order.
        """
        original_dataset = SingleDataset(
            name="mock_dataset", input_files=[mock_dataset]
        )
        subset_dataset = original_dataset.subset(variables=["temperature", "ice_conc"])
        assert subset_dataset.variable_names == ["ice_conc", "temperature"]

    def test_subset_preserves_date_ranges(
        self,
        mock_dataset: Path,
        dates_as_str: tuple[str, ...],
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        """Test that select_variables preserves date ranges."""
        original_dataset = SingleDataset(
            name="mock_dataset_multi",
            input_files=[mock_dataset],
            date_ranges=[{"start": dates_as_str[0], "end": dates_as_str[2]}],
        )
        subset_dataset = original_dataset.subset(variables=["ice_thickness"])
        assert subset_dataset.start_date == dates_as_np[0]
        assert subset_dataset.end_date == dates_as_np[2]
        assert len(subset_dataset) == 3

    def test_to_index(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        assert dataset.to_index(dates_as_np[0]) == 0
        assert dataset.to_index(dates_as_np[3]) == 3
        with pytest.raises(
            IndexError, match="Date 1970-01-01 not found in the dataset"
        ):
            dataset.to_index(np.datetime64("1970-01-01"))

    def test_to_index_with_missing_dates(
        self,
        mock_dataset_missing_dates: Path,
        dates_as_np: tuple[np.datetime64, ...],
    ) -> None:
        """Test that to_index works correctly when dates are missing."""
        dataset = SingleDataset(
            name="test_missing", input_files=[mock_dataset_missing_dates]
        )
        assert dataset.to_index(dates_as_np[0]) == 0  # 2020-01-01
        assert dataset.to_index(dates_as_np[2]) == 1  # 2020-01-03
        assert dataset.to_index(dates_as_np[4]) == 2  # 2020-01-05
        with pytest.raises(
            IndexError, match="Date 2020-01-02 not found in the dataset"
        ):
            dataset.to_index(dates_as_np[1])
        with pytest.raises(
            IndexError, match="Date 2020-01-04 not found in the dataset"
        ):
            dataset.to_index(dates_as_np[3])

    def test_variable_selection_all(self, mock_dataset: Path) -> None:
        """Test selecting all variables from a multi-variable dataset."""
        dataset = SingleDataset(
            name="mock_dataset_multi",
            input_files=[mock_dataset],
            variables=["ice_conc", "ice_thickness", "temperature"],
        )
        assert dataset.space.channels == 3
        data_array = dataset[0]
        assert data_array.shape == (3, 2, 2)

    def test_variable_selection_multiple(self, mock_dataset: Path) -> None:
        """Test selecting multiple variables from a multi-variable dataset."""
        dataset = SingleDataset(
            name="mock_dataset_multi",
            input_files=[mock_dataset],
            variables=["ice_conc", "temperature"],
        )
        assert dataset.space.channels == 2
        data_array = dataset[0]
        assert data_array.shape == (2, 2, 2)

    def test_variable_selection_none(self, mock_dataset: Path) -> None:
        """Test that not specifying variables loads all variables."""
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        assert dataset.space.channels == 3

    def test_variable_selection_single(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        """Test selecting a single variable from a multi-variable dataset."""
        dataset = SingleDataset(
            name="mock_dataset",
            input_files=[mock_dataset],
            variables=["ice_conc"],
        )
        assert dataset.space.channels == 1
        data_array = dataset[0]
        assert data_array.shape == (1, 2, 2)
        assert dataset.start_date == dates_as_np[0]
        assert dataset.end_date == dates_as_np[-1]

    def test_normalise_formula_chw(self, mock_dataset: Path) -> None:
        """Normalised __getitem__ output equals (raw - minimum) / (maximum - minimum).

        Verifies the statistics key names ('minimum', 'maximum') and the per-channel
        [C,1,1] reshape for CHW broadcasting.
        """
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        dataset_raw = SingleDataset(
            name="mock_dataset", input_files=[mock_dataset], normalise=False
        )
        min_ = dataset.statistics["minimum"][:, None, None].astype(np.float64)
        max_ = dataset.statistics["maximum"][:, None, None].astype(np.float64)
        for idx in range(len(dataset)):
            raw = dataset_raw[idx].astype(np.float64)
            expected = (raw - min_) / (max_ - min_)
            np.testing.assert_allclose(
                dataset[idx].astype(np.float64), expected, rtol=1e-5
            )

    def test_normalise_formula_tchw(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        """Normalised get_tchw_slice output equals (raw - minimum) / (maximum - minimum).

        Verifies that [C,1,1] broadcasting works correctly against TCHW arrays —
        a regression in the reshape would silently misalign channels across time.
        """
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        dataset_raw = SingleDataset(
            name="mock_dataset", input_files=[mock_dataset], normalise=False
        )
        min_ = dataset.statistics["minimum"][:, None, None].astype(np.float64)
        max_ = dataset.statistics["maximum"][:, None, None].astype(np.float64)
        raw = dataset_raw.get_tchw_slice(dates_as_np[0], len(dataset)).astype(
            np.float64
        )
        expected = (raw - min_) / (max_ - min_)
        result = dataset.get_tchw_slice(dates_as_np[0], len(dataset)).astype(np.float64)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_normalise_false_returns_raw_chw(self, mock_dataset: Path) -> None:
        """normalise=False __getitem__ returns raw values, not normalised ones."""
        dataset_raw = SingleDataset(
            name="mock_dataset", input_files=[mock_dataset], normalise=False
        )
        dataset_norm = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        raw = dataset_raw[0]
        normalised = dataset_norm[0]
        assert not np.allclose(raw, normalised), (
            "normalise=False output equals normalised output"
        )

    def test_normalise_false_returns_raw_tchw(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        """normalise=False get_tchw_slice returns raw values, not normalised ones."""
        dataset_raw = SingleDataset(
            name="mock_dataset", input_files=[mock_dataset], normalise=False
        )
        dataset_norm = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        raw = dataset_raw.get_tchw_slice(dates_as_np[0], 3)
        normalised = dataset_norm.get_tchw_slice(dates_as_np[0], 3)
        assert not np.allclose(raw, normalised), (
            "normalise=False TCHW output equals normalised output"
        )

    def test_normalise_constant_channel_produces_nan(
        self, mock_dataset_constant_values: Path
    ) -> None:
        """When max == min for a channel, normalisation divides by zero (0 * inf = NaN).

        This documents the current behaviour so that any future fix is made deliberately.
        """
        dataset = SingleDataset(
            name="constant", input_files=[mock_dataset_constant_values]
        )
        assert np.all(np.isnan(dataset[0]))

    @pytest.mark.parametrize("flag", [True, False], ids=["normalise", "raw"])
    def test_subset_preserves_normalise_flag(
        self, mock_dataset: Path, *, flag: bool
    ) -> None:
        """subset() propagates the normalise flag to the child dataset."""
        dataset = SingleDataset(
            name="mock_dataset", input_files=[mock_dataset], normalise=flag
        )
        subset = dataset.subset(variables=["ice_conc"])
        assert subset._normalise is flag

    def test_normalise_date_sets_time_to_noon(self) -> None:
        """Normalize dates to noon while preserving the calendar date."""
        result = SingleDataset.normalise_date(np.datetime64("2026-08-21T03:14:15.926"))

        assert result == np.datetime64("2026-08-21T12:00:00.000")

    def test_normalise_date_ranges_adjacent_merge(self) -> None:
        """Consecutive ranges (next starts the day after) merge into one span."""
        assert SingleDataset.normalise_date_ranges(
            [
                {"start": "2020-01-01", "end": "2020-06-30"},
                {"start": "2020-07-01", "end": "2020-12-31"},
            ]
        ) == [{"start": "2020-01-01", "end": "2020-12-31"}]

    def test_normalise_date_ranges_shared_boundary_merge(self) -> None:
        """Ranges meeting on the same day merge (treated as consecutive)."""
        assert SingleDataset.normalise_date_ranges(
            [
                {"start": "2020-01-01", "end": "2020-07-01"},
                {"start": "2020-07-01", "end": "2020-12-31"},
            ]
        ) == [{"start": "2020-01-01", "end": "2020-12-31"}]

    def test_normalise_date_ranges_gap_not_merged(self) -> None:
        """A missing day between ranges leaves them as separate spans."""
        ranges: list[dict[str, str | None]] = [
            {"start": "2020-01-01", "end": "2020-06-30"},
            {"start": "2020-07-02", "end": "2020-12-31"},
        ]
        assert SingleDataset.normalise_date_ranges(ranges) == ranges

    def test_normalise_date_ranges_overlap_merge(self) -> None:
        """A range fully containing another merges to the outer span (no truncation)."""
        assert SingleDataset.normalise_date_ranges(
            [
                {"start": "2020-01-01", "end": "2020-12-31"},
                {"start": "2020-03-01", "end": "2020-06-30"},
            ]
        ) == [{"start": "2020-01-01", "end": "2020-12-31"}]

    def test_normalise_date_ranges_partial_overlap_merge(self) -> None:
        """Partially overlapping ranges merge to earliest start and latest end."""
        assert SingleDataset.normalise_date_ranges(
            [
                {"start": "2025-01-01", "end": "2025-03-31"},
                {"start": "2025-02-01", "end": "2025-05-31"},
            ]
        ) == [{"start": "2025-01-01", "end": "2025-05-31"}]

    def test_normalise_date_ranges_end_none_merge(self) -> None:
        """An open end (None = after any date) overlaps a later range, so they merge."""
        assert SingleDataset.normalise_date_ranges(
            [
                {"start": "2020-01-01", "end": None},
                {"start": "2020-06-01", "end": "2020-09-30"},
            ]
        ) == [{"start": "2020-01-01", "end": None}]

    def test_normalise_date_ranges_end_none_inside_other_merge(self) -> None:
        """An open-ended range starting inside another keeps the earliest start, open end."""
        assert SingleDataset.normalise_date_ranges(
            [
                {"start": "2020-06-01", "end": None},
                {"start": "2020-01-01", "end": "2020-09-30"},
            ]
        ) == [{"start": "2020-01-01", "end": None}]

    def test_normalise_date_ranges_start_none_merge(self) -> None:
        """Two open starts (None = before any date) overlap, so they merge."""
        assert SingleDataset.normalise_date_ranges(
            [
                {"start": None, "end": "2020-03-31"},
                {"start": None, "end": "2020-06-30"},
            ]
        ) == [{"start": None, "end": "2020-06-30"}]

    def test_normalise_date_ranges_open_outer_ends_merge(self) -> None:
        """Open-beginning + open-ending halves that meet merge to the whole range."""
        assert SingleDataset.normalise_date_ranges(
            [
                {"start": None, "end": "2020-06-30"},
                {"start": "2020-07-01", "end": None},
            ]
        ) == [{"start": None, "end": None}]

    def test_normalise_date_ranges_warns_on_merge(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A merge emits exactly one warning so the user notices ranges were altered."""
        with caplog.at_level(logging.WARNING):
            result = SingleDataset.normalise_date_ranges(
                [
                    {"start": "2020-01-01", "end": "2020-06-30"},
                    {"start": "2020-07-01", "end": "2020-12-31"},
                ]
            )
        assert result == [{"start": "2020-01-01", "end": "2020-12-31"}]
        merge_warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING
            and "Merged overlapping or touching date ranges" in record.getMessage()
        ]
        assert len(merge_warnings) == 1

    def test_normalise_date_ranges_single_and_empty_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty or single-range input is returned unchanged with no merge warning."""
        with caplog.at_level(logging.WARNING):
            assert SingleDataset.normalise_date_ranges([]) == []
            assert SingleDataset.normalise_date_ranges(
                [{"start": "2020-01-01", "end": "2020-06-30"}]
            ) == [{"start": "2020-01-01", "end": "2020-06-30"}]
        assert not [
            record for record in caplog.records if record.levelno == logging.WARNING
        ]


class TestSingleDatasetDateLookupNormalization:
    def test_date_lookups_normalise_to_noon(
        self, mock_dataset_non_normalized_times: Path
    ) -> None:
        """Confirm that a midnight datetime is normalised during lookup."""
        dataset = SingleDataset(
            name="test_normalized",
            input_files=[mock_dataset_non_normalized_times],
        )
        midnight = np.datetime64("2020-01-01")

        assert dataset.to_index(midnight) == 0
        np.testing.assert_array_equal(
            dataset.get_tchw_slice(midnight, 1),
            dataset.get_tchw([dataset.dates[0]]),
        )


class TestSingleDatasetCache:
    """Anemoi datasets are opened once per fileset and shared via a class-level cache."""

    def test_anemoi_dataset_is_loaded_once_per_fileset(
        self, mock_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated reads of one fileset should reuse the class-level dataset cache."""
        mock_open_dataset = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.data.single_dataset.open_dataset", mock_open_dataset
        )
        monkeypatch.setattr(SingleDataset, "anemoi_cache", {})
        input_files = (mock_dataset,)

        first = SingleDataset.load_dataset(input_files)
        second = SingleDataset.load_dataset(input_files)

        assert first is second
        mock_open_dataset.assert_called_once_with(input_files)

    def test_subsets_share_cached_anemoi_dataset(
        self, mock_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SingleDataset subsets should not reopen their shared source files."""
        mock_open_dataset = MagicMock()
        monkeypatch.setattr(
            "icenet_mp.data.single_dataset.open_dataset", mock_open_dataset
        )
        monkeypatch.setattr(SingleDataset, "anemoi_cache", {})
        dataset = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        subset = dataset.subset(variables=["ice_conc"])

        first = dataset.load_dataset(dataset._input_files)
        second = subset.load_dataset(subset._input_files)

        assert first is second
        mock_open_dataset.assert_called_once_with((mock_dataset,))
