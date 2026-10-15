from pathlib import Path

import numpy as np
import pytest

from cryocast.data.calendar_day_climatology import CalendarDayClimatology
from cryocast.data.combined_dataset import CombinedDataset
from cryocast.data.single_dataset import SingleDataset

N_DAYS = len(CalendarDayClimatology.DAY_INDEX)


class TestCombinedDataset:
    def test_raises_on_different_frequencies(
        self, mock_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds_24h = SingleDataset(name="mock_dataset", input_files=[mock_dataset])
        ds_12h = SingleDataset(name="mock_dataset_2", input_files=[mock_dataset])
        monkeypatch.setitem(ds_12h.__dict__, "frequency", np.timedelta64(12, "h"))
        with pytest.raises(
            ValueError, match="Cannot combine datasets with different frequencies"
        ):
            CombinedDataset(
                datasets=[ds_24h, ds_12h],
                target_group_name="mock_dataset",
                target_variables=["ice_conc"],
            )

    def test_no_valid_dates_non_overlapping_ranges(
        self, mock_dataset: Path, dates_as_str: tuple[str, ...]
    ) -> None:
        """Test that CombinedDataset raises ValueError when datasets have non-overlapping date ranges."""
        # Create and combine two datasets with non-overlapping date ranges
        dataset1 = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
            date_ranges=[{"start": dates_as_str[0], "end": dates_as_str[1]}],
        )
        dataset2 = SingleDataset(
            name="dataset2",
            input_files=[mock_dataset],
            date_ranges=[{"start": dates_as_str[3], "end": dates_as_str[4]}],
        )
        combined = CombinedDataset(
            datasets=[dataset1, dataset2],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=1,
        )
        # Confirm that no valid dates are available
        with pytest.raises(ValueError, match="CombinedDataset has no valid dates"):
            _ = combined.dates

    def test_no_valid_dates_insufficient_history_steps(
        self, mock_dataset: Path, dates_as_str: tuple[str, ...]
    ) -> None:
        """Test that CombinedDataset raises ValueError when history steps exceed available dates."""
        dataset = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
            date_ranges=[{"start": dates_as_str[0], "end": dates_as_str[1]}],
        )
        # Create combined dataset with history steps larger than available dates
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=10,  # Only 2 dates available
            n_forecast_steps=1,
        )
        # Confirm that no valid dates are available
        with pytest.raises(ValueError, match="CombinedDataset has no valid dates"):
            _ = combined.dates

    def test_no_valid_dates_insufficient_forecast_steps(
        self, mock_dataset: Path, dates_as_str: tuple[str, ...]
    ) -> None:
        """Test that CombinedDataset raises ValueError when forecast steps exceed available dates."""
        dataset = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
            date_ranges=[{"start": dates_as_str[0], "end": dates_as_str[1]}],
        )
        # Create combined dataset with forecast steps larger than available dates
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=10,  # Only 2 dates available
        )
        # Confirm that no valid dates are available
        with pytest.raises(ValueError, match="CombinedDataset has no valid dates"):
            _ = combined.dates

    def test_no_valid_dates_incompatible_times(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        """Test that CombinedDataset raises ValueError when datasets have incompatible times."""
        dataset = SingleDataset(name="dataset", input_files=[mock_dataset])
        # Create a second dataset with an offset of 12h
        dataset_offset = SingleDataset(
            name="dataset_offset", input_files=[mock_dataset]
        )
        dataset_offset.dates = [date + np.timedelta64(720, "m") for date in dates_as_np]
        # Create combined dataset
        combined = CombinedDataset(
            datasets=[dataset, dataset_offset],
            target_group_name="dataset",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=1,
        )
        with pytest.raises(ValueError, match="CombinedDataset has no valid dates"):
            _ = combined.dates

    def test_valid_dates(self, mock_dataset: Path) -> None:
        """Test that CombinedDataset works correctly with valid overlapping dates."""
        dataset1 = SingleDataset(name="dataset1", input_files=[mock_dataset])
        dataset2 = SingleDataset(name="dataset2", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset1, dataset2],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=2,
            n_forecast_steps=1,
        )
        dates = combined.dates
        assert len(dates) > 0
        assert all(isinstance(date, np.datetime64) for date in dates)

    def test_dates_windowing(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        dataset = SingleDataset(name="dataset1", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=2,
            n_forecast_steps=1,
        )
        # With these settings we have a window of (2+1=3) steps.
        # This means that only the first three dates will be valid start dates.
        assert combined.dates == list(dates_as_np[:3])

    def test_dates_windowing_tight_window(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        dataset = SingleDataset(name="dataset1", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=3,
            n_forecast_steps=2,
        )
        # With these settings we have a window of (2+3=5) steps.
        # This means that only the first date will be a valid start date.
        assert combined.dates == [dates_as_np[0]]

    def test_get_history_steps(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        dataset = SingleDataset(name="dataset1", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=3,
            n_forecast_steps=1,
        )
        steps = combined.get_history_steps(dates_as_np[0])
        assert steps == [dates_as_np[0], dates_as_np[1], dates_as_np[2]]

    def test_contemporaneous_target_offset(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        """A zero target offset pairs low- and high-resolution data on the same date."""
        source = SingleDataset(name="source", input_files=[mock_dataset])
        target = SingleDataset(name="target", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[source, target],
            target_group_name="target",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=1,
            target_offset_steps=0,
        )

        assert combined.get_forecast_steps(dates_as_np[0]) == [dates_as_np[0]]
        batch = combined[0]
        np.testing.assert_array_equal(
            batch["target"],
            combined.target.get_tchw([dates_as_np[0]]),
        )

    def test_negative_target_offset_rejected(self, mock_dataset: Path) -> None:
        """Targets cannot start before the input window."""
        dataset = SingleDataset(name="target", input_files=[mock_dataset])
        with pytest.raises(ValueError, match="target_offset_steps"):
            CombinedDataset(
                datasets=[dataset],
                target_group_name="target",
                target_variables=["ice_conc"],
                target_offset_steps=-1,
            )

    def test_get_forecast_steps(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        dataset = SingleDataset(name="dataset1", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=2,
            n_forecast_steps=2,
        )
        steps = combined.get_forecast_steps(dates_as_np[0])
        assert steps == [dates_as_np[2], dates_as_np[3]]

    def test_start_and_end_dates(
        self, mock_dataset: Path, dates_as_str: tuple[str, ...]
    ) -> None:
        """Test that start_date and end_date are correctly calculated from available dates."""
        dataset1 = SingleDataset(
            name="dataset1",
            input_files=[mock_dataset],
            date_ranges=[{"start": dates_as_str[0], "end": dates_as_str[-2]}],
        )
        dataset2 = SingleDataset(
            name="dataset2",
            input_files=[mock_dataset],
            date_ranges=[{"start": dates_as_str[1], "end": dates_as_str[-1]}],
        )
        combined = CombinedDataset(
            datasets=[dataset1, dataset2],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=1,
        )
        assert combined.start_date == combined.dates[0]
        assert combined.end_date == combined.dates[-1]

    def test_getitem(self, mock_dataset: Path) -> None:
        """__getitem__ returns a dict with the expected keys and array shapes."""
        dataset = SingleDataset(name="dataset1", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=2,
            n_forecast_steps=1,
        )
        batch = combined[0]
        assert isinstance(batch, dict)
        assert set(batch.keys()) == {"dataset1", "target"}
        # input has shape [n_history_steps, C_input, H, W]
        assert batch["dataset1"].shape == (2, 3, 2, 2)
        # target has shape (n_forecast_steps, C_target, H, W)
        assert batch["target"].shape == (1, 1, 2, 2)

    def test_getitem_time_offset(
        self, mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
    ) -> None:
        """__getitem__ places the target window n_history_steps ahead of the input window."""
        dataset = SingleDataset(name="dataset1", input_files=[mock_dataset])
        combined = CombinedDataset(
            datasets=[dataset],
            target_group_name="dataset1",
            target_variables=["ice_conc"],
            n_history_steps=2,
            n_forecast_steps=1,
        )
        # With start_date 2020-01-01, n_history_steps=2 and n_forecast_steps=1 we expect
        # history covers 2020-01-01 to 2020-01-02
        # target covers 2020-01-03
        batch = combined[0]
        np.testing.assert_array_equal(
            batch["dataset1"],
            combined.inputs[0].get_tchw([dates_as_np[0], dates_as_np[1]]),
        )
        np.testing.assert_array_equal(
            batch["target"],
            combined.target.get_tchw([dates_as_np[2]]),
        )


class TestCombinedDatasetClimatology:
    """Tests for the CombinedDataset climatology key and accessor."""

    @staticmethod
    def _combined(
        climatology_zarr: Path,
        climatology: np.ndarray | None,
    ) -> CombinedDataset:
        ds = SingleDataset(
            name="sic_south",
            input_files=[climatology_zarr],
            date_ranges=[{"start": "2019-12-28", "end": "2020-01-05"}],
        )
        return CombinedDataset(
            datasets=[ds],
            target_group_name="sic_south",
            target_variables=["ice_conc"],
            n_history_steps=1,
            n_forecast_steps=2,
            climatology=climatology,
        )

    @staticmethod
    def _table() -> np.ndarray:
        """A [366, C, H, W] table whose value encodes the calendar-day index."""
        table = np.zeros((N_DAYS, 1, 2, 2), dtype=np.float32)
        for index in range(N_DAYS):
            table[index] = 100.0 * index + 0.5
        return table

    def test_getitem_includes_climatology(self, climatology_zarr: Path) -> None:
        """Batches gain a climatology key with the calendar day of each forecast step."""
        table = self._table()
        combined = self._combined(climatology_zarr, table)
        idx = combined.dates.index(np.datetime64("2019-12-30T12:00:00", "s"))

        batch = combined[idx]
        assert set(batch.keys()) == {"sic_south", "target", "climatology"}
        assert batch["climatology"].shape == (2, 1, 2, 2)
        # Forecast steps are 2019-12-31 and 2020-01-01.
        dec_31 = CalendarDayClimatology.day_index(np.datetime64("2019-12-31"))
        jan_1 = CalendarDayClimatology.day_index(np.datetime64("2020-01-01"))
        np.testing.assert_array_equal(batch["climatology"][0], table[dec_31])
        np.testing.assert_array_equal(batch["climatology"][1], table[jan_1])

    def test_getitem_without_climatology_is_unchanged(
        self, climatology_zarr: Path
    ) -> None:
        """Without a climatology table the batch keys and values are exactly as before."""
        combined = self._combined(climatology_zarr, None)
        idx = combined.dates.index(np.datetime64("2019-12-30T12:00:00", "s"))

        batch = combined[idx]
        assert set(batch.keys()) == {"sic_south", "target"}
        np.testing.assert_array_equal(
            batch["target"],
            combined.target.get_tchw(
                [
                    np.datetime64("2019-12-31T12:00:00", "s"),
                    np.datetime64("2020-01-01T12:00:00", "s"),
                ]
            ),
        )

    def test_climatology_for(self, climatology_zarr: Path) -> None:
        """climatology_for returns the calendar-day fields for the forecast steps."""
        table = self._table()
        combined = self._combined(climatology_zarr, table)
        start = np.datetime64("2019-12-31T12:00:00", "s")

        result = combined.climatology_for(start)
        assert result is not None
        # Forecast steps are 2020-01-01 and 2020-01-02.
        jan_1 = CalendarDayClimatology.day_index(np.datetime64("2020-01-01"))
        jan_2 = CalendarDayClimatology.day_index(np.datetime64("2020-01-02"))
        np.testing.assert_array_equal(result[0], table[jan_1])
        np.testing.assert_array_equal(result[1], table[jan_2])

    def test_climatology_for_none(self, climatology_zarr: Path) -> None:
        """climatology_for returns None when no climatology table was provided."""
        combined = self._combined(climatology_zarr, None)
        assert (
            combined.climatology_for(np.datetime64("2019-12-31T12:00:00", "s")) is None
        )
