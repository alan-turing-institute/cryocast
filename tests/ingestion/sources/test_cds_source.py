"""Tests for the Copernicus Climate Data Store source."""

from datetime import datetime, timedelta
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from anemoi.datasets.create.recipe.dates import StartEndDates
from anemoi.datasets.dates.groups import GroupOfDates

from cryocast.ingestion.sources import CDSSource


class TestCDSSource:
    """Test exact date translation into CDS API requests."""

    context: ClassVar[MagicMock] = MagicMock()
    provider: ClassVar[StartEndDates] = StartEndDates(
        start=datetime(2024, 1, 1, 12),
        end=datetime(2024, 3, 1, 12),
        frequency=timedelta(days=1),
    )

    @classmethod
    def dates(cls, *dates: datetime) -> GroupOfDates:
        """Build a GroupOfDates for the requested timestamps."""
        return GroupOfDates(list(dates), provider=cls.provider)

    def test_execute_injects_exact_dates_and_time(self) -> None:
        """Recipe dates become CDS year/month/day/time fields."""
        mock_from_source = MagicMock(return_value=MagicMock())
        mock_multi = MagicMock()
        request = {
            "level_type": "single_levels",
            "variable": ["sea_ice_area_fraction"],
            "product_type": "analysis",
            "data_format": "grib",
            "area": [81.0, 15.0, 76.0, 35.0],
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.cds.from_source", mock_from_source)
            mp.setattr("cryocast.ingestion.sources.cds.MultiFieldList", mock_multi)
            source = CDSSource(
                self.context,
                dataset="reanalysis-pan-carra",
                request=request,
                time_from_dates=True,
            )
            source.execute(
                self.dates(
                    datetime(2024, 1, 2, 12),
                    datetime(2024, 1, 4, 12),
                )
            )

        expected = {
            **request,
            "year": ["2024"],
            "month": ["01"],
            "day": ["02", "04"],
            "time": ["12:00"],
        }
        mock_from_source.assert_called_once_with(
            "cds",
            "reanalysis-pan-carra",
            request=expected,
            prompt=False,
        )
        mock_multi.assert_called_once()

    def test_execute_splits_months_without_cartesian_extra_dates(self) -> None:
        """Each year/month gets only the exact requested days."""
        mock_from_source = MagicMock(side_effect=[MagicMock(), MagicMock()])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.cds.from_source", mock_from_source)
            source = CDSSource(
                self.context,
                dataset="reanalysis-pan-carra",
                request={"variable": ["sea_ice_area_fraction"]},
            )
            source.execute(
                self.dates(
                    datetime(2024, 1, 31, 12),
                    datetime(2024, 2, 1, 12),
                    datetime(2024, 2, 29, 12),
                )
            )

        assert mock_from_source.call_count == 2
        requests = [item.kwargs["request"] for item in mock_from_source.call_args_list]
        assert requests[0]["year"] == ["2024"]
        assert requests[0]["month"] == ["01"]
        assert requests[0]["day"] == ["31"]
        assert requests[1]["year"] == ["2024"]
        assert requests[1]["month"] == ["02"]
        assert requests[1]["day"] == ["01", "29"]
        assert "time" not in requests[0]
        assert "time" not in requests[1]

    def test_varying_times_are_split_without_cartesian_overfetch(self) -> None:
        """Different valid times are sent as separate exact CDS requests."""
        mock_from_source = MagicMock(side_effect=[MagicMock(), MagicMock()])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.cds.from_source", mock_from_source)
            source = CDSSource(
                self.context,
                dataset="reanalysis-pan-carra",
                request={"variable": ["sea_ice_area_fraction"]},
                time_from_dates=True,
            )
            source.execute(
                self.dates(
                    datetime(2024, 1, 1, 0),
                    datetime(2024, 1, 2, 12),
                )
            )

        requests = [item.kwargs["request"] for item in mock_from_source.call_args_list]
        assert requests[0]["day"] == ["01"]
        assert requests[0]["time"] == ["00:00"]
        assert requests[1]["day"] == ["02"]
        assert requests[1]["time"] == ["12:00"]

    def test_execute_does_not_mutate_request_template(self) -> None:
        """Generated date fields must not leak into subsequent chunks."""
        request = {"variable": ["sea_ice_area_fraction"]}
        original = {"variable": ["sea_ice_area_fraction"]}
        mock_from_source = MagicMock(return_value=MagicMock())

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.cds.from_source", mock_from_source)
            source = CDSSource(self.context, dataset="dataset", request=request)
            source.execute(self.dates(datetime(2024, 1, 2, 12)))

        assert request == original
        assert source.request == original

    @pytest.mark.parametrize("date_key", ["year", "month", "day"])
    def test_rejects_recipe_managed_date_fields(self, date_key: str) -> None:
        """Static date fields would conflict with Anemoi's requested chunk."""
        with pytest.raises(ValueError, match="date fields"):
            CDSSource(
                self.context,
                dataset="dataset",
                request={date_key: ["2024"]},
            )

    def test_rejects_time_when_generated_from_dates(self) -> None:
        """Time cannot be both static and generated from recipe timestamps."""
        with pytest.raises(ValueError, match="time_from_dates"):
            CDSSource(
                self.context,
                dataset="dataset",
                request={"time": ["12:00"]},
                time_from_dates=True,
            )

    def test_empty_dates_do_not_contact_cds(self) -> None:
        """An empty chunk returns without a network request."""
        mock_from_source = MagicMock()
        mock_multi = MagicMock()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.cds.from_source", mock_from_source)
            mp.setattr("cryocast.ingestion.sources.cds.MultiFieldList", mock_multi)
            source = CDSSource(self.context, dataset="dataset", request={})
            source.execute(self.dates())

        mock_from_source.assert_not_called()
        mock_multi.assert_called_once_with([])
