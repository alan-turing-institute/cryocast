"""Tests for the Argo data source."""

from datetime import datetime, timedelta
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
from anemoi.datasets.create.recipe.dates import StartEndDates
from anemoi.datasets.dates.groups import GroupOfDates
from anemoi.utils.registry import Registry

from icenet_mp.ingestion.sources import ArgoSource, register_sources
from icenet_mp.ingestion.sources.argo import _fetch_argo_dataframe_with_retry


class TestArgoSource:
    """Test suite for ArgoSource class."""

    mock_context: ClassVar[MagicMock] = MagicMock()
    date_range: ClassVar[StartEndDates] = StartEndDates(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 3),
        frequency=timedelta(days=1),
    )
    dates: ClassVar[GroupOfDates] = GroupOfDates(list(date_range), provider=date_range)

    @pytest.fixture(autouse=True)
    def _patch_erddap_get(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prevent erddapy from making a real request to GitHub."""
        original_get = requests.get

        def _failed_get(url: str, *args: Any, **kwargs: Any) -> requests.Response:
            if "erddaps.json" in url:
                msg = "Force erddapy to use a local server list fallback"
                raise requests.HTTPError(msg)
            return original_get(url, *args, **kwargs)

        monkeypatch.setattr(requests, "get", _failed_get)

    def test_argo_source_registration(self) -> None:
        """Test that ArgoSource is properly registered."""
        # Mock source registry
        mock_registry: Registry = Registry("anemoi.datasets.create.sources")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "icenet_mp.ingestion.sources.source_registry",
                mock_registry,
            )
            assert "argo" not in mock_registry.registered
            register_sources()
            assert "argo" in mock_registry.registered
            assert mock_registry.lookup("argo") == ArgoSource

    def test_argo_source_execute_basic(self) -> None:
        """Test basic Argo source execution with mocked Argo connection."""
        # Build a realistic DataFetcher chain: DataFetcher().region(...).to_dataframe()
        mock_fetcher_instance = MagicMock()
        mock_region_fetcher = MagicMock()
        mock_fetcher_instance.region.return_value = mock_region_fetcher

        # DataFrame must include LATITUDE, LONGITUDE, and requested variable(s)
        df = pd.DataFrame(
            {
                "LATITUDE": [10.0, 11.0, 12.0],
                "LONGITUDE": [21.0, 22.0, 23.0],
                "TEMP": [1.0, 2.0, 3.0],
            }
        )
        mock_region_fetcher.to_dataframe.return_value = df

        mock_datafetcher_cls = MagicMock(return_value=mock_fetcher_instance)

        # load_one is called once for the whole dataset, not once per date
        mock_load_one = MagicMock()
        n_dates = len(self.dates.dates)
        n_params = 1
        mock_load_one.return_value = [MagicMock() for _ in range(n_dates * n_params)]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "icenet_mp.ingestion.sources.lazy_argopy.DataFetcher",
                mock_datafetcher_cls,
                raising=False,
            )
            mp.setattr("icenet_mp.ingestion.sources.argo.load_one", mock_load_one)

            source = ArgoSource(
                area="20/30/0/40",
                context=self.mock_context,
                crs="EPSG:6931",
                param=["TEMP"],
                resolution="25p0km",
                shape=(4, 4),
            )
            result = source.execute(argument=self.dates)

        # DataFetcher instantiated once per requested date (inside retry helper)
        assert mock_datafetcher_cls.call_count == n_dates
        assert mock_fetcher_instance.region.call_count == n_dates
        assert mock_region_fetcher.to_dataframe.call_count == n_dates

        # load_one called once with all dates
        assert mock_load_one.call_count == 1
        assert len(result) == n_dates * n_params

    def test_fetch_argo_dataframe_with_retry_retries_then_succeeds(self) -> None:
        region = [20.0, 30.0, 0.0, 40.0, 0.0, 50.0]

        # First call raises an TimeoutError due to a 503, second succeeds
        first_fetcher = MagicMock()
        first_fetcher.region.side_effect = TimeoutError("503 Service Unavailable")

        second_fetcher = MagicMock()
        second_region_fetcher = MagicMock()
        second_fetcher.region.return_value = second_region_fetcher
        second_region_fetcher.to_dataframe.return_value = pd.DataFrame(
            {"LATITUDE": [10.0], "LONGITUDE": [20.0], "TEMP": [1.0]}
        )

        datafetcher_cls = MagicMock(side_effect=[first_fetcher, second_fetcher])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "icenet_mp.ingestion.sources.lazy_argopy.DataFetcher",
                datafetcher_cls,
                raising=False,
            )
            mp.setattr("icenet_mp.ingestion.sources.argo.time.sleep", MagicMock())

            df = _fetch_argo_dataframe_with_retry(
                region=region,
                time_window=list(self.dates.dates),
                max_attempts=5,
                initial_backoff_s=0.0,
            )

        assert not df.empty
        assert datafetcher_cls.call_count == 2

    def test_fetch_argo_dataframe_with_retry_raises_after_max_attempts(self) -> None:
        region = [20.0, 30.0, 0.0, 40.0, 0.0, 50.0]
        max_attempts = 3

        failing_fetcher = MagicMock()
        failing_fetcher.region.side_effect = FileNotFoundError(
            "503 Service Unavailable"
        )
        datafetcher_cls = MagicMock(return_value=failing_fetcher)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "icenet_mp.ingestion.sources.lazy_argopy.DataFetcher",
                datafetcher_cls,
                raising=False,
            )
            mp.setattr("icenet_mp.ingestion.sources.argo.time.sleep", MagicMock())

            with pytest.raises(
                LookupError, match=r"^Argo float data for .* is unavailable\.$"
            ):
                _fetch_argo_dataframe_with_retry(
                    region=region,
                    time_window=list(self.dates.dates),
                    max_attempts=max_attempts,
                    initial_backoff_s=0.0,
                )

        assert datafetcher_cls.call_count == max_attempts

    def test_argo_source_execute_missing_date_raises(self) -> None:
        """Test that a date with no Argo data raises LookupError without being swallowed."""
        mock_region_no_data = MagicMock()
        mock_region_no_data.to_dataframe.side_effect = LookupError("no data for region")
        mock_fetcher_no_data = MagicMock()
        mock_fetcher_no_data.region.return_value = mock_region_no_data

        mock_datafetcher_cls = MagicMock(side_effect=[mock_fetcher_no_data])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "icenet_mp.ingestion.sources.lazy_argopy.DataFetcher",
                mock_datafetcher_cls,
                raising=False,
            )
            mp.setattr("icenet_mp.ingestion.sources.argo.load_one", MagicMock())

            source = ArgoSource(
                context=self.mock_context,
                area="20/30/0/40",
                crs="EPSG:6932",
                param=["TEMP"],
                resolution="25p0km",
                shape=(432, 432),
            )
            with pytest.raises(LookupError, match="no data for region"):
                source.execute(argument=self.dates)

    def test_argo_source_execute_with_load_one(self) -> None:
        """Execute against the anemoi load_one by mocking only the DataFetcher."""
        df = pd.DataFrame(
            {
                "LATITUDE": [10.0, 11.0],
                "LONGITUDE": [21.0, 22.0],
                "TEMP": [1.0, 2.0],
            }
        )
        mock_region_fetcher = MagicMock()
        mock_region_fetcher.to_dataframe.return_value = df
        mock_fetcher_instance = MagicMock()
        mock_fetcher_instance.region.return_value = mock_region_fetcher
        mock_datafetcher_cls = MagicMock(return_value=mock_fetcher_instance)

        context = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "icenet_mp.ingestion.sources.lazy_argopy.DataFetcher",
                mock_datafetcher_cls,
                raising=False,
            )

            source = ArgoSource(
                context=context,
                area="20/30/0/40",
                crs="EPSG:6931",
                param=["TEMP"],
                resolution="500km",
                shape=(2, 2),
            )
            result = source.execute(argument=self.dates)

        assert len(result) == len(self.dates.dates)
        for field, date in zip(iter(result), sorted(self.dates.dates), strict=True):
            assert field.metadata("valid_datetime") == date.isoformat()
            assert field.to_numpy().shape == (2, 2)
        context.trace.assert_called_once()
