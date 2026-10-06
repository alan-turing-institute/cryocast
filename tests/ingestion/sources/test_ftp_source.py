"""Tests for the FTP data source."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from anemoi.datasets.create.recipe.dates import StartEndDates
from anemoi.datasets.dates.groups import GroupOfDates
from anemoi.utils.registry import Registry

from cryocast.ingestion.sources import FTPSource, register_sources


class TestFTPSource:
    """Test suite for FTPSource class."""

    mock_context: ClassVar[MagicMock] = MagicMock()
    date_range: ClassVar[StartEndDates] = StartEndDates(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 3),
        frequency=timedelta(days=1),
    )
    dates: ClassVar[GroupOfDates] = GroupOfDates(list(date_range), provider=date_range)

    def test_ftp_source_registration(self) -> None:
        """Test that FTPSource is properly registered."""
        # Mock source registry
        mock_registry: Registry = Registry("anemoi.datasets.create.sources")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "cryocast.ingestion.sources.source_registry",
                mock_registry,
            )
            assert "ftp" not in mock_registry.registered
            register_sources()
            assert "ftp" in mock_registry.registered
            assert mock_registry.lookup("ftp") == FTPSource

    def test_ftp_source_execute_basic(self) -> None:
        """Test basic FTP source execution with mocked FTP connection."""
        # Mock ftp.FTP
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        # Mock ftp.load_one
        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("cryocast.ingestion.sources.ftp.load_one", mock_load_one)

            # Execute
            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/file.nc",
                user="testuser",
                passwd="testpass",  # noqa: S106
            )
            source.execute(argument=self.dates)

            # Verify FTP session was created with correct credentials
            mock_ftp_class.assert_called_once_with("example.com", timeout=60.0)
            mock_ftp.login.assert_called_once_with(user="testuser", passwd="testpass")  # noqa: S106

            # Verify load_one was called for each date
            assert mock_load_one.call_count == 3

    def test_ftp_source_execute_anonymous_login(self) -> None:
        """Test FTP source with anonymous login (default)."""
        # Mock ftp.FTP
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        # Mock ftp.load_one
        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("cryocast.ingestion.sources.ftp.load_one", mock_load_one)

            # Execute without providing user/passwd
            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/file.nc",
            )
            source.execute(argument=self.dates)

            # Verify FTP session was created with correct credentials
            mock_ftp_class.assert_called_once_with("example.com", timeout=60.0)
            mock_ftp.login.assert_called_once_with(user="anonymous", passwd="")

            # Verify load_one was called for each date
            assert mock_load_one.call_count == 3

    def test_ftp_source_execute_file_download(self) -> None:
        """Test that URL parsing works correctly."""
        # Mock ftp.FTP
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        # Mock ftp.load_one
        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        # Mock ftp.MultiFieldList
        mock_multi_field_list = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("cryocast.ingestion.sources.ftp.load_one", mock_load_one)
            mp.setattr(
                "cryocast.ingestion.sources.ftp.MultiFieldList",
                mock_multi_field_list,
            )

            # Execute with a complex URL
            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://data.server.com/archive/datasets/file.nc",
            )
            source.execute(argument=self.dates)

            # Verify correct server was used
            mock_ftp_class.assert_called_once_with("data.server.com", timeout=60.0)

            # Verify directory change was attempted
            mock_ftp.cwd.assert_called_with("/archive/datasets")
            assert mock_ftp.cwd.call_count == 3

            # Verify retrbinary was called to download files
            assert mock_ftp.retrbinary.call_args.args[0] == "RETR file.nc"
            assert mock_ftp.retrbinary.call_count == 3

            # Verify MultiFieldList was created with all downloaded files
            mock_multi_field_list.assert_called_once()

    def test_ftp_source_execute_pattern_substitution(self) -> None:
        """Test that date patterns are substituted correctly."""
        # Mock ftp.FTP
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        # Mock ftp.load_one
        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("cryocast.ingestion.sources.ftp.load_one", mock_load_one)

            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/{date:strftime(%Y%m%d)}.nc",
            )
            source.execute(argument=self.dates)

            # Verify load_one was called with correct iso dates
            calls = mock_load_one.call_args_list
            assert len(calls) == 3

            # Check that iso dates are passed correctly
            assert "20200101.nc" in str(calls[0])
            assert "20200102.nc" in str(calls[1])
            assert "20200103.nc" in str(calls[2])

    def test_ftp_source_execute_with_load_one(self, tmp_path: Path) -> None:
        """Execute against the anemoi load_one by mocking only FTP retrbinary."""
        date = datetime(2020, 1, 1)
        real_dates: GroupOfDates = GroupOfDates([date], provider=self.date_range)

        ds = xr.Dataset(
            data_vars={
                "siconc": (("time", "lat", "lon"), np.array([[[0.1, 0.2], [0.3, 0.4]]]))
            },
            coords={
                "time": (
                    "time",
                    np.array([np.datetime64(date)]),
                    {"standard_name": "time"},
                ),
                "lat": (
                    "lat",
                    np.array([80.0, 85.0]),
                    {"units": "degrees_north", "standard_name": "latitude"},
                ),
                "lon": (
                    "lon",
                    np.array([0.0, 90.0]),
                    {"units": "degrees_east", "standard_name": "longitude"},
                ),
            },
        )
        nc_path = tmp_path / "source.nc"
        ds.to_netcdf(nc_path, engine="netcdf4")
        raw_bytes = nc_path.read_bytes()

        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None
        mock_ftp.retrbinary.side_effect = lambda _cmd, write_fn: write_fn(raw_bytes)

        context = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.ftp.FTP", mock_ftp_class)

            source = FTPSource(context=context, url="ftp://example.com/data/file.nc")
            result = source.execute(argument=real_dates)

        assert len(result) == 1
        field = next(iter(result))
        assert field.metadata("param") == "siconc"
        np.testing.assert_array_equal(field.to_numpy(), [[[0.1, 0.2], [0.3, 0.4]]])
        context.trace.assert_called_once()

    def test_ftp_source_custom_timeout(self) -> None:
        """Test that a custom timeout is passed through to the FTP connection."""
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None

        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("cryocast.ingestion.sources.ftp.load_one", mock_load_one)

            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/file.nc",
                timeout=5.0,
            )
            source.execute(argument=self.dates)

            mock_ftp_class.assert_called_once_with("example.com", timeout=5.0)

    def test_ftp_source_execute_logs_per_file_progress(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each file attempt logs a [count/total] progress line, success or failure."""
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None
        mock_ftp.retrbinary.side_effect = [None, TimeoutError("timed out"), None]

        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("cryocast.ingestion.sources.ftp.load_one", mock_load_one)

            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/file.nc",
            )
            with caplog.at_level("INFO"):
                source.execute(argument=self.dates)

        messages = [record.message for record in caplog.records]
        assert "Downloaded 'data/file.nc'." in messages
        assert any(
            message.startswith("Failed to download from 'data/file.nc':")
            for message in messages
        )

    def test_ftp_source_execute_handles_os_error(self) -> None:
        """A stalled/dropped connection (OSError, e.g. socket.timeout) is caught per-file."""
        mock_ftp_class = MagicMock()
        mock_ftp = MagicMock()
        mock_ftp_class.return_value.__enter__.return_value = mock_ftp
        mock_ftp_class.return_value.__exit__.return_value = None
        mock_ftp.retrbinary.side_effect = TimeoutError("timed out")

        mock_load_one = MagicMock()
        mock_load_one.return_value = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cryocast.ingestion.sources.ftp.FTP", mock_ftp_class)
            mp.setattr("cryocast.ingestion.sources.ftp.load_one", mock_load_one)

            source = FTPSource(
                context=self.mock_context,
                url=r"ftp://example.com/data/file.nc",
            )
            # Should not raise: the OSError is caught and logged per-file.
            result = source.execute(argument=self.dates)

            assert mock_load_one.call_count == 0
            assert len(result) == 0
