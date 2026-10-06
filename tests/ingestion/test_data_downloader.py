import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import anemoi.datasets.create.tasks
import numpy as np
import pytest

from icenet_mp.ingestion.data_downloader import DataDownloader
from icenet_mp.types import AnemoiDatasetStatus, AnemoiInspectArgs
from icenet_mp.utils import mask_dir
from tests.conftest import build_zarr


@dataclass
class MockDsInfo:
    """Configurable stand-in for the object returned by InspectZarr()._info()."""

    copy_in_progress: bool = False
    statistics_ready: bool = False
    dataset: object = field(default_factory=object)
    build_flags: list[bool] | None = None
    statistics_started: str | None = None


class MockInspectZarr:
    """Stand-in for InspectZarr; _info() and run() are pre-configured or raise."""

    def __init__(self) -> None:
        """Initialise a MockInspectZarr."""
        self.ds_info: MockDsInfo | BaseException = MockDsInfo()
        self.run_error: BaseException | None = None
        self.run_args: AnemoiInspectArgs | None = None

    def _info(self, _: str) -> MockDsInfo:
        if isinstance(self.ds_info, BaseException):
            raise self.ds_info
        return self.ds_info

    def run(self, args: AnemoiInspectArgs) -> None:
        self.run_args = args
        if self.run_error is not None:
            raise self.run_error


def _make_data_downloader(tmp_path: Path, name: str) -> DataDownloader:
    """Build a DataDownloader for `name` from a minimal dataset config."""
    postprocessor_target = {
        "samp-sic-ssmis": "icenet_mp.ingestion.postprocessors.StatusFlagMaskGenerator",
        "samp-sic-synthetic": "icenet_mp.ingestion.postprocessors.SyntheticMaskGenerator",
    }.get(name)
    anemoi_config: dict[str, Any] = {
        "name": name,
        "dates": {"start": "2020-01-01", "end": "2020-01-31", "frequency": "24h"},
    }
    if postprocessor_target:
        anemoi_config["postprocessors"] = {
            "mask_generator": {"_target_": postprocessor_target}
        }
    return DataDownloader(name, tmp_path, anemoi_config)


@pytest.fixture
def mock_data_downloader(tmp_path: Path) -> DataDownloader:
    """A DataDownloader built from a minimal dataset config."""
    return _make_data_downloader(tmp_path, "test")


@pytest.fixture
def mock_data_downloader_ssmis(tmp_path: Path) -> DataDownloader:
    """A DataDownloader for an SSMIS dataset, used to test mask generation."""
    return _make_data_downloader(tmp_path, "samp-sic-ssmis")


@pytest.fixture
def mock_data_downloader_synthetic(tmp_path: Path) -> DataDownloader:
    """A DataDownloader for synthetic data, used to test dummy mask generation."""
    return _make_data_downloader(tmp_path, "samp-sic-synthetic")


@pytest.fixture
def mock_dispatcher(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch anemoi TaskDispatcher to return a MagicMock for testing."""
    dispatcher = MagicMock()

    def mock_init(creator: object) -> MagicMock:
        dispatcher.creator = creator
        return dispatcher

    monkeypatch.setattr(anemoi.datasets.create.tasks, "TaskDispatcher", mock_init)
    return dispatcher


@pytest.fixture
def mock_inspect_zarr(monkeypatch: pytest.MonkeyPatch) -> MockInspectZarr:
    """Patch InspectZarr with MockInspectZarr; returns the instance for per-test configuration."""
    instance = MockInspectZarr()
    monkeypatch.setattr(
        "icenet_mp.ingestion.data_downloader.InspectZarr",
        lambda: instance,
    )
    return instance


class TestDataDownloader:
    """Verify that DataDownloader delivers the right recipe and path to the anemoi creator."""

    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2020, 1, 31)
    frequency = datetime.timedelta(hours=24)

    def test_initialise_recipe_and_path_reach_creator(
        self,
        mock_data_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
    ) -> None:
        mock_data_downloader.initialise()
        mock_dispatcher.task_init.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_data_downloader.path_dataset)

    def test_finalise_recipe_and_path_reach_creator(
        self,
        mock_data_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(mock_data_downloader, "postprocessor", MagicMock())
        status = AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=False
        )
        mock_data_downloader.finalise(overwrite=False, status=status)
        mock_dispatcher.task_finalise.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_data_downloader.path_dataset)

    def test_load_in_chunks_recipe_and_path_reach_creator(
        self,
        mock_data_downloader: DataDownloader,
        mock_dispatcher: MagicMock,
    ) -> None:
        mock_data_downloader.load_in_chunks()
        mock_dispatcher.task_load.assert_called_once()
        assert mock_dispatcher.creator.recipe.dates.start == self.start_date
        assert mock_dispatcher.creator.recipe.dates.end == self.end_date
        assert mock_dispatcher.creator.recipe.dates.frequency == self.frequency
        assert mock_dispatcher.creator.path == str(mock_data_downloader.path_dataset)

    def test_copy_in_progress_not_complete(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(copy_in_progress=True)
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=True, download_complete=False, is_finalised=False
        )

    def test_dataset_none_not_complete(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(dataset=None)
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=False, is_finalised=False
        )

    def test_build_flags_all_true_statistics_ready(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(
            statistics_ready=True, build_flags=[True, True, True]
        )
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=True
        )

    def test_build_flags_all_true_statistics_not_ready(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(build_flags=[True, True, True])
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=False
        )

    def test_build_flags_empty_not_complete(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(build_flags=[])
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=False, is_finalised=False
        )

    def test_build_flags_partial_not_complete(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(build_flags=[True, False, True])
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=False, is_finalised=False
        )

    def test_no_build_flags_statistics_ready(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(statistics_ready=True)
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=True
        )

    def test_no_build_flags_statistics_started_not_finalised(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(statistics_started="2020-01-31")
        assert mock_data_downloader.check_status() == AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=False
        )

    def test_no_build_flags_no_statistics_raises(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = MockDsInfo(statistics_started=None)
        with pytest.raises(RuntimeError, match="Unable to determine readiness"):
            mock_data_downloader.check_status()

    def test_file_not_found_raises(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = FileNotFoundError("no dataset at path")
        with pytest.raises(RuntimeError, match="Unable to get status"):
            mock_data_downloader.check_status()

    def test_attribute_error_raises(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
    ) -> None:
        mock_inspect_zarr.ds_info = AttributeError("unexpected ds_info shape")
        with pytest.raises(RuntimeError, match="Unable to get status"):
            mock_data_downloader.check_status()

    def test_inspect_verbose_calls_inspect_zarr_run(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """inspect(verbose=True) reaches the real InspectZarr().run() call."""
        monkeypatch.setattr(mock_data_downloader, "integrity_check", MagicMock())
        mock_data_downloader.path_dataset.mkdir(parents=True)
        mock_data_downloader.inspect(verbose=True)
        assert mock_inspect_zarr.run_args is not None
        args = mock_inspect_zarr.run_args
        assert args.path == str(mock_data_downloader.path_dataset)
        assert args.detailed is True
        assert args.progress is False
        assert args.statistics is False
        assert args.size is True

    def test_inspect_verbose_calls_integrity_check(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """inspect(verbose=True) also runs the sequential-read integrity check."""
        mock_integrity_check = MagicMock()
        monkeypatch.setattr(
            mock_data_downloader, "integrity_check", mock_integrity_check
        )
        mock_data_downloader.path_dataset.mkdir(parents=True)
        mock_data_downloader.inspect(verbose=True)
        mock_integrity_check.assert_called_once()

    def test_inspect_verbose_suppresses_value_error(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """inspect(verbose=True) does not raise when InspectZarr().run() raises ValueError."""
        monkeypatch.setattr(mock_data_downloader, "integrity_check", MagicMock())
        mock_data_downloader.path_dataset.mkdir(parents=True)
        mock_inspect_zarr.run_error = ValueError("statistics not available")
        with caplog.at_level(logging.WARNING):
            mock_data_downloader.inspect(verbose=True)
        assert "Further dataset information unavailable" in caplog.text

    def test_inspect_verbose_wraps_file_not_found_error(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """inspect(verbose=True) wraps FileNotFoundError from InspectZarr().run() as RuntimeError."""
        monkeypatch.setattr(mock_data_downloader, "integrity_check", MagicMock())
        mock_data_downloader.path_dataset.mkdir(parents=True)
        mock_inspect_zarr.run_error = FileNotFoundError("no dataset at path")
        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            mock_data_downloader.inspect(verbose=True)

    def test_inspect_verbose_propagates_integrity_check_runtime_error(
        self,
        mock_data_downloader: DataDownloader,
        mock_inspect_zarr: MockInspectZarr,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A corrupt-chunk RuntimeError from integrity_check() is not swallowed."""
        monkeypatch.setattr(
            mock_data_downloader,
            "integrity_check",
            MagicMock(side_effect=RuntimeError("Zarr integrity check failed")),
        )
        mock_data_downloader.path_dataset.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="Zarr integrity check failed"):
            mock_data_downloader.inspect(verbose=True)

    def test_finalise_suppresses_cleanup_value_error_and_logs_residual(
        self,
        mock_data_downloader: DataDownloader,
        mock_data: dict[str, dict[str, Any]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test the ValueError() suppression in finalise() by providing a dataset with no recipe."""
        build_zarr(mock_data_downloader.path_dataset, mock_data)
        residual = mock_data_downloader.path_dataset.parent / (
            f"{mock_data_downloader.path_dataset.stem}.tmp"
        )
        residual.mkdir()
        (residual / "dummy.txt").write_text("x")

        status = AnemoiDatasetStatus(
            copy_in_progress=False, download_complete=True, is_finalised=True
        )
        with caplog.at_level(logging.WARNING):
            mock_data_downloader.finalise(overwrite=False, status=status)

        # Cleanup's real ValueError is suppressed, so the residual artifact remains
        # and is logged as a warning rather than raising.
        # The artifa
        assert residual.exists()
        assert "Residual artifacts" in caplog.text

    @pytest.mark.parametrize("overwrite", [False, True], ids=["create", "recreate"])
    def test_postprocessor_process_creates_or_recreates_land_and_active_masks(
        self,
        mock_data_downloader_ssmis: DataDownloader,
        mock_data_status_flag: dict[str, dict[str, Any]],
        tmp_path: Path,
        *,
        overwrite: bool,
    ) -> None:
        build_zarr(mock_data_downloader_ssmis.path_dataset, mock_data_status_flag)
        path_masks = mask_dir(tmp_path, mock_data_downloader_ssmis.name)
        land_mask_path = path_masks / "land_mask.npy"
        active_mask_path = path_masks / "active_mask.npy"
        if overwrite:
            # Pre-seed masks with sentinel values to prove overwrite=True recomputes them.
            path_masks.mkdir(parents=True)
            np.save(land_mask_path, np.full((2, 2), fill_value=9))
            np.save(active_mask_path, np.full((2, 2), fill_value=9))

        mock_data_downloader_ssmis.postprocessor.process(
            mock_data_downloader_ssmis.path_dataset, overwrite=overwrite
        )

        np.testing.assert_array_equal(np.load(land_mask_path), [[0, 1], [1, 1]])
        np.testing.assert_array_equal(np.load(active_mask_path), [[0, 1], [0, 1]])

    def test_postprocessor_process_skips_missing_leading_date(
        self,
        mock_data_downloader_ssmis: DataDownloader,
        mock_data_status_flag: dict[str, dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """A missing index 0 must not crash the status_flag conversion (regression, #379)."""
        dates = mock_data_status_flag["coords"]["time"]["data"]
        data_with_gap = dict(mock_data_status_flag)
        data_with_gap["coords"] = {
            **mock_data_status_flag["coords"],
            "time": {**mock_data_status_flag["coords"]["time"], "data": dates[1:]},
        }
        data_with_gap["data_vars"] = {
            "status_flag": {
                **mock_data_status_flag["data_vars"]["status_flag"],
                "data": mock_data_status_flag["data_vars"]["status_flag"]["data"][1:],
            }
        }
        build_zarr(
            mock_data_downloader_ssmis.path_dataset,
            data_with_gap,
            full_dates=dates,
            missing_dates=[dates[0]],
        )

        mock_data_downloader_ssmis.postprocessor.process(
            mock_data_downloader_ssmis.path_dataset, overwrite=False
        )

        path_masks = mask_dir(tmp_path, mock_data_downloader_ssmis.name)
        np.testing.assert_array_equal(
            np.load(path_masks / "land_mask.npy"), [[0, 1], [1, 1]]
        )
        np.testing.assert_array_equal(
            np.load(path_masks / "active_mask.npy"), [[0, 1], [0, 1]]
        )

    def test_postprocessor_process_creates_all_active_synthetic_masks(
        self,
        mock_data_downloader_synthetic: DataDownloader,
        mock_data: dict[str, dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        build_zarr(mock_data_downloader_synthetic.path_dataset, mock_data)

        mock_data_downloader_synthetic.postprocessor.process(
            mock_data_downloader_synthetic.path_dataset, overwrite=False
        )

        expected = np.ones((2, 2), dtype=np.uint8)
        path_masks = mask_dir(tmp_path, mock_data_downloader_synthetic.name)
        np.testing.assert_array_equal(np.load(path_masks / "land_mask.npy"), expected)
        np.testing.assert_array_equal(np.load(path_masks / "active_mask.npy"), expected)

    def test_postprocessor_process_noop_for_default_dataset(
        self, mock_data_downloader: DataDownloader, tmp_path: Path
    ) -> None:
        """No postprocessor is configured for this dataset, so process() is a no-op."""
        mock_data_downloader.postprocessor.process(
            mock_data_downloader.path_dataset, overwrite=False
        )
        assert not mask_dir(tmp_path, mock_data_downloader.name).exists()

    def test_postprocessor_process_skips_when_masks_already_exist(
        self, mock_data_downloader_ssmis: DataDownloader, tmp_path: Path
    ) -> None:
        path_masks = mask_dir(tmp_path, mock_data_downloader_ssmis.name)
        path_masks.mkdir(parents=True)
        land_mask_path = path_masks / "land_mask.npy"
        active_mask_path = path_masks / "active_mask.npy"
        np.save(land_mask_path, np.zeros((2, 2)))
        np.save(active_mask_path, np.ones((2, 2)))

        # No dataset exists on disk; a real open_dataset() call would raise if reached.
        mock_data_downloader_ssmis.postprocessor.process(
            mock_data_downloader_ssmis.path_dataset, overwrite=False
        )

        np.testing.assert_array_equal(np.load(land_mask_path), np.zeros((2, 2)))
        np.testing.assert_array_equal(np.load(active_mask_path), np.ones((2, 2)))

    def test_integrity_check_passes_on_clean_dataset(
        self,
        mock_data_downloader: DataDownloader,
        mock_data: dict[str, dict[str, Any]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """integrity_check() does not raise for a dataset with no missing dates."""
        build_zarr(mock_data_downloader.path_dataset, mock_data)
        with caplog.at_level(logging.INFO):
            mock_data_downloader.integrity_check()
        assert "✅ Integrity check" in caplog.text

    def test_integrity_check_skips_missing_dates(
        self,
        mock_data_downloader: DataDownloader,
        mock_data_missing_dates: dict[str, dict[str, Any]],
        dates_as_dt: tuple[datetime.datetime, ...],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """integrity_check() skips known-missing dates instead of raising MissingDateError."""
        build_zarr(
            mock_data_downloader.path_dataset,
            mock_data_missing_dates,
            full_dates=list(dates_as_dt),
            missing_dates=[dates_as_dt[1], dates_as_dt[3]],
        )
        with caplog.at_level(logging.INFO):
            mock_data_downloader.integrity_check()
        assert (
            "Integrity check for test: 3/5 date(s) verified (2 known missing)."
            in caplog.text
        )

    def test_integrity_check_raises_on_corrupt_chunk(
        self,
        mock_data_downloader: DataDownloader,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """integrity_check() raises RuntimeError if a non-missing chunk cannot be read."""

        def _getitem(idx: int) -> np.ndarray:
            if idx == 1:
                msg = "corrupt"
                raise OSError(msg)
            return np.zeros((2, 2))

        mock_dataset = MagicMock(missing=set())
        mock_dataset.__len__.return_value = 3
        mock_dataset.__getitem__.side_effect = _getitem
        monkeypatch.setattr(
            "icenet_mp.ingestion.data_downloader.open_dataset",
            lambda _: mock_dataset,
        )
        with pytest.raises(
            RuntimeError, match="Integrity check for test: date 1 unreadable"
        ):
            mock_data_downloader.integrity_check()
