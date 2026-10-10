"""Regression coverage for interrupted, resumed and failed Anemoi downloads.

All lifecycle operations are mocked at the Anemoi boundary. No remote dataset,
CDS request, or pre-existing local data is required to exercise these cases.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zarr.errors import PathNotFoundError

from cryocast.ingestion.data_downloader import DataDownloader
from cryocast.types import AnemoiDatasetStatus


@pytest.fixture
def downloader(tmp_path: Path) -> DataDownloader:
    """Construct a minimal real recipe, storing files inside the test directory."""
    return DataDownloader(
        "sea-ice",
        tmp_path,
        {
            "name": "sea-ice",
            "dates": {
                "start": "2025-01-01",
                "end": "2025-01-03",
                "frequency": "24h",
            },
        },
    )


def _status(
    *, copying: bool = False, complete: bool = False, finalised: bool = False
) -> AnemoiDatasetStatus:
    """Describe a dataset without creating an Anemoi Zarr store."""
    return AnemoiDatasetStatus(
        copy_in_progress=copying,
        download_complete=complete,
        is_finalised=finalised,
    )


def test_temporary_artifacts_excludes_dataset_and_unrelated_files(
    downloader: DataDownloader,
) -> None:
    """Only files with the matching dataset stem, except the store, are artifacts."""
    parent = downloader.path_dataset.parent
    parent.mkdir(parents=True)
    downloader.path_dataset.mkdir()
    expected = {parent / "sea-ice.tmp", parent / "sea-ice.statistics"}
    for path in expected:
        path.write_text("partial")
    (parent / "another-sea-ice.tmp").write_text("other dataset")

    assert set(downloader.artifacts()) == expected


def test_create_new_dataset_downloads_without_inspecting(
    downloader: DataDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For an absent dataset, create delegates directly to download."""
    download = MagicMock()
    inspect = MagicMock()
    monkeypatch.setattr(downloader, "download", download)
    monkeypatch.setattr(downloader, "inspect", inspect)

    downloader.create()

    download.assert_called_once_with(overwrite=False)
    inspect.assert_not_called()


def test_create_overwrite_deletes_existing_data_then_downloads(
    downloader: DataDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit overwrite removes the old Zarr store before starting again."""
    downloader.path_dataset.mkdir(parents=True)
    (downloader.path_dataset / "stale_chunk").write_text("stale")
    download = MagicMock()
    status = MagicMock()
    monkeypatch.setattr(downloader, "download", download)
    monkeypatch.setattr(downloader, "check_status", status)

    downloader.create(overwrite=True)

    assert not downloader.path_dataset.exists()
    download.assert_called_once_with(overwrite=True)
    status.assert_not_called()


@pytest.mark.parametrize(
    ("dataset_status", "expected"),
    [
        (_status(copying=True), "busy"),
        (_status(complete=True, finalised=True), "ready"),
        (_status(complete=True, finalised=False), "ready"),
        (_status(complete=False), "resume"),
    ],
    ids=["another-writer", "already-finalised", "needs-finalise", "partial-download"],
)
def test_create_respects_existing_dataset_lifecycle(
    downloader: DataDownloader,
    monkeypatch: pytest.MonkeyPatch,
    dataset_status: AnemoiDatasetStatus,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Copying, complete, and partial datasets follow distinct safe paths."""
    downloader.path_dataset.mkdir(parents=True)
    check = MagicMock(return_value=dataset_status)
    download = MagicMock()
    finalise = MagicMock()
    inspect = MagicMock()
    monkeypatch.setattr(downloader, "check_status", check)
    monkeypatch.setattr(downloader, "download", download)
    monkeypatch.setattr(downloader, "finalise", finalise)
    monkeypatch.setattr(downloader, "inspect", inspect)

    with caplog.at_level(logging.WARNING):
        downloader.create(overwrite=False)

    check.assert_called_once_with()
    assert downloader.path_dataset.exists()
    if expected == "ready":
        finalise.assert_called_once_with(overwrite=False, status=dataset_status)
        inspect.assert_called_once_with()
        download.assert_not_called()
    elif expected == "resume":
        download.assert_called_once_with(overwrite=False)
        finalise.assert_not_called()
        inspect.assert_not_called()
    else:
        assert "being downloaded by another process" in caplog.text
        download.assert_not_called()
        finalise.assert_not_called()
        inspect.assert_not_called()


def test_create_raises_contextual_error_when_existing_status_unreadable(
    downloader: DataDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown store status never silently causes another writer to overwrite it."""
    downloader.path_dataset.mkdir(parents=True)
    check = MagicMock(side_effect=RuntimeError("metadata corrupted"))
    download = MagicMock()
    monkeypatch.setattr(downloader, "check_status", check)
    monkeypatch.setattr(downloader, "download", download)

    with pytest.raises(RuntimeError, match="could not be determined") as exc:
        downloader.create()

    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "metadata corrupted" in str(exc.value.__cause__)
    download.assert_not_called()


def test_create_raises_contextual_error_when_completed_store_cannot_be_inspected(
    downloader: DataDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed but corrupt store does not count as a successful download."""
    downloader.path_dataset.mkdir(parents=True)
    finalise = MagicMock()
    download = MagicMock()
    monkeypatch.setattr(
        downloader, "check_status", MagicMock(return_value=_status(complete=True))
    )
    monkeypatch.setattr(downloader, "finalise", finalise)
    monkeypatch.setattr(
        downloader, "inspect", MagicMock(side_effect=RuntimeError("corrupt store"))
    )
    monkeypatch.setattr(downloader, "download", download)

    with pytest.raises(RuntimeError, match="could not be inspected") as exc:
        downloader.create()

    assert "corrupt store" in str(exc.value.__cause__)
    finalise.assert_called_once()
    download.assert_not_called()


@pytest.mark.parametrize(
    ("dataset_status", "expected_finalise", "warning"),
    [
        (_status(complete=True), True, ""),
        (_status(copying=True), False, "still being copied"),
        (_status(), False, "not fully loaded"),
    ],
    ids=["fully-loaded", "copy-in-progress", "missing-chunks"],
)
def test_download_initialises_loads_then_finalises_only_when_ready(
    downloader: DataDownloader,
    monkeypatch: pytest.MonkeyPatch,
    dataset_status: AnemoiDatasetStatus,
    *,
    expected_finalise: bool,
    warning: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Never attempt finalisation while an import is incomplete or in progress."""
    order: list[str] = []
    preprocessing = MagicMock()
    preprocessing.process.side_effect = lambda *, overwrite: order.append(
        f"preprocess:{overwrite}"
    )
    initialise = MagicMock(side_effect=lambda: order.append("initialise"))
    load = MagicMock(side_effect=lambda: order.append("load"))

    def record_status() -> AnemoiDatasetStatus:
        order.append("status")
        return dataset_status

    check = MagicMock(side_effect=record_status)
    finalise = MagicMock(side_effect=lambda **_: order.append("finalise"))
    monkeypatch.setattr(downloader, "preprocessor", preprocessing)
    monkeypatch.setattr(downloader, "initialise", initialise)
    monkeypatch.setattr(downloader, "load_in_chunks", load)
    monkeypatch.setattr(downloader, "check_status", check)
    monkeypatch.setattr(downloader, "finalise", finalise)

    with caplog.at_level(logging.WARNING):
        downloader.download(overwrite=True)

    assert order == (
        ["preprocess:True", "initialise", "load", "status", "finalise"]
        if expected_finalise
        else ["preprocess:True", "initialise", "load", "status"]
    )
    if expected_finalise:
        finalise.assert_called_once_with(overwrite=True, status=dataset_status)
    else:
        finalise.assert_not_called()
        assert warning in caplog.text


def test_download_does_not_start_writing_when_preprocessing_fails(
    downloader: DataDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preprocessor failure halts loading rather than leaving a half-built dataset."""
    preprocessing = MagicMock()
    preprocessing.process.side_effect = OSError("input unavailable")
    initialise = MagicMock()
    load = MagicMock()
    monkeypatch.setattr(downloader, "preprocessor", preprocessing)
    monkeypatch.setattr(downloader, "initialise", initialise)
    monkeypatch.setattr(downloader, "load_in_chunks", load)

    with pytest.raises(OSError, match="input unavailable"):
        downloader.download(overwrite=False)

    initialise.assert_not_called()
    load.assert_not_called()


def test_download_propagates_unreadable_final_status(
    downloader: DataDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed post-load status check cannot trigger finalisation."""
    monkeypatch.setattr(downloader, "preprocessor", MagicMock())
    monkeypatch.setattr(downloader, "initialise", MagicMock())
    monkeypatch.setattr(downloader, "load_in_chunks", MagicMock())
    monkeypatch.setattr(
        downloader, "check_status", MagicMock(side_effect=RuntimeError("status failed"))
    )
    finalise = MagicMock()
    monkeypatch.setattr(downloader, "finalise", finalise)

    with pytest.raises(RuntimeError, match="status failed"):
        downloader.download(overwrite=False)

    finalise.assert_not_called()


@pytest.mark.parametrize("already_finalised", [False, True])
def test_finalise_runs_anemoi_only_when_needed_but_always_postprocesses(
    downloader: DataDownloader,
    monkeypatch: pytest.MonkeyPatch,
    *,
    already_finalised: bool,
) -> None:
    """Finalisation is idempotent while postprocessor masks are refreshed."""
    finalise_command = MagicMock()
    finalise_class = MagicMock(return_value=finalise_command)
    postprocessor = MagicMock()
    monkeypatch.setattr("cryocast.ingestion.data_downloader.Finalise", finalise_class)
    monkeypatch.setattr(downloader, "postprocessor", postprocessor)
    monkeypatch.setattr(downloader, "artifacts", MagicMock(return_value=[]))
    dataset_status = _status(complete=True, finalised=already_finalised)

    downloader.finalise(overwrite=True, status=dataset_status)

    if already_finalised:
        finalise_class.assert_not_called()
    else:
        finalise_command.run.assert_called_once()
        args = finalise_command.run.call_args.args[0]
        assert args.path == str(downloader.path_dataset)
        assert args.recipe is downloader.recipe
    postprocessor.process.assert_called_once_with(
        downloader.path_dataset, overwrite=True
    )


def test_finalise_attempts_cleanup_when_temporary_artifacts_exist(
    downloader: DataDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remove recipe artifacts after a successful finalise/postprocess pass."""
    artifact = downloader.path_dataset.parent / "sea-ice.tmp"
    artifacts = MagicMock(side_effect=[[artifact], []])
    cleanup = MagicMock()
    monkeypatch.setattr(downloader, "artifacts", artifacts)
    monkeypatch.setattr(downloader, "postprocessor", MagicMock())
    monkeypatch.setattr(
        "cryocast.ingestion.data_downloader.Cleanup",
        MagicMock(return_value=cleanup),
    )

    downloader.finalise(overwrite=False, status=_status(complete=True, finalised=True))

    cleanup.run.assert_called_once()
    assert cleanup.run.call_args.args[0].path == str(downloader.path_dataset)
    assert artifacts.call_count == 2


def test_initialise_existing_store_does_not_reinitialise(
    downloader: DataDownloader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing store is not rebuilt before resuming partial downloads."""
    downloader.path_dataset.mkdir(parents=True)
    init = MagicMock()
    monkeypatch.setattr("cryocast.ingestion.data_downloader.Init", init)

    downloader.initialise()

    init.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("missing recipe field"),
        FileNotFoundError("missing parent"),
        PathNotFoundError("missing Zarr path"),
    ],
)
def test_initialise_wraps_anemoi_store_failures_with_dataset_context(
    downloader: DataDownloader,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """Translate known Anemoi exceptions into actionable dataset errors."""
    init = MagicMock()
    init.run.side_effect = error
    monkeypatch.setattr(
        "cryocast.ingestion.data_downloader.Init", MagicMock(return_value=init)
    )

    with pytest.raises(
        RuntimeError, match="Failed to initialise dataset sea-ice"
    ) as exc:
        downloader.initialise()

    assert exc.value.__cause__ is error


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([True, False, True], "67% (2/3"),
        ([], "0% (0/0"),
        (None, "100% (all"),
    ],
)
def test_inspect_brief_reports_chunk_progress_without_loading_data(
    downloader: DataDownloader,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[bool] | None,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Report progress correctly even when the metadata has no build flags."""
    downloader.path_dataset.mkdir(parents=True)
    inspect = MagicMock()
    inspect._info.return_value = SimpleNamespace(
        path=str(downloader.path_dataset), build_flags=flags
    )
    monkeypatch.setattr(
        "cryocast.ingestion.data_downloader.InspectZarr",
        MagicMock(return_value=inspect),
    )

    with caplog.at_level(logging.INFO):
        downloader.inspect(verbose=False)

    assert expected in caplog.text
    inspect._info.assert_called_once_with(str(downloader.path_dataset))
    inspect.run.assert_not_called()


@pytest.mark.parametrize("error", [ValueError("metadata"), KeyError("build_flags")])
def test_inspect_brief_warns_when_metadata_unavailable(
    downloader: DataDownloader,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    """Optional inspection metadata failures should not abort a completed store."""
    downloader.path_dataset.mkdir(parents=True)
    inspect = MagicMock()
    inspect._info.side_effect = error
    monkeypatch.setattr(
        "cryocast.ingestion.data_downloader.InspectZarr",
        MagicMock(return_value=inspect),
    )

    with caplog.at_level(logging.WARNING):
        downloader.inspect()

    assert "Further dataset information unavailable" in caplog.text


def test_inspect_absent_store_fails_before_reading_metadata(
    downloader: DataDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing dataset path raises immediately without calling InspectZarr."""
    inspect = MagicMock()
    monkeypatch.setattr("cryocast.ingestion.data_downloader.InspectZarr", inspect)

    with pytest.raises(RuntimeError, match="Dataset sea-ice not found"):
        downloader.inspect()

    inspect.assert_not_called()


def test_integrity_check_skips_known_missing_indices_before_reading(
    downloader: DataDownloader,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Skip exact known-missing positions while reading each other frame."""
    dataset = MagicMock()
    dataset.missing = [1, 3]
    dataset.__len__.return_value = 5
    dataset.__getitem__.return_value = "ok"
    monkeypatch.setattr(
        "cryocast.ingestion.data_downloader.open_dataset",
        MagicMock(return_value=dataset),
    )

    with caplog.at_level(logging.INFO):
        downloader.integrity_check()

    assert [call.args[0] for call in dataset.__getitem__.call_args_list] == [0, 2, 4]
    assert "3/5 date(s) verified (2 known missing)" in caplog.text
