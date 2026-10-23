"""Finalise the SEAS5 trajectory recipe with local, synthetic GRIB responses."""

from importlib.resources import files
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import eccodes
import numpy as np
import pytest
import yaml
import zarr
from anemoi.datasets.create.sources.mars import retrieval
from anemoi.datasets.usage import analytics
from earthkit import data as ekd

from cryocast.ingestion.data_downloader import DataDownloader

DATASET_NAME = "demo-sicnorth-seas5-1p0-2024-2024-24h-v1"
STEPS = list(range(24, 169, 24))
VALUES = np.array([(np.arange(6) + day * 8) / 64 for day in range(1, 8)])


@pytest.fixture
def local_seas5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DataDownloader, list[dict[str, Any]]]:
    """Replace only CDS transport; keep actual recipe and finalisation code."""
    path = tmp_path / "forecast.grib"
    with path.open("wb") as stream:
        for lead, values in zip(STEPS, VALUES, strict=True):
            handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            try:
                metadata = {
                    "Ni": 3,
                    "Nj": 2,
                    "latitudeOfFirstGridPointInDegrees": 20,
                    "latitudeOfLastGridPointInDegrees": 19,
                    "longitudeOfFirstGridPointInDegrees": 0,
                    "longitudeOfLastGridPointInDegrees": 2,
                    "iDirectionIncrementInDegrees": 1,
                    "jDirectionIncrementInDegrees": 1,
                    "dataDate": 20240101,
                    "dataTime": 0,
                    "paramId": 31,
                    "forecastTime": lead,
                }
                for key, value in metadata.items():
                    eccodes.codes_set(handle, key, value)
                eccodes.codes_set_values(handle, values)
                eccodes.codes_write(handle, stream)
            finally:
                eccodes.codes_release(handle)

    fields = ekd.from_source("file", str(path))
    calls: list[dict[str, Any]] = []
    original = retrieval.from_source

    def local_response(source: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
        if source == "empty":
            return original(source, *args, **kwargs)
        assert source == "cds", "Unexpected external source in offline regression"
        dataset, request = args
        assert dataset == "seasonal-original-single-levels"
        assert request["date"] == "20240101"
        assert request["system"] == "51"
        assert request["param"] == "31.128"
        calls.append(dict(request))
        steps = request["step"]
        if isinstance(steps, str):
            steps = [int(value) for value in steps.split("/")]
        return fields.sel(step=steps)

    monkeypatch.setattr(retrieval, "from_source", local_response)
    # Anemoi submits analytics jobs even when collection is disabled. The
    # numerical integration does not need external telemetry or subprocesses.
    monkeypatch.setattr(analytics, "_executor", Mock())
    recipe_path = (
        files("cryocast.config")
        / "data/datasets/demo_sicnorth_seas5_1p0_2024_2024_24h_v1.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text())[DATASET_NAME]
    downloader = DataDownloader(DATASET_NAME, tmp_path, recipe)
    downloader.path_dataset.parent.mkdir(parents=True, exist_ok=True)
    return downloader, calls


def test_seas5_create_finalises_complete_trajectory(
    local_seas5: tuple[DataDownloader, list[dict[str, Any]]],
) -> None:
    """Write all seven leads, compute real statistics and reuse the finalised store."""
    downloader, calls = local_seas5
    downloader.create(overwrite=False)

    status = downloader.check_status()
    assert status.download_complete
    assert status.is_finalised
    dataset = zarr.open_group(str(downloader.path_dataset), mode="r")
    assert dataset.attrs["layout"] == "trajectories"
    assert dataset.attrs["dimensions"] == [
        "base_dates",
        "variables",
        "ensembles",
        "steps",
        "values",
    ]
    assert dataset["data"].shape == (1, 1, 1, 7, 6)
    np.testing.assert_array_equal(dataset["data"][:].reshape(7, 6), VALUES)
    np.testing.assert_array_equal(
        dataset["base_dates"][:], np.array(["2024-01-01"], dtype="datetime64[s]")
    )
    np.testing.assert_array_equal(
        dataset["steps"][:], np.array(STEPS, dtype="timedelta64[h]")
    )
    for statistic, expected in {
        "mean": VALUES.mean(),
        "stdev": VALUES.std(),
        "minimum": VALUES.min(),
        "maximum": VALUES.max(),
    }.items():
        np.testing.assert_allclose(dataset[statistic][:], [expected], rtol=1e-12)

    request_count = len(calls)
    assert request_count > 0
    downloader.create(overwrite=False)
    downloader.inspect(verbose=True)
    assert len(calls) == request_count, "Reopening must not download again"


def test_seas5_finalisation_rejects_truncated_statistics_envelope(
    local_seas5: tuple[DataDownloader, list[dict[str, Any]]],
) -> None:
    """Reproduce the reviewer's failure when the last forecast lead is excluded."""
    downloader, _ = local_seas5
    downloader.recipe.statistics.end = "2024-01-07T00:00:00"
    with pytest.raises(ValueError, match="no trajectory falls entirely within"):
        downloader.create(overwrite=False)
    status = downloader.check_status()
    assert status.download_complete
    assert not status.is_finalised
