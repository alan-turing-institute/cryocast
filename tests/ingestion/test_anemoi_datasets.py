import datetime as dt
from pathlib import Path
from typing import Any

from anemoi.datasets.commands.inspect import InspectZarr

from tests.conftest import build_zarr


class TestAnemoiDatasetsRegressions:
    """Guards against regressions in anemoi-datasets.

    anemoi-datasets==0.5.40 introduced a bug in usage.misc._open() that passes the
    `options` key to the `path` parameter of ZarrStore.from_group(). This only becomes
    apparent when the `path` parameter is accessed, which happens when calling
    GriddedZarr.mutate(), for example on a dataset with missing dates.

    """

    def test_inspect_zarr_info_handles_dataset_with_missing_dates(
        self,
        tmp_path: Path,
        mock_data: dict[str, dict[str, Any]],
        dates_as_dt: tuple[dt.datetime, ...],
    ) -> None:
        """InspectZarr()._info() must not raise on a dataset with missing dates."""
        missing_date = dt.datetime(2020, 1, 6, 0, 0, 0)
        zarr_path = tmp_path / "regression.zarr"
        build_zarr(
            zarr_path,
            mock_data,
            full_dates=[*dates_as_dt, missing_date],
            missing_dates=[missing_date],
        )

        ds_info = InspectZarr()._info(str(zarr_path))

        assert ds_info.dataset is not None
