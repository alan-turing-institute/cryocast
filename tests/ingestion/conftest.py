import datetime
from typing import Any

import pytest


@pytest.fixture(scope="session")
def mock_data_status_flag() -> dict[str, dict[str, Any]]:
    """Fixture to create a mock status_flag dataset with a known land/inactive pattern.

    Cell (0,0): status_flag=1 (land bit set every timestep) -> land.
    Cell (0,1): status_flag=0 -> sea, active.
    Cell (1,0): status_flag=128 (inactive bit set every timestep) -> inactive.
    Cell (1,1): status_flag=0 -> sea, active.
    """
    dates = [
        datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i) for i in range(3)
    ]
    return {
        "coords": {
            "lat": {"dims": "lat", "attrs": {}, "data": [80.0, 85.0]},
            "lon": {"dims": "lon", "attrs": {}, "data": [0.0, 90.0]},
            "time": {"dims": ("time",), "attrs": {}, "data": dates},
        },
        "attrs": {},
        "dims": {"lat": 2, "lon": 2, "time": 3},
        "data_vars": {
            "status_flag": {
                "dims": ("time", "lat", "lon"),
                "attrs": {},
                "data": [
                    [[1, 0], [128, 0]],
                    [[1, 0], [128, 0]],
                    [[1, 0], [128, 0]],
                ],
            }
        },
    }
