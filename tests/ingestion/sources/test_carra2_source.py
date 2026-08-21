import datetime
from unittest.mock import MagicMock

import pytest
from anemoi.datasets.create.sources import source_registry

from icenet_mp.ingestion.sources import CARRA2Source, register_sources


def test_carra2_builds_request_for_exact_analysis_time() -> None:
    """Build a CARRA2 request for the exact requested analysis time."""
    source = CARRA2Source(
        MagicMock(),
        variables=["temperature"],
        level_type="pressure_levels",
        level_location=["850"],
        area=[81.0, 15.0, 76.0, 35.0],
    )

    request = source._request_for_date(datetime.datetime(2001, 2, 28, 12, 0, 0))

    assert request == {
        "level_type": "pressure_levels",
        "variable": ["temperature"],
        "product_type": "analysis",
        "time": ["12:00"],
        "year": ["2001"],
        "month": ["02"],
        "day": ["28"],
        "data_format": "grib",
        "level_location": ["850"],
        "area": [81.0, 15.0, 76.0, 35.0],
    }


def test_carra2_request_omits_optional_fields_when_unset() -> None:
    """Omit optional request fields when they are not configured."""
    source = CARRA2Source(MagicMock(), variables=["2m_temperature"])

    request = source._request_for_date(datetime.datetime(2026, 1, 2, 3, 0, 0))

    assert "level_location" not in request
    assert "area" not in request
    assert request["time"] == ["03:00"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"variables": []}, "At least one CARRA2 variable"),
        (
            {"variables": ["2m_temperature"], "data_format": "csv"},
            "data_format",
        ),
        (
            {"variables": ["2m_temperature"], "area": [81.0, 15.0, 76.0]},
            "area",
        ),
    ],
)
def test_carra2_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    """Reject invalid CARRA2 source configurations."""
    with pytest.raises(ValueError, match=message):
        CARRA2Source(MagicMock(), **kwargs)  # type: ignore[arg-type]


def test_carra2_source_is_registered_with_anemoi() -> None:
    """Register CARRA2 with the Anemoi source registry."""
    register_sources()

    assert "carra2" in source_registry.registered
    assert source_registry.lookup("carra2") is CARRA2Source
