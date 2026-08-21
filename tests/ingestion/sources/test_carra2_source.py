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


def test_carra2_execute_downloads_each_requested_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieve each requested valid time and combine the loaded field lists."""
    source = CARRA2Source(MagicMock(), variables=["2m_temperature"])
    dates = [
        datetime.datetime(2026, 1, 2, 3, 0, 0),
        datetime.datetime(2026, 1, 2, 6, 0, 0),
    ]
    client = MagicMock()
    client_factory = MagicMock(return_value=client)
    fields = [MagicMock(), MagicMock()]
    load_fields = MagicMock(side_effect=fields)
    combined_fields = MagicMock()
    combine = MagicMock(return_value=combined_fields)

    monkeypatch.setattr(
        "icenet_mp.ingestion.sources.carra2.cdsapi.Client", client_factory
    )
    monkeypatch.setattr("icenet_mp.ingestion.sources.carra2.from_source", load_fields)
    monkeypatch.setattr("icenet_mp.ingestion.sources.carra2.MultiFieldList", combine)

    result = source.execute(dates)

    assert result is combined_fields
    client_factory.assert_called_once_with()
    assert client.retrieve.call_count == 2
    first_dataset, first_request, first_target = client.retrieve.call_args_list[0].args
    second_dataset, second_request, second_target = client.retrieve.call_args_list[
        1
    ].args
    assert first_dataset == second_dataset == "reanalysis-pan-carra"
    assert first_request["time"] == ["03:00"]
    assert second_request["time"] == ["06:00"]
    assert first_target.endswith("carra2-202601020300.grib")
    assert second_target.endswith("carra2-202601020600.grib")
    load_fields.assert_any_call("file", first_target, stream=True, read_all=True)
    load_fields.assert_any_call("file", second_target, stream=True, read_all=True)
    combine.assert_called_once_with(fields)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"variables": []}, "At least one CARRA2 variable"),
        (
            {"variables": ["2m_temperature"], "data_format": "netcdf"},
            "data_format",
        ),
        (
            {"variables": ["2m_temperature"], "area": [81.0, 15.0, 76.0, 35.0, 40.0]},
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
