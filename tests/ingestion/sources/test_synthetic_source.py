"""Deterministic, offline checks for synthetic Anemoi source data preparation."""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from anemoi.datasets.create.recipe.dates import StartEndDates
from anemoi.datasets.dates.groups import GroupOfDates

from cryocast.ingestion.sources.synthetic import SyntheticSource


@pytest.fixture
def fixture_frames() -> tuple[list[datetime], np.ndarray]:
    """Three distinguishable 4-by-4 frames on nonconsecutive dates."""
    dates = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 5),
    ]
    frames = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    return dates, frames


def test_source_initialization_passes_parsed_dates_and_parameters_to_generator(
    fixture_frames: tuple[list[datetime], np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward dynamics/grid settings and convert ISO timestamps to datetimes."""
    dates, frames = fixture_frames
    generate = MagicMock(return_value=SimpleNamespace(dates=dates, frames=frames))
    monkeypatch.setattr(
        "cryocast.ingestion.sources.synthetic.generate_default_dataset", generate
    )
    context = MagicMock()

    source = SyntheticSource(
        context,
        dynamics="grow-shrink",
        grid_size=4,
        trajectory_start_dates=["2024-01-01T00:00:00", "2024-02-01T00:00:00"],
        variable_name="sea_ice_fraction",
    )

    generate.assert_called_once_with(
        dynamics="grow-shrink",
        grid_size=4,
        start_dates=[datetime(2024, 1, 1), datetime(2024, 2, 1)],
    )
    assert source.context is context
    assert source.variable_name == "sea_ice_fraction"
    for date, expected_frame in zip(dates, frames, strict=True):
        np.testing.assert_array_equal(
            source.frames_by_date[date.date()], expected_frame
        )


def test_source_execute_filters_unavailable_dates_and_preserves_frame_alignment(
    fixture_frames: tuple[list[datetime], np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sort available dates and return a correctly aligned xarray input."""
    dates, frames = fixture_frames
    generator = MagicMock(return_value=SimpleNamespace(dates=dates, frames=frames))
    load_one = MagicMock(return_value=object())
    monkeypatch.setattr(
        "cryocast.ingestion.sources.synthetic.generate_default_dataset", generator
    )
    monkeypatch.setattr("cryocast.ingestion.sources.synthetic.load_one", load_one)
    context = MagicMock()
    source = SyntheticSource(
        context,
        dynamics="moving",
        grid_size=4,
        trajectory_start_dates=["2024-01-01T00:00:00"],
        variable_name="custom_sic",
    )

    result = source.execute(
        [
            datetime(2024, 1, 5),
            datetime(2024, 1, 4),  # Not available; must not create a synthetic frame.
            datetime(2024, 1, 1),
        ]
    )

    assert result is load_one.return_value
    load_one.assert_called_once()
    label, passed_context, date_strings, ds = load_one.call_args.args
    assert label == "🔬"
    assert passed_context is context
    assert date_strings == ["2024-01-01T00:00:00", "2024-01-05T00:00:00"]
    assert ds.sizes == {"time": 2, "x_pos": 4, "y_pos": 4}
    assert list(ds.data_vars) == ["custom_sic"]
    assert ds["custom_sic"].dims == ("time", "x_pos", "y_pos")
    np.testing.assert_array_equal(
        ds["custom_sic"].values, np.stack([frames[0], frames[2]])
    )
    np.testing.assert_array_equal(
        ds.time.values, np.asarray([dates[0], dates[2]], dtype="datetime64[ns]")
    )
    assert ds.time.attrs["standard_name"] == "time"
    assert ds.time.attrs["calendar"] == "standard"
    assert ds.lat.attrs["units"] == "degrees_north"
    assert ds.lon.attrs["units"] == "degrees_east"
    assert ds.lat.shape == (4, 4)
    assert ds.lon.shape == (4, 4)
    assert ds.lat.values[0, 0] == -90
    assert ds.lat.values[-1, 0] == 90
    assert ds.lon.values[0, 0] == -180
    assert ds.lon.values[0, -1] == 180


def test_source_accepts_anemoi_group_of_dates(
    fixture_frames: tuple[list[datetime], np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Anemoi date-group wrapper is accepted in addition to plain lists."""
    dates, frames = fixture_frames
    monkeypatch.setattr(
        "cryocast.ingestion.sources.synthetic.generate_default_dataset",
        MagicMock(return_value=SimpleNamespace(dates=dates, frames=frames)),
    )
    load_one = MagicMock(return_value=[])
    monkeypatch.setattr("cryocast.ingestion.sources.synthetic.load_one", load_one)
    source = SyntheticSource(
        MagicMock(),
        dynamics="moving",
        grid_size=4,
        trajectory_start_dates=["2024-01-01"],
    )
    schedule = StartEndDates(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 5),
        frequency=timedelta(days=1),
    )
    group = GroupOfDates(list(schedule), provider=schedule)

    source.execute(group)

    assert load_one.call_args.args[2] == [
        "2024-01-01T00:00:00",
        "2024-01-02T00:00:00",
        "2024-01-05T00:00:00",
    ]


@pytest.mark.parametrize("dynamics", ["moving", "grow-shrink"])
def test_real_synthetic_source_generates_deterministic_nonempty_trajectories(
    dynamics: str,
) -> None:
    """Exercise the actual trajectories generator in both supported modes."""
    first = SyntheticSource(
        MagicMock(),
        dynamics=dynamics,
        grid_size=32,
        trajectory_start_dates=["2024-01-01T00:00:00"],
    )
    second = SyntheticSource(
        MagicMock(),
        dynamics=dynamics,
        grid_size=32,
        trajectory_start_dates=["2024-01-01T00:00:00"],
    )

    assert len(first.frames_by_date) == 20
    assert first.frames_by_date.keys() == second.frames_by_date.keys()
    for date in first.frames_by_date:
        frame = first.frames_by_date[date]
        assert frame.shape == (32, 32)
        assert np.isfinite(frame).all()
        np.testing.assert_array_equal(frame, second.frames_by_date[date])
    assert np.any(first.frames_by_date[datetime(2024, 1, 1).date()] > 0)


@pytest.mark.parametrize(
    ("start_dates", "match"),
    [
        (["not-a-date"], "Invalid isoformat string"),
        (
            ["2024-01-15", "2024-01-01"],
            "start_dates must be strictly increasing",
        ),
    ],
)
def test_synthetic_source_fails_early_on_invalid_trajectory_dates(
    start_dates: list[str], match: str
) -> None:
    """Invalid date parsing or chronology must not produce incorrect frames."""
    with pytest.raises(ValueError, match=match):
        SyntheticSource(
            MagicMock(),
            dynamics="moving",
            grid_size=32,
            trajectory_start_dates=start_dates,
        )


def test_synthetic_source_rejects_unsupported_dynamics() -> None:
    """Unknown synthetic physics must not silently choose another generator."""
    with pytest.raises(ValueError, match="Unknown dynamics"):
        SyntheticSource(
            MagicMock(),
            dynamics="rotation",
            grid_size=32,
            trajectory_start_dates=["2024-01-01"],
        )


@pytest.mark.parametrize("dynamics", ["moving", "grow-shrink"])
def test_real_source_preserves_frames_through_anemoi_field_conversion(
    dynamics: str,
) -> None:
    """Exercise the actual generator, xarray conversion and Earthkit field adapter."""
    context = MagicMock()
    context.use_grib_paramid = False
    source = SyntheticSource(
        context,
        dynamics=dynamics,
        grid_size=32,
        trajectory_start_dates=["2024-01-01"],
        variable_name="custom_sic",
    )
    requested = [datetime(2024, 1, 3), datetime(2024, 1, 1)]

    fields = source.execute(requested)

    assert len(fields) == 2
    for field, date in zip(fields, sorted(requested), strict=True):
        assert field.metadata("valid_datetime") == date.isoformat()
        assert field.metadata("param") == "custom_sic"
        values = field.to_numpy()
        assert values.shape == (32, 32)
        np.testing.assert_array_equal(values, source.frames_by_date[date.date()])


def test_shipped_synthetic_recipe_initialises_in_a_fresh_interpreter(
    tmp_path: Path,
) -> None:
    """Registration must work without another test first rebuilding Anemoi's schema."""
    root = Path(__file__).resolve().parents[3]
    recipe_path = (
        root
        / "cryocast/config/data/datasets"
        / "samp_sicglobal_synthetic_5p625_2021_2024_24h_v2.yaml"
    )
    script = """
import sys
from anemoi.datasets.create.recipe import Recipe
from omegaconf import OmegaConf
from cryocast.ingestion import build_downloaders
from cryocast.ingestion.sources import register_sources

register_sources()
register_sources()
config = OmegaConf.create({
    "base_path": sys.argv[2],
    "data": {"datasets": OmegaConf.load(sys.argv[1])},
})
downloader, = build_downloaders(config)
source = downloader.recipe.model_dump()["input"]["synthetic"]
assert source["dynamics"] == "moving"
assert source["grid_size"] == 32
assert len(source["trajectory_start_dates"]) == 8
assert source["trajectory_start_dates"][0] == "2021-08-24T00:00:00"
assert source["trajectory_start_dates"][-1] == "2024-01-01T00:00:00"
assert not downloader.path_dataset.exists()
assert not downloader.path_dataset.parent.exists()
print("synthetic recipe registered and constructed")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(recipe_path), str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "synthetic recipe registered and constructed" in result.stdout
    assert not (tmp_path / "data").exists()
