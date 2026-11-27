"""Regression tests for the CS2SMOS sea-ice thickness (SIT) dataset integration."""

from collections.abc import Callable
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path

import pytest
from earthkit.data.utils.patterns import Pattern
from omegaconf import DictConfig, OmegaConf

from cryocast.ingestion.downloaders import build_downloaders
from cryocast.ingestion.filters import register_filters
from cryocast.ingestion.sources import register_sources

SIT_GROUP = "sit-cs2smos"
SIT_DATASETS = {
    "sample_north": "samp-sitnorth-cs2smos-25p0km-2020-2024-24h-v1",
    "full_north": "full-sitnorth-cs2smos-25p0km-2010-2025-24h-v1",
}
# CS2SMOS v2.06 has no files between these centre dates (inclusive) in any year
SUMMER_GAP = ((4, 13), (10, 17))
# Freeze-season days with no file upstream
KNOWN_GAPS = [("2025-02-07", "2025-02-13"), ("2025-04-08", "2025-04-08")]
# Date ranges excluded from every sample dataset, to match the sample split
SAMPLE_EXCLUDED = [
    ("2022-01-01", "2022-06-30"),
    ("2022-08-01", "2022-12-31"),
    ("2023-02-01", "2023-12-31"),
    ("2024-02-01", "2024-06-30"),
]
MODEL_CONFIGS = sorted(
    p.stem
    for p in (Path(str(files("cryocast.config"))) / "model").glob("*.yaml")
    if not p.name.endswith("local.yaml")
)


def _day(value: object) -> datetime:
    """Parse a config date or datetime into a midnight datetime."""
    return datetime.fromisoformat(str(value).split("T", maxsplit=1)[0])


def _days(start: object, end: object) -> list[datetime]:
    day, last, days = _day(start), _day(end), []
    while day <= last:
        days.append(day)
        day += timedelta(days=1)
    return days


def _missing_days(dates: DictConfig) -> set[datetime]:
    """Expand a dataset's missing dates into a set of days."""
    missing: set[datetime] = set()
    for entry in dates.missing or []:
        if isinstance(entry, str):
            missing.add(_day(entry))
        else:
            missing.update(_days(entry.start, entry.end))
    return missing


class TestSeaIceThicknessConfigs:
    """Tests for the SIT datasets, variables and allow_missing configs."""

    @pytest.fixture(params=sorted(SIT_DATASETS))
    def data_config(
        self, request: pytest.FixtureRequest, compose_config: Callable[..., DictConfig]
    ) -> tuple[str, DictConfig]:
        return request.param, compose_config(
            "sample", overrides=[f"data={request.param}"]
        )

    def test_sit_dataset_is_in_northern_data_group(
        self, data_config: tuple[str, DictConfig]
    ) -> None:
        data_name, config = data_config
        dataset = config.data.datasets[SIT_DATASETS[data_name]]
        assert dataset.group_as == SIT_GROUP

    @pytest.mark.parametrize(
        ("date", "expected_path"),
        [
            # File names are the 7-day window around the centre date; the folder is
            # the month of the centre date, even when the window starts the month before
            (
                datetime(2020, 1, 1, 12),
                "2020/01/W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_20191229_20200104_r_v206_01_l4sit.nc",
            ),
            (
                datetime(2020, 11, 1, 12),
                "2020/11/W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_20201029_20201104_r_v206_01_l4sit.nc",
            ),
            (
                datetime(2020, 2, 29, 12),
                "2020/02/W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_20200226_20200303_r_v206_01_l4sit.nc",
            ),
        ],
        ids=["year-boundary", "month-boundary", "leap-day"],
    )
    def test_ftp_url_resolves_to_awi_filename(
        self, data_config: tuple[str, DictConfig], date: datetime, expected_path: str
    ) -> None:
        data_name, config = data_config
        dataset = config.data.datasets[SIT_DATASETS[data_name]]
        resolved = Pattern(dataset.input.pipe[0].ftp.url).substitute(
            date=date, allow_extra=True
        )
        assert resolved == (
            "ftp://ftp.awi.de/sea_ice/product/cryosat2_smos/v206/nh/" + expected_path
        )

    def test_recipe_validates_with_anemoi(
        self, data_config: tuple[str, DictConfig], tmp_path: Path
    ) -> None:
        """The dataset recipe must be accepted by anemoi's Recipe model."""
        data_name, config = data_config
        dataset_name = SIT_DATASETS[data_name]
        # Filters must be registered before sources rebuild the Recipe schema
        register_filters()
        register_sources()
        recipe_config = OmegaConf.create(
            {
                "base_path": str(tmp_path),
                "data": {
                    "datasets": {
                        dataset_name: OmegaConf.to_container(
                            config.data.datasets[dataset_name]
                        )
                    }
                },
            }
        )
        (downloader,) = build_downloaders(recipe_config)
        assert downloader.name == dataset_name

    def test_missing_dates_match_upstream_gaps(
        self, data_config: tuple[str, DictConfig]
    ) -> None:
        """Missing dates are exactly the summer gaps, known gaps and sample exclusions.

        A date missing from this list but absent upstream breaks the build, and a
        listed date that has a file upstream silently throws data away.
        """
        data_name, config = data_config
        dates = config.data.datasets[SIT_DATASETS[data_name]].dates
        excluded = KNOWN_GAPS + (SAMPLE_EXCLUDED if data_name == "sample_north" else [])
        (gap_start_month, gap_start_day), (gap_end_month, gap_end_day) = SUMMER_GAP
        expected = {
            day
            for day in _days(dates.start, dates.end)
            if (gap_start_month, gap_start_day)
            <= (day.month, day.day)
            <= (gap_end_month, gap_end_day)
            or any(_day(start) <= day <= _day(end) for start, end in excluded)
        }
        assert _missing_days(dates) == expected

    def test_sample_matches_other_sample_datasets(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        """Same date range and excluded ranges as the other northern sample datasets."""
        config = compose_config("sample", overrides=["data=sample_north"])
        sit = config.data.datasets[SIT_DATASETS["sample_north"]].dates
        sic = config.data.datasets["samp-sicnorth-osisaf-25p0km-2020-2024-24h-v1"].dates
        assert (_day(sit.start), _day(sit.end)) == (_day(sic.start), _day(sic.end))
        missing = _missing_days(sit)
        for start, end in SAMPLE_EXCLUDED:
            assert set(_days(start, end)) <= missing

    def test_input_groups_have_encoders(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        config = compose_config(
            "sample",
            overrides=[
                "data=sample_north",
                "variables=sic-osisaf-sit-cs2smos",
                "data/allow_missing=sit-cs2smos",
            ],
        )
        assert SIT_GROUP in config.variables.input
        assert list(config.data.allow_missing.groups) == [SIT_GROUP]
        for group_name in config.variables.input:
            assert group_name in config.model.encoders

    def test_default_variables_do_not_use_sit(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        """Adding the dataset to sample_north must not change the default inputs."""
        config = compose_config("sample", overrides=["data=sample_north"])
        assert SIT_GROUP not in config.variables.input

    @pytest.mark.parametrize("model_name", MODEL_CONFIGS)
    def test_models_with_sic_encoders_have_sit_encoders(
        self, compose_config: Callable[..., DictConfig], model_name: str
    ) -> None:
        model = compose_config("sample", overrides=[f"model={model_name}"]).model
        encoders = model.get("encoders", {})
        if "sic-ssmis" in encoders:
            assert encoders[SIT_GROUP] == encoders["sic-ssmis"]
        else:
            assert SIT_GROUP not in encoders
