"""Regression tests for the CS2SMOS sea-ice thickness (SIT) dataset integration."""

from collections.abc import Callable
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path

import pytest
from earthkit.data.utils.patterns import Pattern
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

from cryocast.ingestion.downloaders import build_downloaders
from cryocast.ingestion.filters import register_filters
from cryocast.ingestion.sources import register_sources

SIC_DATASET = "samp-sicnorth-ssmis-25p0km-2019-2023-24h-v1"
SIT_DATASET = "samp-sitnorth-cs2smos-25p0km-2019-2023-24h-v1"
SAMPLE_DATASETS = [SIC_DATASET, SIT_DATASET]
SIC_GROUP = "sic-ssmis"
SIT_GROUP = "sit-cs2smos"
FREEZE_SEASON_MONTHS = {11, 12, 1, 2, 3}
MODEL_CONFIGS = sorted(
    p.stem for p in (Path(str(files("cryocast.config"))) / "model").glob("*.yaml")
)


def _day(value: object) -> datetime:
    """Parse a config date or datetime into a midnight datetime."""
    return datetime.fromisoformat(str(value).split("T", maxsplit=1)[0])


def _missing_days(dates: DictConfig) -> tuple[set[datetime], set[datetime]]:
    """Expand a dataset's missing dates into (single missing days, days in ranges)."""
    single: set[datetime] = set()
    ranged: set[datetime] = set()
    for entry in dates.missing or []:
        if isinstance(entry, str):
            single.add(_day(entry))
        else:
            day, end = _day(entry.start), _day(entry.end)
            while day <= end:
                ranged.add(day)
                day += timedelta(days=1)
    return single, ranged


class TestSeaIceThicknessConfigs:
    """Tests for the SIT datasets, data group, split and top-level config."""

    @pytest.fixture
    def sit_config(self, compose_config: Callable[..., DictConfig]) -> DictConfig:
        return compose_config("sample_north_sit")

    def test_data_group_pairs_sic_and_sit(self, sit_config: DictConfig) -> None:
        datasets = sit_config.data.datasets
        assert sorted(datasets) == sorted(SAMPLE_DATASETS)
        assert datasets[SIC_DATASET].group_as == SIC_GROUP
        assert datasets[SIT_DATASET].group_as == SIT_GROUP

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
        self, sit_config: DictConfig, date: datetime, expected_path: str
    ) -> None:
        url = sit_config.data.datasets[SIT_DATASET].input.pipe[0].ftp.url
        resolved = Pattern(url).substitute(date=date, allow_extra=True)
        assert resolved == (
            "ftp://ftp.awi.de/sea_ice/product/cryosat2_smos/v206/nh/" + expected_path
        )

    @pytest.mark.parametrize("dataset_name", SAMPLE_DATASETS)
    def test_recipe_validates_with_anemoi(
        self, sit_config: DictConfig, tmp_path: Path, dataset_name: str
    ) -> None:
        """The dataset recipe must be accepted by anemoi's Recipe model."""
        # Filters must be registered before sources rebuild the Recipe schema
        register_filters()
        register_sources()
        config = OmegaConf.create(
            {
                "base_path": str(tmp_path),
                "data": {
                    "datasets": {
                        dataset_name: OmegaConf.to_container(
                            sit_config.data.datasets[dataset_name]
                        )
                    }
                },
            }
        )
        (downloader,) = build_downloaders(config)
        assert downloader.name == dataset_name
        assert (
            downloader.path_dataset
            == tmp_path.resolve() / "data" / "anemoi" / f"{dataset_name}.zarr"
        )

    @pytest.mark.parametrize("dataset_name", SAMPLE_DATASETS)
    def test_statistics_stop_at_training_boundary(
        self, sit_config: DictConfig, dataset_name: str
    ) -> None:
        split = sit_config.data.split
        statistics_end = _day(sit_config.data.datasets[dataset_name].statistics.end)
        assert statistics_end <= _day(split.train[-1].end)
        assert statistics_end < _day(split.validate[0].start)
        assert statistics_end < _day(split.test[0].start)

    @pytest.mark.parametrize("dataset_name", SAMPLE_DATASETS)
    def test_missing_ranges_are_exactly_the_summers(
        self, sit_config: DictConfig, dataset_name: str
    ) -> None:
        """Every Apr-Oct day is missing; Nov-Mar days are only missing if listed singly.

        Single missing days are known gaps in the upstream source.
        """
        dates = sit_config.data.datasets[dataset_name].dates
        single, ranged = _missing_days(dates)
        day, end = _day(dates.start), _day(dates.end)
        while day <= end:
            if day.month in FREEZE_SEASON_MONTHS:
                assert day not in ranged, f"{day:%Y-%m-%d} is in a missing range"
            else:
                assert day in ranged, f"{day:%Y-%m-%d} is not marked missing"
            day += timedelta(days=1)
        assert all(d.month in FREEZE_SEASON_MONTHS for d in single)

    def test_split_periods_are_freeze_season_only(self, sit_config: DictConfig) -> None:
        split = sit_config.data.split
        for split_name in ("train", "validate", "test"):
            for period in split[split_name]:
                start, end = _day(period.start), _day(period.end)
                assert start.month in FREEZE_SEASON_MONTHS
                assert end.month in FREEZE_SEASON_MONTHS
                # Each period must be a single winter (no summer in between)
                assert (end - start).days < 183

    def test_split_periods_are_ordered(self, sit_config: DictConfig) -> None:
        split = sit_config.data.split
        assert _day(split.train[-1].end) < _day(split.validate[0].start)
        assert _day(split.validate[-1].end) < _day(split.test[0].start)

    @pytest.mark.parametrize("dataset_name", SAMPLE_DATASETS)
    def test_split_periods_fall_inside_dataset(
        self, sit_config: DictConfig, dataset_name: str
    ) -> None:
        dates = sit_config.data.datasets[dataset_name].dates
        for split_name in ("train", "validate", "test"):
            for period in sit_config.data.split[split_name]:
                assert _day(dates.start) <= _day(period.start)
                assert _day(period.end) <= _day(dates.end)

    def test_input_groups_have_encoders(self, sit_config: DictConfig) -> None:
        for group_name in sit_config.variables.input:
            assert group_name in sit_config.model.encoders

    @pytest.mark.parametrize("model_name", MODEL_CONFIGS)
    def test_sit_encoder_mirrors_sic_encoder(
        self, compose_config: Callable[..., DictConfig], model_name: str
    ) -> None:
        """Models with a SIC encoder get the same encoder for SIT; others fail loudly."""
        model = compose_config("sample", overrides=[f"model={model_name}"]).model
        config = compose_config("sample_north_sit", overrides=[f"model={model_name}"])
        if SIC_GROUP in model.get("encoders", {}):
            encoders = OmegaConf.to_container(config.model.encoders, resolve=True)
            assert isinstance(encoders, dict)
            assert encoders[SIT_GROUP] == encoders[SIC_GROUP]
        else:
            with pytest.raises(InterpolationKeyError):
                OmegaConf.to_container(config.model.encoders, resolve=True)

    def test_sic_only_baseline_override_drops_sit_inputs(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        config = compose_config(
            "sample_north_sit", overrides=[f"~variables.input.{SIT_GROUP}"]
        )
        assert list(config.variables.input) == [SIC_GROUP]
        assert list(config.variables.target) == [SIC_GROUP]
