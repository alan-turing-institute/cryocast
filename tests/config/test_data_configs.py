from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import pytest
from omegaconf import DictConfig

DATA_DIR = Path(str(files("icenet_mp.config"))) / "data"
DATA_GROUP_CONFIGS = sorted(
    p.stem for p in DATA_DIR.glob("*.yaml") if not p.name.endswith(".local.yaml")
)


class TestDataConfigs:
    """Regression tests for icenet-mp's top-level data= config groups."""

    @pytest.mark.parametrize("config_name", DATA_GROUP_CONFIGS, ids=DATA_GROUP_CONFIGS)
    def test_data_groups_compose(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        config = compose_config("sample", overrides=[f"data={config_name}"])

        assert config.data.datasets
        assert config.data.split.train
        assert config.data.split.test
        assert config.data.split.validate

    @pytest.mark.parametrize(
        "config_name",
        ["full_north", "full_south", "sample_north", "sample_south"],
        ids=["full_north", "full_south", "sample_north", "sample_south"],
    )
    def test_dataset_statistics_stop_at_training_boundary(
        self, compose_config: Callable[..., DictConfig], config_name: str
    ) -> None:
        """Datasets must not use validation/test/holdout dates for normalisation stats."""
        config = compose_config("sample", overrides=[f"data={config_name}"])

        training_end = str(config.data.split.train[-1].end)
        held_out_starts = [
            str(period.start)
            for split_name in ("validate", "test", "predict")
            for period in config.data.split[split_name]
            if period.start is not None
        ]
        earliest_held_out = min(held_out_starts)

        assert training_end < earliest_held_out
        for dataset in config.data.datasets.values():
            assert dataset.statistics.end is not None
            statistics_end = str(dataset.statistics.end)
            statistics_end_date = statistics_end.split("T", maxsplit=1)[0].split(
                " ", maxsplit=1
            )[0]
            assert statistics_end_date == training_end
            assert statistics_end_date < earliest_held_out

    def test_downscaling_north_defines_svalbard_roi_and_carra2_target(
        self, compose_config: Callable[..., DictConfig]
    ) -> None:
        """The downscaling data config pins one reproducible CARRA2 ROI."""
        config = compose_config("sample", overrides=["data=downscaling_north"])

        assert config.data.roi.name == "svalbard"
        assert config.data.roi.crs == "EPSG:4326"
        assert [
            config.data.roi.north,
            config.data.roi.west,
            config.data.roi.south,
            config.data.roi.east,
        ] == [81.0, 15.0, 76.0, 35.0]

        assert len(config.data.datasets) == 1
        dataset = next(iter(config.data.datasets.values()))
        assert dataset.group_as == "sic-carra2"
        cds = dataset.input.pipe[0].cds
        assert cds.dataset == "reanalysis-pan-carra"
        assert cds.time_from_dates is True
        assert list(cds.request.area) == [81.0, 15.0, 76.0, 35.0]
        assert list(cds.request.variable) == ["sea_ice_area_fraction"]
        assert cds.request.product_type == "analysis"
        assert cds.request.data_format == "grib"
        crop = dataset.input.pipe[1]["crop-latlon"]
        assert crop.resolution == "2p5km"
        assert [crop.north, crop.west, crop.south, crop.east] == [
            81.0,
            15.0,
            76.0,
            35.0,
        ]
        assert dataset.dates.start.endswith("T12:00:00")
