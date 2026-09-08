from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from icenet_mp.ingestion.data_downloader import DataDownloader

DATASET_NAME = "demo-sicnorth-seas5-1p0-2024-2024-24h-v1"
RECIPE_PATH = (
    files("icenet_mp.config")
    / "data"
    / "datasets"
    / "demo_sicnorth_seas5_1p0_2024_2024_24h_v1.yaml"
)


def _load_recipe() -> dict[str, Any]:
    config: dict[str, dict[str, Any]] = yaml.safe_load(RECIPE_PATH.read_text())
    return config[DATASET_NAME]


def test_seas5_recipe_is_accepted_by_data_downloader(tmp_path: Path) -> None:
    """Construct the SEAS5 recipe through DataDownloader without network access."""
    recipe = _load_recipe()

    downloader = DataDownloader(DATASET_NAME, tmp_path, recipe)

    assert downloader.name == DATASET_NAME
    assert downloader.recipe is not None


def test_seas5_recipe_uses_cds_forecast_trajectory() -> None:
    """Keep the critical CDS trajectory request fields configured for SEAS5."""
    recipe = _load_recipe()
    mars = recipe["input"]["mars"]

    assert recipe["output"]["layout"] == "trajectories"
    assert mars["use_cdsapi_dataset"] == "seasonal-original-single-levels"
    assert mars["origin"] == "ecmf"
    assert mars["system"] == "51"
    assert mars["param"] == ["31.128"]
    assert mars["type"] == "fc"
    assert mars["stream"] == "mmsf"


def test_seas5_statistics_cover_the_complete_demo_trajectory(tmp_path: Path) -> None:
    """Keep the full forecast trajectory inside Anemoi's statistics envelope."""
    downloader = DataDownloader(DATASET_NAME, tmp_path, _load_recipe())
    base_dates = np.array(["2024-01-01T00:00:00"], dtype="datetime64[s]")
    steps = np.arange(1, 8) * np.timedelta64(24, "h")

    stats_filter = downloader.recipe.statistics.trajectory_statistics_filter(
        base_dates, steps
    )

    assert stats_filter.mask(base_dates).tolist() == [True]
