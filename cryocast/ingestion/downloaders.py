from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .data_downloader import DataDownloader


def build_downloaders(config: DictConfig) -> list[DataDownloader]:
    """Build a DataDownloader for each dataset in the config."""
    base_path = Path(config["base_path"]).resolve()
    downloaders = []
    for dataset_name, dataset_config in config["data"]["datasets"].items():
        # Anemoi 'forcings' need to be escaped with `\${}` to avoid being resolved here
        anemoi_config = cast("dict[str, Any]", OmegaConf.to_object(dataset_config))
        downloaders.append(DataDownloader(dataset_name, base_path, anemoi_config))
    return downloaders
