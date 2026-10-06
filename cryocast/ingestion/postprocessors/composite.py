from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra

if TYPE_CHECKING:
    from .ipostprocessor import IPostprocessor


class CompositePostprocessor:
    def __init__(
        self, dataset_name: str, postprocessor_cfgs: dict[str, Any], base_path: Path
    ) -> None:
        """Initialise a CompositePostprocessor from its child postprocessors."""
        self.children: list[IPostprocessor] = [
            hydra.utils.instantiate(
                spec, base_path=base_path, dataset_name=dataset_name
            )
            for spec in postprocessor_cfgs.values()
        ]

    def process(self, path_dataset: Path, *, overwrite: bool) -> None:
        """Process the dataset with each child postprocessor, in order."""
        for child in self.children:
            child.process(path_dataset, overwrite=overwrite)
