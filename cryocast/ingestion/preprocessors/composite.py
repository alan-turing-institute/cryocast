from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra

if TYPE_CHECKING:
    from .ipreprocessor import IPreprocessor


class CompositePreprocessor:
    def __init__(
        self, dataset_name: str, preprocessor_cfgs: dict[str, Any], base_path: Path
    ) -> None:
        """Initialise a CompositePreprocessor from its child preprocessors."""
        self.children: list[IPreprocessor] = [
            hydra.utils.instantiate(
                spec, base_path=base_path, dataset_name=dataset_name
            )
            for spec in preprocessor_cfgs.values()
        ]

    def process(self, *, overwrite: bool) -> None:
        """Process the dataset with each child preprocessor, in order."""
        for child in self.children:
            child.process(overwrite=overwrite)
