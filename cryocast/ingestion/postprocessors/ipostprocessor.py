from abc import ABC, abstractmethod
from pathlib import Path


class IPostprocessor(ABC):
    def __init__(self, base_path: Path, dataset_name: str) -> None:
        """Initialise the IPostprocessor base class."""
        self.base_path = base_path
        self.dataset_name = dataset_name

    @abstractmethod
    def process(self, path_dataset: Path, *, overwrite: bool) -> None:
        """Perform post-processing actions on a given dataset."""
