from abc import ABC, abstractmethod
from pathlib import Path


class IPreprocessor(ABC):
    def __init__(self, base_path: Path, dataset_name: str) -> None:
        """Initialise the IPreprocessor base class."""
        self.base_path = base_path
        self.dataset_name = dataset_name

    @abstractmethod
    def process(self, *, overwrite: bool) -> None:
        """Perform pre-processing actions at the specified path."""
