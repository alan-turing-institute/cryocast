from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest

from cryocast.ingestion.postprocessors import FiniteValueMaskGenerator
from cryocast.utils import mask_dir


class _Dataset:
    field_shape = (2, 2)
    missing: ClassVar[list[int]] = [1]

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> np.ndarray:
        values = {
            0: np.array([[1.0, np.nan], [1.0, 1.0]]),
            1: np.ones((2, 2)),
            2: np.array([[1.0, 1.0], [np.nan, 1.0]]),
        }
        return values[index]


class TestFiniteValueMaskGenerator:
    """Tests for masks derived from persistent finite target cells."""

    def test_intersects_finite_cells_across_available_dates(
        self, tmp_path: Path
    ) -> None:
        """Only cells finite at every non-missing timestep remain valid."""
        generator = FiniteValueMaskGenerator(tmp_path, "carra2", variable="ice_conc")
        with pytest.MonkeyPatch.context() as mp:
            open_dataset = MagicMock(return_value=_Dataset())
            mp.setattr(
                "cryocast.ingestion.postprocessors.finite_value_mask_generator.open_dataset",
                open_dataset,
            )
            generator.process(tmp_path / "target.zarr", overwrite=False)

        expected = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(
            np.load(mask_dir(tmp_path, "carra2") / "land_mask.npy"), expected
        )
        np.testing.assert_array_equal(
            np.load(mask_dir(tmp_path, "carra2") / "active_mask.npy"), expected
        )
        open_dataset.assert_called_once_with(
            tmp_path / "target.zarr", select="ice_conc"
        )

    def test_existing_masks_skip_dataset_read(self, tmp_path: Path) -> None:
        """Existing masks are reused unless overwrite is requested."""
        output = mask_dir(tmp_path, "carra2")
        output.mkdir(parents=True)
        np.save(output / "land_mask.npy", np.ones((2, 2), dtype=np.uint8))
        np.save(output / "active_mask.npy", np.ones((2, 2), dtype=np.uint8))
        generator = FiniteValueMaskGenerator(tmp_path, "carra2", variable="ice_conc")
        with pytest.MonkeyPatch.context() as mp:
            open_dataset = MagicMock()
            mp.setattr(
                "cryocast.ingestion.postprocessors.finite_value_mask_generator.open_dataset",
                open_dataset,
            )
            generator.process(tmp_path / "target.zarr", overwrite=False)

        open_dataset.assert_not_called()

    def test_requires_variable(self, tmp_path: Path) -> None:
        """A target variable is required to define validity."""
        with pytest.raises(ValueError, match="variable"):
            FiniteValueMaskGenerator(tmp_path, "carra2", variable="")
