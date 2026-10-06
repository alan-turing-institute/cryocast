from pathlib import Path

import numpy as np
import pytest

from icenet_mp.data.combined_dataset import CombinedDataset
from icenet_mp.data.single_dataset import SingleDataset


def test_combined_dataset_returns_physical_target_uncertainty(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    """Return target uncertainty without model-input normalisation."""
    dataset = SingleDataset(name="target", input_files=[mock_dataset])
    combined = CombinedDataset(
        datasets=[dataset],
        target_group_name="target",
        target_variables=["ice_conc"],
        target_uncertainty_variable="ice_thickness",
        n_history_steps=2,
        n_forecast_steps=1,
    )

    batch = combined[0]

    assert set(batch) == {"target", "target_uncertainty"}
    assert batch["target"].shape == batch["target_uncertainty"].shape
    assert batch["target_uncertainty"].shape == (1, 1, 2, 2)

    raw_uncertainty = SingleDataset(
        name="target",
        input_files=[mock_dataset],
        normalise=False,
        variables=["ice_thickness"],
    ).get_tchw([dates_as_np[2]])
    np.testing.assert_array_equal(batch["target_uncertainty"], raw_uncertainty)
    assert batch["target_uncertainty"].max() > 1.0


def test_combined_dataset_rejects_missing_uncertainty_variable(
    mock_dataset: Path,
) -> None:
    """Reject an uncertainty variable absent from the target dataset."""
    dataset = SingleDataset(name="target", input_files=[mock_dataset])

    with pytest.raises(ValueError, match=r"uncertainty variable.*not found"):
        CombinedDataset(
            datasets=[dataset],
            target_group_name="target",
            target_variables=["ice_conc"],
            target_uncertainty_variable="not_a_variable",
        )


def test_uncertainty_weighting_requires_single_target_variable(
    mock_dataset: Path,
) -> None:
    """Reject uncertainty weighting for multiple predicted target variables."""
    dataset = SingleDataset(name="target", input_files=[mock_dataset])

    with pytest.raises(ValueError, match="exactly one predicted variable"):
        CombinedDataset(
            datasets=[dataset],
            target_group_name="target",
            target_variables=["ice_conc", "temperature"],
            target_uncertainty_variable="ice_thickness",
        )
