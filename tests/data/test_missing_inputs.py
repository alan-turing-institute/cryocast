from pathlib import Path

import numpy as np

from icenet_mp.data.combined_dataset import MISSING_INPUT_VALUE, CombinedDataset
from icenet_mp.data.single_dataset import SingleDataset


def test_single_dataset_fills_unavailable_dates(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    dataset = SingleDataset(name="conditioning", input_files=[mock_dataset])
    dataset.dates = [date for date in dates_as_np if date != dates_as_np[1]]

    values = dataset.get_tchw_with_fill(dates_as_np[:3])

    assert values.shape == (3, 3, 2, 2)
    assert np.all(values[1] == MISSING_INPUT_VALUE)
    assert np.any(values[0] != MISSING_INPUT_VALUE)
    assert np.any(values[2] != MISSING_INPUT_VALUE)


def test_missing_conditioning_data_does_not_remove_target_windows(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    target = SingleDataset(name="target", input_files=[mock_dataset])
    conditioning = SingleDataset(name="conditioning", input_files=[mock_dataset])
    conditioning.dates = list(dates_as_np[2:])

    combined = CombinedDataset(
        datasets=[target, conditioning],
        target_group_name="target",
        target_variables=["ice_conc"],
        allow_missing_inputs=True,
        n_history_steps=2,
        n_forecast_steps=1,
    )

    assert combined.dates == list(dates_as_np[:3])
    batch = combined[0]
    assert np.all(batch["conditioning"] == MISSING_INPUT_VALUE)
    assert np.any(batch["target"] != MISSING_INPUT_VALUE)


def test_missing_target_history_still_removes_window(
    mock_dataset: Path, dates_as_np: tuple[np.datetime64, ...]
) -> None:
    target = SingleDataset(name="target", input_files=[mock_dataset])
    target.dates = [date for date in dates_as_np if date != dates_as_np[1]]
    conditioning = SingleDataset(name="conditioning", input_files=[mock_dataset])

    combined = CombinedDataset(
        datasets=[target, conditioning],
        target_group_name="target",
        target_variables=["ice_conc"],
        allow_missing_inputs=True,
        n_history_steps=2,
        n_forecast_steps=1,
    )

    assert dates_as_np[0] not in combined.dates
    assert dates_as_np[2] in combined.dates
