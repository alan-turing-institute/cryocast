from collections.abc import Sequence
from functools import cached_property

import numpy as np
from torch.utils.data import Dataset

from icenet_mp.types import ArrayTCHW

from .calendar_day import calendar_day_index
from .single_dataset import SingleDataset


class CombinedDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[SingleDataset],
        target_group_name: str,
        target_variables: Sequence[str],
        *,
        n_forecast_steps: int = 1,
        n_history_steps: int = 1,
        climatology: ArrayTCHW | None = None,
    ) -> None:
        """Initialise a combined dataset from a sequence of SingleDatasets.

        One of the datasets must be the target and all must have the same frequency. The
        number of forecast and history steps can be set, which will determine the shape
        of the NTCHW tensors returned by __getitem__.

        Args:
            datasets: The datasets to combine.
            target_group_name: The name of the target dataset.
            target_variables: The names of the target variables.
            n_forecast_steps: The number of forecast steps.
            n_history_steps: The number of history steps.
            climatology: Optional [366, C, H, W] table of calendar-day means of the
                target variables (29 February holds its own slot). When given, each
                batch also contains a ``climatology`` key holding the calendar-day
                mean field for each forecast step. When ``None`` the batches are
                unchanged.

        """
        super().__init__()

        # Store the number of forecast and history steps
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps

        # Optional climatology table (calendar-day means of the target variables)
        self.climatology = climatology

        # Create a new dataset for the target with only the selected variables
        self.target = next(
            ds for ds in datasets if ds.name == target_group_name
        ).subset(variables=target_variables)
        self.inputs = list(datasets)

        # Require that all datasets have the same frequency
        frequencies = sorted({ds.frequency for ds in datasets})  # type: ignore[type-var]
        if len(frequencies) != 1:
            msg = f"Cannot combine datasets with different frequencies: {frequencies}."
            raise ValueError(msg)
        self.frequency = frequencies[0]

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Get list of dates that are available in all datasets."""
        # Identify dates that exist in all input datasets
        input_date_set = set.intersection(*(set(ds.dates) for ds in self.inputs))
        target_date_set = set(self.target.dates)
        available_dates = sorted(
            available_date
            for available_date in input_date_set
            # ... if all inputs have n_history_steps starting on start_date
            if all(
                date in input_date_set
                for date in self.get_history_steps(available_date)
            )
            # ... and if the target has n_forecast_steps starting after the history dates
            and all(
                date in target_date_set
                for date in self.get_forecast_steps(available_date)
            )
        )
        if len(available_dates) == 0:
            msg = (
                "CombinedDataset has no valid dates. This can happen when there "
                "are no valid windows given the configured history/forecast steps or "
                "when the input datasets do not have overlapping time ranges."
            )
            raise ValueError(msg)
        return available_dates

    @property
    def end_date(self) -> np.datetime64:
        """Return the end date of the dataset."""
        return self.dates[-1]

    @property
    def start_date(self) -> np.datetime64:
        """Return the start date of the dataset."""
        return self.dates[0]

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict[str, ArrayTCHW]:
        """Return the data for a single timestep as a dictionary.

        Note that because we have already checked which starting dates are valid in the
        `dates` property, we know that the requested dates will be consecutive in each
        data input, so we can use `get_tchw_slice` rather than `get_tchw`.

        Returns:
            A dictionary with dataset names as keys and a numpy array as the value.
            The shape of each array is:
            - input datasets: [n_history_steps, C_input_k, H_input_k, W_input_k]
            - target dataset: [n_forecast_steps, C_target, H_target, W_target]

            If a climatology table was provided, the dictionary also contains a
            ``climatology`` key with shape
            [n_forecast_steps, C_target, H_target, W_target] holding the calendar-day
            mean field for each forecast step.

        """
        start_date = self.dates[idx]
        batch: dict[str, ArrayTCHW] = {
            ds.name: ds.get_tchw_slice(start_date, self.n_history_steps, check=False)
            for ds in self.inputs
        }
        batch["target"] = self.target.get_tchw_slice(
            start_date + self.n_history_steps * self.frequency,
            self.n_forecast_steps,
            check=False,
        )
        if (climatology := self.climatology_for(start_date)) is not None:
            batch["climatology"] = climatology
        return batch

    def climatology_for(self, start_date: np.datetime64) -> ArrayTCHW | None:
        """Return the climatology field for each forecast step following the start date.

        The calendar day (month/day label) of each forecast step indexes the
        [366, C, H, W] climatology table, so the result has shape
        [n_forecast_steps, C, H, W].

        Args:
            start_date: The start date of the sample.

        Returns:
            The stack of calendar-day mean fields for the forecast steps, or ``None``
            if no climatology table was provided.

        """
        if self.climatology is None:
            return None
        day_indices = [
            calendar_day_index(forecast_step)
            for forecast_step in self.get_forecast_steps(start_date)
        ]
        return self.climatology[day_indices]

    def get_forecast_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive forecast dates for a given start date."""
        return [
            start_date + (idx + self.n_history_steps) * self.frequency
            for idx in range(self.n_forecast_steps)
        ]

    def get_history_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive history dates for a given start date."""
        return [
            start_date + idx * self.frequency for idx in range(self.n_history_steps)
        ]
