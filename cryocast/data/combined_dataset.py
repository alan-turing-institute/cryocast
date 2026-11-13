from collections.abc import Sequence
from functools import cached_property

import numpy as np
from torch.utils.data import Dataset

from cryocast.types import ArrayTCHW

from .calendar_day_climatology import CalendarDayClimatology
from .single_dataset import SingleDataset


class CombinedDataset(Dataset):
    def __init__(  # noqa: PLR0913
        self,
        datasets: Sequence[SingleDataset],
        target_group_name: str,
        target_variables: Sequence[str],
        *,
        n_forecast_steps: int = 1,
        n_history_steps: int = 1,
        climatology: ArrayTCHW | None = None,
        include_future_inputs: bool = False,
        optional_groups: Sequence[str] = (),
        optional_fill: float = -1.0,
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
            include_future_inputs: Whether each batch should also contain every
                input dataset over the forecast steps (the same dates as ``target``),
                under ``future_<input>`` keys. If ``True``, only start dates where every
                required input covers the forecast steps are valid, and optional inputs
                hold ``optional_fill`` on any forecast step they are missing. If
                ``False``, the batches are unchanged.
            optional_groups: Input groups that can have missing days, e.g.
                ``["float-argo"]``. A window is kept even if one of these groups is
                missing a day, and that day is filled with ``optional_fill``. The
                target cannot be optional. Empty by default.
            optional_fill: The value for missing days of optional inputs, in
                normalised units. Real data lies in [0, 1], so the default of -1 can
                never be mistaken for it.

        """
        super().__init__()

        # Store the number of forecast and history steps
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps

        # Optional climatology table (calendar-day means of the target variables)
        self.climatology = climatology

        # Whether to also include the inputs over the forecast steps
        self.include_future_inputs = include_future_inputs
        self.n_input_steps = n_history_steps + (
            n_forecast_steps if include_future_inputs else 0
        )

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

        # Inputs that are allowed to be missing on some days
        input_names = {ds.name for ds in self.inputs}
        if unknown := sorted(set(optional_groups) - input_names):
            msg = (
                f"Optional groups {unknown} are not inputs of this dataset. "
                f"Available inputs: {sorted(input_names)}."
            )
            raise ValueError(msg)
        if target_group_name in optional_groups:
            msg = f"The target dataset {target_group_name!r} cannot be optional."
            raise ValueError(msg)
        if isinstance(optional_fill, bool) or not isinstance(
            optional_fill, (int, float)
        ):
            msg = f"optional_fill must be a number, not {optional_fill!r}."
            raise TypeError(msg)
        self.optional_groups = set(optional_groups)
        self.optional_fill = float(optional_fill)

    @cached_property
    def dates(self) -> list[np.datetime64]:
        """Get list of dates that are available in all required datasets."""
        # Identify dates that exist in all required input datasets. Optional inputs
        # are filled on any days they are missing, so they do not restrict the dates.
        required_inputs = [
            ds for ds in self.inputs if ds.name not in self.optional_groups
        ]
        input_date_set = set.intersection(*(set(ds.dates) for ds in required_inputs))
        target_date_set = set(self.target.dates)
        available_dates = sorted(
            available_date
            for available_date in input_date_set
            # ... if all required inputs have n_input_steps starting on start_date
            if all(
                date in input_date_set for date in self.get_input_steps(available_date)
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
        required data input, so `get_input_window` can read them as a single slice.
        Optional inputs hold `optional_fill` on any of those dates they are missing.

        Returns:
            A dictionary with dataset names as keys and a numpy array as the value.
            The shape of each array is:
            - input: [n_history_steps, C_input_k, H_input_k, W_input_k]
            - target: [n_forecast_steps, C_target, H_target, W_target]

            If a climatology table was provided, the dictionary also contains the
            calendar-day mean field:
            - climatology: [n_forecast_steps, C_target, H_target, W_target]

            If ``include_future_inputs`` is set, the dictionary also contains the
            input data for the forecast steps:
            - `future_<input>`: [n_forecast_steps, C_input_k, H_input_k, W_input_k]

        """
        start_date = self.dates[idx]
        # Read history and (if requested) forecast steps as one contiguous window
        batch: dict[str, ArrayTCHW] = {}
        for ds in self.inputs:
            window = self.get_input_window(ds, start_date, self.n_input_steps)
            batch[ds.name] = window[: self.n_history_steps]
            if self.include_future_inputs:
                batch[f"future_{ds.name}"] = window[self.n_history_steps :]
        batch["target"] = self.target.get_tchw_slice(
            start_date + self.n_history_steps * self.frequency,
            self.n_forecast_steps,
            check=False,
        )
        if (climatology := self.climatology_for(start_date)) is not None:
            batch["climatology"] = climatology
        return batch

    def _consecutive_dates(
        self, start_date: np.datetime64, n_steps: int
    ) -> list[np.datetime64]:
        """Return list of n_steps consecutive dates, starting on the given date."""
        return [start_date + idx * self.frequency for idx in range(n_steps)]

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
            CalendarDayClimatology.day_index(forecast_step)
            for forecast_step in self.get_forecast_steps(start_date)
        ]
        return self.climatology[day_indices]

    def get_forecast_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive forecast dates for a given start date."""
        return self._consecutive_dates(
            start_date + self.n_history_steps * self.frequency, self.n_forecast_steps
        )

    def get_history_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive history dates for a given start date."""
        return self._consecutive_dates(start_date, self.n_history_steps)

    def get_input_steps(self, start_date: np.datetime64) -> list[np.datetime64]:
        """Return list of consecutive input dates for a given start date.

        These are the history steps, followed by the forecast steps if
        ``include_future_inputs`` is set.
        """
        return self._consecutive_dates(start_date, self.n_input_steps)

    def get_input_window(
        self, dataset: SingleDataset, start_date: np.datetime64, n_steps: int
    ) -> ArrayTCHW:
        """Return normalised data of an input for consecutive timesteps, [T, C, H, W].

        Inputs must either have data on every date, enabling them to be read as a
        contiguous slice or have a fill value for any missing dates. Required inputs
        are read without checking for dataslice boundaries, so the caller must request
        a valid window, e.g. a start date from `dates`.

        Args:
            dataset: The input dataset to read from.
            start_date: The date of the first timestep to return.
            n_steps: The number of consecutive timesteps to return.

        Returns:
            The normalised data for the specified input and timesteps.

        """
        if dataset.name not in self.optional_groups:
            return dataset.get_tchw_slice(start_date, n_steps, check=False)
        return dataset.get_tchw(
            self._consecutive_dates(start_date, n_steps),
            fill_value=self.optional_fill,
        )

    def variable_list(self) -> list[str]:
        """Return a pretty-formatted list of input variables in the combined dataset.

        Returns:
            A list of pretty-formatted strings like:

            [
                "Combined dataset has 3 input variables:",
                "  1) dataset1/variableA",
                "  2) dataset1/variableB",
                "  3) dataset2/variableC",
            ]

        """
        variable_ids = [
            f"{ds.name}/{variable_name}"
            for ds in self.inputs
            for variable_name in ds.variable_names
        ]
        return [
            f"Combined dataset has {len(variable_ids)} input variables:",
            *(
                f"{idx:>{len(str(len(variable_ids)))}}) {variable_id}"
                for idx, variable_id in enumerate(variable_ids, start=1)
            ),
        ]
