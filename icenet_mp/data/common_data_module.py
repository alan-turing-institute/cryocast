import logging
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from icenet_mp.types import ArrayTCHW, DataloaderArgs, DataSpace, Hemisphere, MaskType
from icenet_mp.utils import mask_dir

from .calendar_day import (
    CALENDAR_DAY_LABELS,
    FEBRUARY_28_INDEX,
    FEBRUARY_29_INDEX,
    N_CALENDAR_DAYS,
    calendar_day_index,
)
from .combined_dataset import CombinedDataset
from .single_dataset import SingleDataset

logger = logging.getLogger(__name__)


class CommonDataModule(LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        """Initialise a CommonDataModule from a config.

        The config specifies all datasets used and how to group them. Data splits are
        also determined from the config, and the appropriate data loaders are created.
        """
        super().__init__()

        # Load paths
        self.base_path = Path(config["base_path"])

        # Construct dataset groups
        self.dataset_groups = defaultdict(list)
        for dataset in config["data"]["datasets"].values():
            self.dataset_groups[dataset["group_as"]].append(
                (
                    self.base_path / "data" / "anemoi" / f"{dataset['name']}.zarr"
                ).resolve()
            )
        logger.info("Found %d dataset groups.", len(self.dataset_groups))
        for idx, (name, paths) in enumerate(self.dataset_groups.items(), start=1):
            logger.info("%d) %s:", idx, name)
            for path in paths:
                logger.info("%s - %s", " " * (len(str(idx)) + 1), path)

        # Check prediction target
        self.target_group_name = config["predict"]["target"]["group_name"]
        if self.target_group_name not in self.dataset_groups:
            available_groups = ", ".join(sorted(self.dataset_groups)) or "<none>"
            msg = (
                f"Prediction target group {self.target_group_name!r} was not found in "
                f"the configured datasets. Available groups: {available_groups}. "
                "When evaluating a checkpoint, ensure the dataset `group_as` matches "
                "the checkpoint's `predict.target.group_name`."
            )
            raise ValueError(msg)
        self._target_variables: list[str] = config["predict"]["target"].get(
            "variables", []
        )

        # Set periods for train, validation, and test
        self.batch_size = int(config["data"]["split"]["batch_size"])
        self.predict_periods = [
            {str(k): None if v is None else str(v) for k, v in period.items()}
            for period in config["data"]["split"]["predict"]
        ]
        self.test_periods = [
            {str(k): None if v is None else str(v) for k, v in period.items()}
            for period in config["data"]["split"]["test"]
        ]
        self.train_periods = [
            {str(k): None if v is None else str(v) for k, v in period.items()}
            for period in config["data"]["split"]["train"]
        ]
        self.val_periods = [
            {str(k): None if v is None else str(v) for k, v in period.items()}
            for period in config["data"]["split"]["validate"]
        ]

        # Set history and forecast steps
        self.n_forecast_steps = int(config["predict"].get("n_forecast_steps", 1))
        self.n_history_steps = int(config["predict"].get("n_history_steps", 1))

        # Set common arguments for the dataloader
        self._common_dataloader_kwargs = DataloaderArgs(
            batch_sampler=None,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,  # must be None when num_workers=0
            sampler=None,
            worker_init_fn=None,
        )

    @cached_property
    def datasets(self) -> dict[str, SingleDataset]:
        """Return a dictionary of dataset group names to SingleDataset objects."""
        return {
            name: SingleDataset(name, paths)
            for name, paths in self.dataset_groups.items()
        }

    @cached_property
    def hemisphere(self) -> Hemisphere:
        """Return the hemisphere of the dataset."""
        hemisphere: set[Hemisphere] = {ds.hemisphere for ds in self.datasets.values()}
        if len(hemisphere) != 1:
            msg = f"Found {len(hemisphere)} different hemisphere indicators across {len(self.dataset_groups)} dataset groups."
            raise ValueError(msg)
        return hemisphere.pop()

    @cached_property
    def input_spaces(self) -> list[DataSpace]:
        """Return the data space for each input."""
        return [ds.space for ds in self.datasets.values()]

    @cached_property
    def latitudes(self) -> dict[str, list[float]]:
        """Return the latitudes of the dataset."""
        return {name: ds.latitudes for name, ds in self.datasets.items()}

    @cached_property
    def longitudes(self) -> dict[str, list[float]]:
        """Return the longitudes of the dataset."""
        return {name: ds.longitudes for name, ds in self.datasets.items()}

    @cached_property
    def mask_directory(self) -> Path:
        """Mask directory for the prediction target group.

        A target group usually holds a single dataset with generated masks, but if it
        holds several, pick the first. Combining masks across datasets is unsupported.
        """
        paths = self.dataset_groups[self.target_group_name]
        available = [
            path
            for path in paths
            if any(
                (mask_dir(self.base_path, path.stem) / f"{kind}_mask.npy").exists()
                for kind in (MaskType.ACTIVE, MaskType.LAND)
            )
        ]
        chosen = (available or paths)[0].stem
        if len(paths) > 1:
            logger.warning(
                "Target group %r has %d datasets; using %r for masks "
                "(combining masks across datasets is not supported).",
                self.target_group_name,
                len(paths),
                chosen,
            )
        return mask_dir(self.base_path, chosen)

    @cached_property
    def output_space(self) -> DataSpace:
        """Return the data space of the desired output."""
        return (
            self.datasets[self.target_group_name]
            .subset(variables=self.target_variables)
            .space
        )

    @cached_property
    def target_variables(self) -> list[str]:
        """Return the names of the variables to predict."""
        if self._target_variables:
            return self._target_variables
        return self.variable_names[self.target_group_name]

    @cached_property
    def target_variable_indices(self) -> list[int]:
        """Return the indices of the variables to predict."""
        return [
            self.variable_names[self.target_group_name].index(variable)
            for variable in self.target_variables
        ]

    @cached_property
    def climatology(self) -> ArrayTCHW:
        """Return the climatology: calendar-day means of the target variables.

        The [366, C, H, W] table holds, for each calendar day (month/day label), the
        mean of the normalised target fields over dates sharing that calendar day
        within the averaging period. The averaging period is the union of the training
        split's date ranges, intersected with the dates available in the target
        dataset; it is never widened to dates outside the configured training periods.
        Dates that are missing from the dataset are never included in a mean.

        29 February is the exception: because a training period spanning only
        non-leap years has no such date, it is not required to have its own data. If
        no date in the averaging period falls on 29 February, that slot instead copies
        the 28 February mean.

        Raises:
            ValueError: If the training periods have no available dates at all, or a
                calendar day other than 29 February has no available dates in the
                period.

        """
        target = self.datasets[self.target_group_name].subset(
            variables=self.target_variables
        )
        period_dates = [day for day in target.dates if self._in_train_periods(day)]
        if not period_dates:
            msg = (
                "Cannot build climatology: none of the configured training periods "
                "have available dates in the target dataset "
                f"({target.start_date} to {target.end_date})."
            )
            raise ValueError(msg)
        by_day: dict[int, list[np.datetime64]] = defaultdict(list)
        for day in period_dates:
            by_day[calendar_day_index(day)].append(day)
        table = np.zeros((N_CALENDAR_DAYS, *target.space.chw), dtype=np.float64)
        for index, label in enumerate(CALENDAR_DAY_LABELS):
            day_dates = by_day.get(index, [])
            if not day_dates:
                if index == FEBRUARY_29_INDEX:
                    logger.info(
                        "Climatology: no 29 February dates in the averaging period; "
                        "using the 28 February mean for that day instead."
                    )
                    table[index] = table[FEBRUARY_28_INDEX]
                    continue
                msg = (
                    f"Cannot build climatology: calendar day {label} has no available "
                    f"dates in the averaging period ({min(period_dates)} to "
                    f"{max(period_dates)}). Check the configured training periods "
                    "against the available data range."
                )
                raise ValueError(msg)
            table[index] = target.get_tchw(day_dates).astype(np.float64).mean(axis=0)
        logger.info(
            "Climatology: computed calendar-day means over %d dates between %s and %s.",
            len(period_dates),
            min(period_dates),
            max(period_dates),
        )
        return table.astype(np.float32)

    @cached_property
    def _climatology_or_none(self) -> ArrayTCHW | None:
        """Return the climatology table, or ``None`` if it cannot be built.

        Climatology is an optional comparison baseline for every model, not just the
        Climatology model itself, so a config whose train-period union does not cover
        every calendar day (e.g. a short demo/synthetic split) must not break every
        other model's dataloaders. Use this instead of ``climatology`` when wiring up
        dataloaders; use ``climatology`` directly when the table is required (e.g. in
        tests) and a missing calendar day should raise loudly.
        """
        try:
            return self.climatology
        except ValueError as err:
            logger.warning(
                "Climatology baseline unavailable, continuing without it: %s", err
            )
            return None

    def _in_train_periods(self, day: np.datetime64) -> bool:
        """Return whether the date falls within any of the training period ranges.

        Bounds are compared at day precision, so a bound carrying a time component
        (e.g. ``2019-01-01T12:00:00``) behaves like ``2019-01-01``.
        """
        day_day = day.astype("datetime64[D]")
        for period in self.train_periods:
            start = period.get("start")
            end = period.get("end")
            if start is not None and day_day < np.datetime64(start).astype(
                "datetime64[D]"
            ):
                continue
            if end is not None and day_day > np.datetime64(end).astype("datetime64[D]"):
                continue
            return True
        return False

    @cached_property
    def variable_names(self) -> dict[str, list[str]]:
        """Return the variable names for each input."""
        return {ds.name: ds.variable_names for ds in self.datasets.values()}

    def assign_workers(self, n_workers: int) -> None:
        """Assign number of workers for data loading."""
        logger.info("Assigning %d workers for data loading.", n_workers)
        self._common_dataloader_kwargs["num_workers"] = n_workers
        self._common_dataloader_kwargs["persistent_workers"] = n_workers > 0
        self._common_dataloader_kwargs["prefetch_factor"] = 1 if n_workers > 0 else None

    def predict_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct predict dataloader."""
        dataset = CombinedDataset(
            [
                ds.subset(date_ranges=self.predict_periods)
                for ds in self.datasets.values()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
            climatology=self._climatology_or_none,
        )
        logger.info(
            "Loaded predict dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)

    def test_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct test dataloader."""
        dataset = CombinedDataset(
            [ds.subset(date_ranges=self.test_periods) for ds in self.datasets.values()],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
            climatology=self._climatology_or_none,
        )
        logger.info(
            "Loaded test dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)

    @cached_property
    def training_dataset(self) -> CombinedDataset:
        """Return the dataset used for training."""
        return CombinedDataset(
            [
                ds.subset(date_ranges=self.train_periods)
                for ds in self.datasets.values()
            ],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
            climatology=self._climatology_or_none,
        )

    def train_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct train dataloader."""
        dataset = self.training_dataset
        logger.info(
            "Loaded training dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, ArrayTCHW]]:
        """Construct validation dataloader."""
        dataset = CombinedDataset(
            [ds.subset(date_ranges=self.val_periods) for ds in self.datasets.values()],
            n_forecast_steps=self.n_forecast_steps,
            n_history_steps=self.n_history_steps,
            target_group_name=self.target_group_name,
            target_variables=self.target_variables,
            climatology=self._climatology_or_none,
        )
        logger.info(
            "Loaded validation dataset with %d dates between %s and %s.",
            len(dataset),
            dataset.start_date,
            dataset.end_date,
        )
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)
