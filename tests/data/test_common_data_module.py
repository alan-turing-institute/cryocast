import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from omegaconf import DictConfig
from torch.utils.data import RandomSampler, SequentialSampler

from icenet_mp.data.calendar_day_climatology import CalendarDayClimatology
from icenet_mp.data.common_data_module import CommonDataModule
from icenet_mp.data.single_dataset import SingleDataset
from icenet_mp.utils import mask_dir
from tests.conftest import (
    CLIMATOLOGY_END,
    CLIMATOLOGY_MISSING,
    CLIMATOLOGY_START,
    CLIMATOLOGY_VARIABLES,
    build_zarr,
    make_climatology_data_dict,
)

FEB_29 = CalendarDayClimatology.day_index(np.datetime64("2000-02-29"))

# Union-of-training-periods arrangement: all of 2017 and 2018 plus the first half of
# 2019, so the 2019 second half is excluded from the climatology averaging period. None
# of these years are leap years, so 29 February is never present in the period.
TRAIN_PERIODS: list[dict[str, str | None]] = [
    {"start": "2017-01-01", "end": "2018-12-31"},
    {"start": "2019-01-01", "end": "2019-06-30"},
]


def _all_dates() -> list[datetime.datetime]:
    """Return every calendar day covered by the climatology zarr."""
    return [
        CLIMATOLOGY_START + datetime.timedelta(days=i)
        for i in range((CLIMATOLOGY_END - CLIMATOLOGY_START).days + 1)
    ]


def _available_dates() -> list[datetime.datetime]:
    """Return the dates present in the climatology zarr (missing dates excluded)."""
    missing = {d.date() for d in CLIMATOLOGY_MISSING}
    return [d for d in _all_dates() if d.date() not in missing]


def _period_dates(periods: list[dict[str, str | None]]) -> list[datetime.datetime]:
    """Return the available dates falling within any of the given ISO-bounded periods."""
    dates: list[datetime.datetime] = []
    for date in _available_dates():
        day = date.strftime("%Y-%m-%d")
        for period in periods:
            start = period.get("start")
            end = period.get("end")
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
            dates.append(date)
            break
    return dates


def _zarr_array(zarr_path: Path, name: str) -> np.ndarray:
    """Read a named array from the climatology zarr as a NumPy array."""
    return np.asarray(zarr.open_group(str(zarr_path), mode="r")[name])


def _normalised_rows(zarr_path: Path, dates: list[datetime.datetime]) -> np.ndarray:
    """Return [n, C, H, W] float32 rows replicating SingleDataset.normalise.

    The per-channel min/max statistics are read from the zarr (float64), the scale is
    computed in float64 and cast to float32, and the normalisation arithmetic is done
    in float32, exactly as SingleDataset does.
    """
    raw = _zarr_array(zarr_path, "data")
    height, width = zarr.open_group(str(zarr_path), mode="r").attrs["field_shape"]
    n_channels = raw.shape[1]
    minimum = _zarr_array(zarr_path, "minimum").astype(np.float64)
    maximum = _zarr_array(zarr_path, "maximum").astype(np.float64)
    scale = (1.0 / (maximum - minimum)).astype(np.float32).reshape(n_channels, 1, 1)
    offset = minimum.astype(np.float32).reshape(n_channels, 1, 1)
    full_index = {d.date(): i for i, d in enumerate(_all_dates())}
    rows = []
    for date in dates:
        row = raw[full_index[date.date()]].reshape(n_channels, 1, height, width)[:, 0]
        rows.append((row - offset) * scale)
    return np.stack(rows, axis=0)


def _spy_from_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[SingleDataset, list[np.datetime64]]]:
    """Record the dataset and dates passed to ``CalendarDayClimatology.from_dataset``.

    The statistics themselves are tested in ``test_calendar_day_climatology.py``; here
    we check which normalised fields ``CommonDataModule`` hands over.
    """
    calls: list[tuple[SingleDataset, list[np.datetime64]]] = []
    from_dataset = CalendarDayClimatology.from_dataset

    def spy(
        dataset: SingleDataset, dates: list[np.datetime64]
    ) -> CalendarDayClimatology:
        calls.append((dataset, list(dates)))
        return from_dataset(dataset, dates)

    monkeypatch.setattr(CalendarDayClimatology, "from_dataset", spy)
    return calls


def _as_days(dates: list[datetime.datetime] | list[np.datetime64]) -> np.ndarray:
    """Return the dates as a day-precision NumPy array for comparison."""
    return np.array(dates, dtype="datetime64[D]")


def _climatology_cfg(
    base_path: Path,
    train_periods: list[dict[str, str | None]],
    target_variables: list[str] = CLIMATOLOGY_VARIABLES,
) -> DictConfig:
    """Build a CommonDataModule config pointing at the climatology zarr."""
    open_period: list[dict[str, Any]] = [{"start": None, "end": None}]
    return DictConfig(
        {
            "base_path": str(base_path),
            "data": {
                "datasets": {"sic": {"name": "sic_south", "group_as": "sic"}},
                "split": {
                    "predict": open_period,
                    "test": open_period,
                    "train": train_periods,
                    "validate": open_period,
                },
            },
            "variables": {
                "input": {"sic": CLIMATOLOGY_VARIABLES},
                "target": {"sic": target_variables},
            },
            "window": {
                "batch_size": 2,
                "n_forecast_steps": 1,
                "n_history_steps": 1,
            },
        }
    )


def _climatology(dm: CommonDataModule) -> CalendarDayClimatology:
    """Return the data module's climatology, failing the test if it is unavailable."""
    climatology = dm.climatology
    assert climatology is not None
    return climatology


NONE_PERIOD = [{"start": None, "end": None}]


def _build_config(
    base_path: str,
    datasets: dict,
    *,
    input_variables: dict[str, list[str]],
    target_variables: dict[str, list[str]],
    split: dict[str, list[dict[str, str | None]]] | None = None,
) -> DictConfig:
    """Build a minimal CommonDataModule config for tests."""
    return DictConfig(
        {
            "base_path": base_path,
            "data": {
                "datasets": datasets,
                "split": split
                or {
                    "predict": NONE_PERIOD,
                    "test": NONE_PERIOD,
                    "train": NONE_PERIOD,
                    "validate": NONE_PERIOD,
                },
            },
            "variables": {"input": input_variables, "target": target_variables},
            "window": {
                "batch_size": 2,
                "n_forecast_steps": 1,
                "n_history_steps": 1,
            },
        }
    )


def _single_group_config(
    mock_dataset: Path,
    *,
    input_variables: list[str],
    target_variables: list[str],
    group_as: str = "group1",
) -> DictConfig:
    """Build a config with one dataset group backed by the real `mock_dataset` fixture."""
    return _build_config(
        str(mock_dataset.parent.parent.parent),
        {"ds1": {"name": mock_dataset.stem, "group_as": group_as}},
        input_variables={group_as: input_variables},
        target_variables={group_as: target_variables},
    )


class TestPeriods:
    """Period bounds are stringified while preserving None (YAML null)."""

    def test_null_preserved_as_none(self, cfg_common_data_module: DictConfig) -> None:
        """Python None (YAML null) must not be stringified to 'None'."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.predict_periods == [{"start": None, "end": None}]

    def test_string_values_unchanged(self, cfg_common_data_module: DictConfig) -> None:
        """Date strings must pass through without modification."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.test_periods == [{"start": "2020-01-01", "end": "2020-12-31"}]
        assert dm.val_periods == [{"start": "2020-01-01", "end": "2020-03-31"}]

    def test_mixed_none_and_string_in_same_period(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """A period with one None bound and one date string normalises both correctly."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.train_periods == [
            {"start": None, "end": "2019-12-31"},
            {"start": "2018-01-01", "end": None},
        ]


class TestInTrainPeriods:
    """`_in_train_periods` decides which dates the climatology average may include.

    These tests isolate the period-membership check, which needs no I/O; its use in
    building the climatology is covered by `TestCommonDataModuleClimatology`.
    """

    @pytest.mark.parametrize(
        ("periods", "day", "expected"),
        [
            ([{"start": "2018-01-01", "end": "2018-12-31"}], "2018-06-15", True),
            ([{"start": "2018-01-01", "end": "2018-12-31"}], "2019-01-01", False),
            (
                [
                    {"start": "2017-01-01", "end": "2017-12-31"},
                    {"start": "2019-01-01", "end": "2019-12-31"},
                ],
                "2019-06-15",
                True,
            ),
            ([{"start": None, "end": "2018-12-31"}], "2000-01-01", True),
            ([{"start": None, "end": "2018-12-31"}], "2019-01-01", False),
            ([{"start": "2018-01-01", "end": None}], "2030-01-01", True),
            ([{"start": "2018-01-01", "end": None}], "2017-12-31", False),
            ([{"start": None, "end": None}], "1900-01-01", True),
            (
                [{"start": "2019-01-01T12:00:00", "end": "2019-01-01T12:00:00"}],
                "2019-01-01T00:00:00",
                True,
            ),
            ([], "2019-01-01", False),
        ],
        ids=[
            "within-bounded-period",
            "outside-every-period",
            "any-one-of-several-periods",
            "unbounded-start-before-end",
            "unbounded-start-after-end",
            "unbounded-end-after-start",
            "unbounded-end-before-start",
            "fully-unbounded",
            "day-precision-bounds",
            "no-periods",
        ],
    )
    def test_membership(
        self,
        cfg_common_data_module: DictConfig,
        periods: list[dict[str, str | None]],
        day: str,
        *,
        expected: bool,
    ) -> None:
        cfg_common_data_module["data"]["split"]["train"] = periods
        dm = CommonDataModule(cfg_common_data_module)
        assert dm._in_train_periods(np.datetime64(day)) is expected


class TestTargetGroupValidation:
    """The target group must exist, be unique, and its variables must be selectable."""

    def test_missing_target_group_raises(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """A target group absent from the configured datasets should raise."""
        available_group = next(
            iter(cfg_common_data_module["data"]["datasets"].values())
        )["group_as"]
        cfg_common_data_module["variables"]["target"] = {"missing-target": ["mock_var"]}
        dm = CommonDataModule(cfg_common_data_module)

        with pytest.raises(ValueError, match="missing-target") as exc_info:
            _ = dm.target_group_name

        message = str(exc_info.value)
        assert str(available_group) in message

    def test_target_variables_returned_in_on_disk_order(
        self, mock_dataset: Path
    ) -> None:
        """target_variables must reflect on-disk order, not the order in `variables.target`.

        `mock_dataset` stores variables as (ice_conc, ice_thickness, temperature).
        Both target variables are requested here in the opposite order, but the
        target tensor built from them is on-disk ordered (see
        `test_target_variable_indices_use_data_order_not_config_order`), so
        `target_variables` - used as channel labels for that tensor - must match.
        """
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc", "ice_thickness", "temperature"],
            target_variables=["temperature", "ice_conc"],
        )
        dm = CommonDataModule(cfg)

        assert dm.target_variables == ["ice_conc", "temperature"]

    def test_empty_target_variable_list_raises(self, mock_dataset: Path) -> None:
        """An empty `variables.target[group]` must raise, not silently select none.

        `SingleDataset.subset()` treats an empty `variables` list as "use all
        variables", so leaving this unchecked would make `output_space` include every
        on-disk channel while `target_variable_indices` (derived from the empty list)
        stayed empty.
        """
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc", "ice_thickness", "temperature"],
            target_variables=[],
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="No variables were requested for group"):
            _ = dm.target_variables

    def test_multiple_target_groups_raises(self, mock_dataset: Path) -> None:
        """Only one target dataset group is supported; more than one must raise."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {
                "ds1": {"name": mock_dataset.stem, "group_as": "group1"},
                "ds2": {"name": mock_dataset.stem, "group_as": "group2"},
            },
            input_variables={
                "group1": ["ice_conc"],
                "group2": ["ice_conc"],
            },
            target_variables={
                "group1": ["ice_conc"],
                "group2": ["ice_conc"],
            },
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="exactly one target variable group"):
            _ = dm.target_group_name

    def test_target_variable_must_be_input_variable(self, mock_dataset: Path) -> None:
        """A target variable not requested as an input variable for its group raises."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc"],
            target_variables=["ice_conc", "ice_thickness"],
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="ice_thickness") as exc_info:
            _ = dm.target_variable_indices

        message = str(exc_info.value)
        assert "ice_conc" in message

    def test_omitted_input_selection_raises(self, mock_dataset: Path) -> None:
        """Omitting the target's own group from `variables.input` raises a clear error."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {
                "ds1": {"name": mock_dataset.stem, "group_as": "group1"},
                "ds2": {"name": mock_dataset.stem, "group_as": "group2"},
            },
            input_variables={"group2": ["ice_conc"]},
            target_variables={"group1": ["ice_conc"]},
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match=r"group1.*no available variables"):
            _ = dm.target_variables

    def test_empty_input_selection_for_target_group_raises(
        self, mock_dataset: Path
    ) -> None:
        """Requesting zero input variables for the target's own group also raises.

        Unlike omitting the key entirely, `group1` is present in `variables.input`
        but maps to an empty list, so it is filtered out of `datasets` the same way
        as an omitted group, hitting the same "no available variables" check in
        `target_variables`.
        """
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {
                "ds1": {"name": mock_dataset.stem, "group_as": "group1"},
                "ds2": {"name": mock_dataset.stem, "group_as": "group2"},
            },
            input_variables={"group1": [], "group2": ["ice_conc"]},
            target_variables={"group1": ["ice_conc"]},
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match=r"group1.*no available variables"):
            _ = dm.target_variables


class TestInputVariableSelection:
    """Requested input variables change what is loaded and validate against the data."""

    def test_unknown_input_group_is_rejected(self, mock_dataset: Path) -> None:
        """An input group not present in the configured datasets should raise."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {"ds1": {"name": mock_dataset.stem, "group_as": "group1"}},
            input_variables={"not-a-group": ["ice_conc"]},
            target_variables={"group1": ["ice_conc"]},
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="not-a-group") as exc_info:
            _ = dm.datasets

        assert "group1" in str(exc_info.value)

    def test_unknown_input_variable_is_rejected(self, mock_dataset: Path) -> None:
        """A variable not present in its dataset group should raise."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["not-a-variable"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        with pytest.raises(ValueError, match="not-a-variable") as exc_info:
            _ = dm.datasets

        message = str(exc_info.value)
        assert "ice_conc" in message

    def test_input_selection_changes_data_space_channels(
        self, mock_dataset: Path
    ) -> None:
        """Requesting a subset of input variables changes the DataSpace.

        The underlying `mock_dataset` fixture has three variables (`ice_conc`,
        `ice_thickness`, `temperature`); requesting only two of them as input should
        produce a DataSpace with 2 channels, not 3.
        """
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc", "temperature"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        assert dm.datasets["group1"].variable_names == ["ice_conc", "temperature"]
        [space] = dm.input_spaces
        assert space.channels == 2

    def test_datasets_unfiltered_ignores_variable_selection(
        self, mock_dataset: Path
    ) -> None:
        """`datasets_unfiltered` always exposes every variable, regardless of `variables.input`."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        assert dm.datasets["group1"].variable_names == ["ice_conc"]
        assert set(dm.datasets_unfiltered["group1"].variable_names) == {
            "ice_conc",
            "ice_thickness",
            "temperature",
        }

    def test_target_variable_indices_use_data_order_not_config_order(
        self, mock_dataset: Path
    ) -> None:
        """target_variable_indices must index into the underlying data's variable order.

        `mock_dataset` stores variables as (ice_conc, ice_thickness, temperature), and
        that storage order survives variable selection regardless of the order
        variables are requested in `variables.input`. Here `temperature` is requested
        before `ice_conc`, but still ends up second in `datasets[...].variable_names`.
        Indexing against the config-requested order instead of the actual data order
        would silently point at `ice_conc` (index 0) rather than `temperature`.
        """
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["temperature", "ice_conc"],
            target_variables=["temperature"],
        )
        dm = CommonDataModule(cfg)

        actual_order = dm.datasets["group1"].variable_names
        assert actual_order == ["ice_conc", "temperature"]
        assert dm.target_variable_indices == [1]
        assert actual_order[dm.target_variable_indices[0]] == "temperature"

    def test_group_with_no_requested_variables_is_excluded_from_datasets(
        self, mock_dataset: Path
    ) -> None:
        """A dataset group requesting an empty variable list is dropped from `datasets`."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {
                "ds1": {"name": mock_dataset.stem, "group_as": "group1"},
                "ds2": {"name": mock_dataset.stem, "group_as": "group2"},
            },
            input_variables={"group1": ["ice_conc"], "group2": []},
            target_variables={"group1": ["ice_conc"]},
        )
        dm = CommonDataModule(cfg)

        assert set(dm.datasets) == {"group1"}
        assert "group2" not in dm.datasets
        assert "group2" in dm.datasets_unfiltered


class TestDerivedProperties:
    """Properties derived from the resolved datasets: hemisphere, spaces, coordinates."""

    def test_hemisphere_returns_consistent_value(self, mock_dataset: Path) -> None:
        """A single dataset group's hemisphere is returned directly."""
        cfg = _single_group_config(
            mock_dataset, input_variables=["ice_conc"], target_variables=["ice_conc"]
        )
        dm = CommonDataModule(cfg)
        assert dm.hemisphere == "south"

    def test_hemisphere_raises_when_groups_disagree(
        self, mock_dataset: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mixed hemispheres across dataset groups must raise, not silently pick one."""
        cfg = _build_config(
            str(mock_dataset.parent.parent.parent),
            {
                "ds1": {"name": mock_dataset.stem, "group_as": "group1"},
                "ds2": {"name": mock_dataset.stem, "group_as": "group2"},
            },
            input_variables={"group1": ["ice_conc"], "group2": ["ice_conc"]},
            target_variables={"group1": ["ice_conc"]},
        )
        dm = CommonDataModule(cfg)
        monkeypatch.setattr(dm.datasets["group2"], "hemisphere", "north")

        with pytest.raises(ValueError, match="different hemisphere"):
            _ = dm.hemisphere

    def test_output_space_reflects_target_variables(self, mock_dataset: Path) -> None:
        """The output space is named after, and sized by, the target group/variables."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc", "temperature"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        assert dm.output_space.name == "group1"
        assert dm.output_space.channels == 1

    def test_latitudes_and_longitudes_keyed_by_group_name(
        self, mock_dataset: Path
    ) -> None:
        """Coordinates are returned per dataset group, matching the grid size."""
        cfg = _single_group_config(
            mock_dataset, input_variables=["ice_conc"], target_variables=["ice_conc"]
        )
        dm = CommonDataModule(cfg)

        assert set(dm.latitudes) == {"group1"}
        assert set(dm.longitudes) == {"group1"}
        assert len(dm.latitudes["group1"]) == len(dm.longitudes["group1"]) > 0


class TestDataLoaders:
    """Dataloader construction: worker assignment, shuffling, and channel content."""

    def test_assign_workers_updates_dataloaders(self, mock_dataset: Path) -> None:
        """assign_workers propagates to num_workers/persistent_workers/prefetch_factor."""
        cfg = _single_group_config(
            mock_dataset, input_variables=["ice_conc"], target_variables=["ice_conc"]
        )
        dm = CommonDataModule(cfg)

        dm.assign_workers(4)
        loader = dm.train_dataloader()
        assert loader.num_workers == 4
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 1

        dm.assign_workers(0)
        loader = dm.train_dataloader()
        assert loader.num_workers == 0
        assert loader.persistent_workers is False
        assert loader.prefetch_factor is None

    def test_only_train_dataloader_shuffles(self, mock_dataset: Path) -> None:
        """Only train_dataloader should shuffle; predict/test/val must stay sequential."""
        cfg = _single_group_config(
            mock_dataset, input_variables=["ice_conc"], target_variables=["ice_conc"]
        )
        dm = CommonDataModule(cfg)

        assert isinstance(dm.train_dataloader().sampler, RandomSampler)
        assert isinstance(dm.test_dataloader().sampler, SequentialSampler)
        assert isinstance(dm.val_dataloader().sampler, SequentialSampler)
        assert isinstance(dm.predict_dataloader().sampler, SequentialSampler)

    def test_dataloader_returns_only_selected_input_channels(
        self, mock_dataset: Path
    ) -> None:
        """A batch's input tensor has one channel per requested (not available) variable."""
        cfg = _single_group_config(
            mock_dataset,
            input_variables=["ice_conc", "temperature"],
            target_variables=["ice_conc"],
        )
        dm = CommonDataModule(cfg)

        batch = next(iter(dm.train_dataloader()))
        assert batch["group1"].shape[2] == 2  # NTCHW: [batch, time, channels, H, W]
        assert batch["target"].shape[2] == 1


class TestTargetMaskDir:
    """Mask directory resolution for the target group, including multi-dataset groups."""

    def test_derived_from_base_path_and_target_dataset_name(
        self, cfg_common_data_module: DictConfig
    ) -> None:
        """The mask dir is built from base_path and the target dataset's name."""
        dm = CommonDataModule(cfg_common_data_module)
        assert dm.mask_directory == Path("/mock/base/path/data/masks/mock")

    def test_picks_first_dataset_with_an_existing_mask(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With several datasets in the group, use the first that has a mask on disk."""
        cfg = _build_config(
            str(tmp_path),
            {
                "ds1": {"name": "sic_a", "group_as": "sic"},  # no mask on disk
                "ds2": {"name": "sic_b", "group_as": "sic"},  # has a mask
            },
            input_variables={"sic": ["mock_var"]},
            target_variables={"sic": ["mock_var"]},
        )
        mdir = mask_dir(tmp_path, "sic_b")
        mdir.mkdir(parents=True)
        np.save(mdir / "active_mask.npy", np.ones((4, 4), dtype=np.uint8))

        dm = CommonDataModule(cfg)
        with caplog.at_level(logging.WARNING):
            chosen = dm.mask_directory
        # Picked sic_b (the available one), not sic_a (the first listed).
        assert chosen == mask_dir(tmp_path, "sic_b")
        assert any("has 2 datasets" in r.getMessage() for r in caplog.records)

    def test_falls_back_to_first_when_no_masks_exist(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No dataset has a mask: fall back to the first (old behaviour) and warn."""
        cfg = _build_config(
            str(tmp_path),
            {
                "ds1": {"name": "sic_a", "group_as": "sic"},
                "ds2": {"name": "sic_b", "group_as": "sic"},
            },
            input_variables={"sic": ["mock_var"]},
            target_variables={"sic": ["mock_var"]},
        )
        dm = CommonDataModule(cfg)
        with caplog.at_level(logging.WARNING):
            chosen = dm.mask_directory
        assert chosen == mask_dir(tmp_path, "sic_a")
        assert any("has 2 datasets" in r.getMessage() for r in caplog.records)


class TestCommonDataModuleClimatology:
    """Tests for the CommonDataModule.climatology calendar-day statistics."""

    def test_table_shapes_and_dtypes(self, climatology_zarr: Path) -> None:
        """Mean and std are float32 [366, C, H, W] tables, even with no leap years."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))

        climatology = _climatology(dm)
        for table in (climatology.mean, climatology.std):
            assert table.shape == (366, 2, 2, 2)
            assert table.dtype == np.float32
        # No date falls on 29 February, but smoothing fills it from its neighbours
        assert climatology.n_dates[FEB_29] == 0

    @pytest.mark.parametrize(
        ("target_variables", "channels"),
        [(CLIMATOLOGY_VARIABLES, [0, 1]), (["ice_thickness"], [1])],
        ids=["all-variables", "one-variable"],
    )
    def test_uses_normalised_target_fields(
        self,
        climatology_zarr: Path,
        monkeypatch: pytest.MonkeyPatch,
        target_variables: list[str],
        channels: list[int],
    ) -> None:
        """Statistics are computed from the normalised target variables only."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _climatology_cfg(base_path, TRAIN_PERIODS, target_variables)
        )
        calls = _spy_from_dataset(monkeypatch)

        _climatology(dm)
        [(dataset, dates)] = calls
        expected = _normalised_rows(climatology_zarr, _period_dates(TRAIN_PERIODS))
        np.testing.assert_allclose(
            dataset.get_tchw(dates), expected[:, channels], rtol=0, atol=1e-6
        )

    def test_uses_union_of_train_periods(
        self, climatology_zarr: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only dates inside the train-period union are used (e.g. not July 2019)."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))
        calls = _spy_from_dataset(monkeypatch)

        _climatology(dm)
        [(_, dates)] = calls
        np.testing.assert_array_equal(
            _as_days(dates), _as_days(_period_dates(TRAIN_PERIODS))
        )
        assert max(_as_days(dates)) == np.datetime64("2019-06-30")

    def test_missing_dates_excluded(
        self, climatology_zarr: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A date missing from the dataset is never passed on for averaging."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))
        calls = _spy_from_dataset(monkeypatch)

        _climatology(dm)
        [(_, dates)] = calls
        days = _as_days(dates)
        assert np.datetime64("2017-03-14") in days
        assert np.datetime64("2017-03-15") not in days
        assert np.datetime64("2017-03-16") in days

    def test_missing_calendar_day_returns_none(
        self, climatology_zarr: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A calendar day with no available dates in the period gives no climatology."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _climatology_cfg(base_path, [{"start": "2017-01-01", "end": "2017-01-31"}])
        )
        with caplog.at_level("WARNING"):
            assert dm.climatology is None
        # January data covers 25 December to 7 February: 45 of 366 calendar days
        assert "321 calendar days have pixels with no finite values" in caplog.text

    def test_always_nan_pixel_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A pixel with no finite value on any date gives no climatology at all.

        This is intentional: a climatology with NaN pixels is never returned, even if
        those pixels are NaN on every date (e.g. fill values).
        """
        zarr_path = build_zarr(
            tmp_path / "data" / "anemoi" / "sic_south.zarr",
            make_climatology_data_dict(
                CLIMATOLOGY_START, CLIMATOLOGY_END, CLIMATOLOGY_MISSING
            ),
            full_dates=_all_dates(),
            missing_dates=CLIMATOLOGY_MISSING,
        )
        # Blank one pixel of the first channel on every date, keeping the per-channel
        # statistics finite so that normalisation leaves the other pixels intact
        store = zarr.open_group(str(zarr_path), mode="r+")
        data = np.asarray(store["data"])
        data[:, 0, 0, 0] = np.nan
        store["data"][:] = data
        store["minimum"][:] = np.nanmin(data, axis=(0, 2, 3)).astype(np.float64)
        store["maximum"][:] = np.nanmax(data, axis=(0, 2, 3)).astype(np.float64)

        dm = CommonDataModule(_climatology_cfg(tmp_path, TRAIN_PERIODS))
        with caplog.at_level("WARNING"):
            assert dm.climatology is None
        assert "366 calendar days have pixels with no finite values" in caplog.text

    def test_returns_none_when_no_dates_in_train_periods(
        self, climatology_zarr: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If no available dates fall in the training periods, there is no climatology."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _climatology_cfg(base_path, [{"start": "2030-01-01", "end": "2030-12-31"}])
        )
        with caplog.at_level("WARNING"):
            assert dm.climatology is None
        assert "none of the configured training periods" in caplog.text

    def test_time_component_bounds_match_day_precision(
        self, climatology_zarr: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bounds carrying a time component behave like day-precision bounds.

        A start bound of ``2017-01-01T12:00:00`` must still include 2017-01-01, so the
        dates used are identical to those selected by plain day-precision bounds.
        """
        base_path = climatology_zarr.parents[2]
        timed_periods: list[dict[str, str | None]] = [
            {"start": "2017-01-01T12:00:00", "end": "2018-12-31T12:00:00"},
            {"start": "2019-01-01T00:00:00", "end": "2019-06-30T23:59:59"},
        ]
        dm = CommonDataModule(_climatology_cfg(base_path, timed_periods))
        calls = _spy_from_dataset(monkeypatch)

        _climatology(dm)
        [(_, dates)] = calls
        np.testing.assert_array_equal(
            _as_days(dates), _as_days(_period_dates(TRAIN_PERIODS))
        )

    def test_dataloaders_include_climatology(self, climatology_zarr: Path) -> None:
        """Every split's dataloader batches contain a correctly-shaped climatology key."""
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(_climatology_cfg(base_path, TRAIN_PERIODS))
        for name in ("train", "val", "test", "predict"):
            loader = getattr(dm, f"{name}_dataloader")()
            batch = next(iter(loader))
            assert "climatology" in batch
            # shape: batch x n_forecast_steps x C_target x H x W
            assert batch["climatology"].shape == (2, 1, 2, 2, 2)

    def test_dataloaders_degrade_gracefully_when_climatology_unavailable(
        self, climatology_zarr: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A train window missing a calendar day must not break other models' loaders.

        ``CommonDataModule.climatology`` is None in this case (see
        ``test_missing_calendar_day_returns_none``). Building a dataloader is a shared
        code path used by every model, not just the Climatology baseline, so it must
        omit the ``climatology`` batch key with a warning instead of crashing.
        """
        base_path = climatology_zarr.parents[2]
        dm = CommonDataModule(
            _climatology_cfg(base_path, [{"start": "2017-01-01", "end": "2017-01-31"}])
        )
        with caplog.at_level("WARNING"):
            loader = dm.train_dataloader()
            batch = next(iter(loader))
        assert "climatology" not in batch
        assert "Cannot build climatology" in caplog.text
