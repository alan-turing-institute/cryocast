import re
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from wandb.wandb_run import Run

from icenet_mp.utils import (
    datetime_from_npdatetime,
    get_device_name,
    get_timestamp,
    get_wandb_run,
    iso_from_date,
    mask_dir,
    npdatetime_from_datetime,
    safe_nanmax,
    safe_nanmin,
    sanitise_filename,
    to_list,
)


class TestDatetimeFromNpdatetime:
    def test_returns_utc_datetime(self) -> None:
        """Convert NumPy datetimes to timezone-aware UTC datetimes."""
        result = datetime_from_npdatetime(np.datetime64("2026-08-21T12:34:56.789"))

        assert result.tzinfo is UTC
        assert result.microsecond == 789000


class TestFormatDateKey:
    def test_date_object(self) -> None:
        """Format a plain date object as an ISO date string."""
        result = iso_from_date(date(2023, 12, 25))

        assert result == "2023-12-25"

    def test_datetime_object_drops_time(self) -> None:
        """Format a datetime object, stripping the time component."""
        result = iso_from_date(datetime(2023, 12, 25, 14, 30))

        assert result == "2023-12-25"


class TestNpdatetimeFromDatetime:
    def test_converts_naive_datetime(self) -> None:
        """Convert a naive datetime to numpy datetime64."""
        result = npdatetime_from_datetime(datetime(2026, 8, 21, 12, 34, 56))

        assert result == np.datetime64("2026-08-21T12:34:56")

    def test_drops_tzinfo_without_shifting_the_wall_clock_time(self) -> None:
        """Drop tzinfo from an aware datetime without converting to another zone."""
        result = npdatetime_from_datetime(datetime(2026, 8, 21, 12, 34, 56, tzinfo=UTC))

        assert result == np.datetime64("2026-08-21T12:34:56")


class TestMaskDir:
    def test_builds_expected_path(self, tmp_path: Path) -> None:
        """Build the preprocessing mask directory beneath the configured root."""
        assert mask_dir(tmp_path, "sic-ssmis") == (
            tmp_path / "data" / "masks" / "sic-ssmis"
        )


class TestGetDeviceName:
    @pytest.mark.parametrize(
        ("accelerator_name", "expected"),
        [("cpu", "CPU"), ("mps", "Apple Silicon GPU"), ("auto", "CPU")],
    )
    def test_for_non_cuda_accelerators(
        self, accelerator_name: str, expected: str
    ) -> None:
        """Return readable names for non-CUDA accelerator selections."""
        assert get_device_name(accelerator_name) == expected

    @pytest.mark.parametrize(
        ("accelerator_name", "torch_module"),
        [("cuda", torch.cuda), ("xpu", torch.xpu)],
        ids=["cuda", "xpu"],
    )
    def test_returns_the_queried_device_name(
        self,
        accelerator_name: str,
        torch_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return the accelerator's own reported name when the query succeeds."""
        monkeypatch.setattr(torch_module, "get_device_name", lambda: "Custom GPU")

        assert get_device_name(accelerator_name) == "Custom GPU"

    @pytest.mark.parametrize(
        ("accelerator_name", "torch_module", "expected"),
        [
            ("cuda", torch.cuda, "Unknown CUDA device"),
            ("xpu", torch.xpu, "Unknown XPU device"),
        ],
        ids=["cuda", "xpu"],
    )
    def test_handles_unavailable_accelerator(
        self,
        accelerator_name: str,
        torch_module: ModuleType,
        expected: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fall back cleanly when the accelerator's device name cannot be queried."""

        def _raise_assertion() -> str:
            raise AssertionError

        monkeypatch.setattr(torch_module, "get_device_name", _raise_assertion)

        assert get_device_name(accelerator_name) == expected


class TestGetTimestamp:
    def test_has_expected_utc_format(self) -> None:
        """Format generated timestamps using the expected UTC pattern."""
        assert re.fullmatch(r"\d{8}_\d{6}", get_timestamp())


class TestGetWandbRun:
    def test_returns_none_without_wandb_logger(self) -> None:
        """Return no W&B run when the trainer has no W&B logger."""
        trainer = MagicMock(spec=Trainer)
        trainer.loggers = [object()]

        assert get_wandb_run(trainer) is None

    def test_returns_the_run_from_a_wandb_logger(self) -> None:
        """Return the active W&B run when a WandbLogger is present."""
        run = MagicMock(spec=Run)
        wandb_logger = MagicMock(spec=WandbLogger)
        wandb_logger.experiment = run
        trainer = MagicMock(spec=Trainer)
        trainer.loggers = [wandb_logger]

        assert get_wandb_run(trainer) is run

    def test_returns_none_when_wandb_logger_has_no_run_experiment(self) -> None:
        """Return None when the WandbLogger's experiment isn't a Run (e.g. offline)."""
        wandb_logger = MagicMock(spec=WandbLogger)
        wandb_logger.experiment = None
        trainer = MagicMock(spec=Trainer)
        trainer.loggers = [wandb_logger]

        assert get_wandb_run(trainer) is None

    def test_returns_the_first_matching_run_when_multiple_loggers_present(
        self,
    ) -> None:
        """Return the first WandbLogger's run when more than one logger qualifies."""
        first_run = MagicMock(spec=Run)
        first_logger = MagicMock(spec=WandbLogger)
        first_logger.experiment = first_run
        second_logger = MagicMock(spec=WandbLogger)
        second_logger.experiment = MagicMock(spec=Run)
        trainer = MagicMock(spec=Trainer)
        trainer.loggers = [object(), first_logger, second_logger]

        assert get_wandb_run(trainer) is first_run


class TestToList:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("ice_conc", ["ice_conc"]), (["ice_conc", "2t"], ["ice_conc", "2t"])],
        ids=["scalar", "list"],
    )
    def test_to_list(self, value: str | list[str], expected: list[str]) -> None:
        """Normalize scalar strings and string lists to list form."""
        assert to_list(value) == expected

    def test_to_list_returns_the_same_list_object_unchanged(self) -> None:
        """Return the given list unchanged (no copy) rather than wrapping it again."""
        values = ["ice_conc", "2t"]

        assert to_list(values) is values


class TestSafeNanmin:
    def test_normal_array(self) -> None:
        """Return the true minimum for a fully finite array."""
        result = safe_nanmin(np.array([3.0, 1.0, 2.0]))

        assert result == pytest.approx(1.0)

    def test_ignores_nan(self) -> None:
        """NaN entries are ignored when a finite value is present."""
        result = safe_nanmin(np.array([np.nan, 5.0, 2.0]))

        assert result == pytest.approx(2.0)

    def test_all_nan_returns_default(self) -> None:
        """An all-NaN array falls back to the default value."""
        result = safe_nanmin(np.array([np.nan, np.nan]), default=-9.0)

        assert result == pytest.approx(-9.0)

    def test_empty_array_returns_default(self) -> None:
        """An empty array falls back to the default value."""
        result = safe_nanmin(np.array([]), default=7.0)

        assert result == pytest.approx(7.0)

    def test_all_infinite_returns_default(self) -> None:
        """An array of only +/-inf falls back to the default value."""
        result = safe_nanmin(np.array([np.inf, -np.inf]), default=3.0)

        assert result == pytest.approx(3.0)

    def test_mixed_negative_infinity_and_finite_ignores_the_infinity(self) -> None:
        """-inf entries must not suppress the true finite minimum."""
        result = safe_nanmin(np.array([-np.inf, 10.0]), default=0.0)

        assert result == pytest.approx(10.0)

    def test_mixed_negative_infinity_nan_and_finite_ignores_both(self) -> None:
        """-inf and NaN entries together must still yield the true finite minimum."""
        result = safe_nanmin(np.array([-np.inf, np.nan, 4.0]), default=0.0)

        assert result == pytest.approx(4.0)


class TestSafeNanmax:
    def test_normal_array(self) -> None:
        """Return the true maximum for a fully finite array."""
        result = safe_nanmax(np.array([3.0, 1.0, 2.0]))

        assert result == pytest.approx(3.0)

    def test_ignores_nan(self) -> None:
        """NaN entries are ignored when a finite value is present."""
        result = safe_nanmax(np.array([np.nan, 5.0, 2.0]))

        assert result == pytest.approx(5.0)

    def test_all_nan_returns_default(self) -> None:
        """An all-NaN array falls back to the default value."""
        result = safe_nanmax(np.array([np.nan, np.nan]), default=42.0)

        assert result == pytest.approx(42.0)

    def test_empty_array_returns_default(self) -> None:
        """An empty array falls back to the default value."""
        result = safe_nanmax(np.array([]), default=8.0)

        assert result == pytest.approx(8.0)

    def test_all_infinite_returns_default(self) -> None:
        """An array of only +/-inf falls back to the default value."""
        result = safe_nanmax(np.array([np.inf, -np.inf]), default=6.0)

        assert result == pytest.approx(6.0)

    def test_mixed_positive_infinity_and_finite_ignores_the_infinity(self) -> None:
        """+inf entries must not suppress the true finite maximum."""
        result = safe_nanmax(np.array([np.inf, 10.0]), default=1.0)

        assert result == pytest.approx(10.0)

    def test_mixed_positive_infinity_nan_and_finite_ignores_both(self) -> None:
        """+inf and NaN entries together must still yield the true finite maximum."""
        result = safe_nanmax(np.array([np.inf, np.nan, 4.0]), default=1.0)

        assert result == pytest.approx(4.0)


class TestSanitiseFilename:
    def test_replaces_a_single_unsafe_character(self) -> None:
        """Replace one unsafe character with an underscore."""
        assert sanitise_filename("a:b") == "a_b"

    def test_collapses_a_run_of_unsafe_characters(self) -> None:
        """Collapse a run of consecutive unsafe characters into one underscore."""
        assert sanitise_filename("a :/b") == "a_b"

    def test_leaves_safe_characters_unchanged(self) -> None:
        """Leave letters, digits, underscores, periods and hyphens untouched."""
        assert sanitise_filename("Safe-Name_123.png") == "Safe-Name_123.png"
