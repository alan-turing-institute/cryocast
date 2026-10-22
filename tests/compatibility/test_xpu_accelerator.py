"""Exercise Intel XPU Lightning support without Intel accelerator hardware."""

import pytest
import torch
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.accelerators import AcceleratorRegistry

from cryocast.compatibility.lightning import register_accelerators
from cryocast.compatibility.lightning.xpu_accelerator import XPUAccelerator


@pytest.mark.parametrize("requested", [0, "0", "[]"])
def test_zero_devices_do_not_require_xpu(
    requested: int | str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the named compatibility behavior without device access."""
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 0)
    assert XPUAccelerator.parse_devices(requested) is None


def test_unavailable_xpu_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the named compatibility behavior without device access."""
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 0)
    with pytest.raises(MisconfigurationException, match="none are available"):
        XPUAccelerator.parse_devices(1)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(1, [0]), (2, [0, 1]), ("1", [0]), ("-1", [0, 1, 2]), ([0, 2], [0, 2])],
)
def test_parse_valid_devices(
    requested: int | str | list[int],
    expected: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the named compatibility behavior without device access."""
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 3)
    assert XPUAccelerator.parse_devices(requested) == expected


@pytest.mark.parametrize("requested", [[0, 0], [3], 4, "4"])
def test_invalid_or_duplicate_devices(
    requested: int | str | list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the named compatibility behavior without device access."""
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 3)
    with pytest.raises(MisconfigurationException):
        XPUAccelerator.parse_devices(requested)


@pytest.mark.parametrize("requested", [1, [0, 2]])
def test_parallel_devices(requested: int | list[int]) -> None:
    """Check the named compatibility behavior without device access."""
    devices = XPUAccelerator.get_parallel_devices(requested)
    indexes = [0] if requested == 1 else [0, 2]
    assert devices == [torch.device("xpu", index) for index in indexes]


def test_device_count_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the named compatibility behavior without device access."""
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 4)
    assert XPUAccelerator.auto_device_count() == 4
    assert XPUAccelerator.is_available() is True
    assert XPUAccelerator.name() == "xpu"


@pytest.mark.parametrize("count", [0, 2])
def test_availability(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """Check the named compatibility behavior without device access."""
    monkeypatch.setattr(torch.xpu, "device_count", lambda: count)
    assert XPUAccelerator.is_available() is (count > 0)


def test_unavailable_backend_attribute_is_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the named compatibility behavior without device access."""

    def missing_device_count() -> int:
        msg = "No xpu backend"
        raise AttributeError(msg)

    monkeypatch.setattr(torch.xpu, "device_count", missing_device_count)
    assert XPUAccelerator.is_available() is False


def test_setup_rejects_cpu_device() -> None:
    """Check the named compatibility behavior without device access."""
    with pytest.raises(MisconfigurationException, match="should be xpu"):
        XPUAccelerator().setup_device(torch.device("cpu"))


def test_setup_rejects_unspecified_index() -> None:
    """Check the named compatibility behavior without device access."""
    with pytest.raises(MisconfigurationException, match="index"):
        XPUAccelerator().setup_device(torch.device("xpu"))


@pytest.mark.parametrize(("index", "expected_backend_index"), [(0, -1), (1, 0), (2, 1)])
def test_setup_preserves_existing_device_index_offset(
    monkeypatch: pytest.MonkeyPatch, index: int, expected_backend_index: int
) -> None:
    """Preserve CryoCast's existing mapping without claiming XPU hardware validation."""
    # The maintainer reports this offset is needed on their XPU system.
    # Mocked calls cover the current contract, not device availability or execution.
    calls: list[int] = []
    monkeypatch.setattr(torch.xpu, "set_device", calls.append)
    XPUAccelerator().setup_device(torch.device("xpu", index))
    assert calls == [expected_backend_index]


def test_stats_use_requested_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the named compatibility behavior without device access."""
    calls: list[torch.device] = []
    wanted = torch.device("xpu", 1)

    def memory_stats(device: torch.device) -> dict[str, int]:
        calls.append(device)
        return {"allocated_bytes": 123}

    monkeypatch.setattr(torch.xpu, "memory_stats", memory_stats)
    assert XPUAccelerator().get_device_stats(wanted) == {"allocated_bytes": 123}
    assert calls == [wanted]


@pytest.mark.parametrize("initialized", [False, True])
def test_teardown_cleans_up_distributed_backend_when_needed(
    monkeypatch: pytest.MonkeyPatch,
    initialized: bool,  # noqa: FBT001
) -> None:
    """Check the named compatibility behavior without device access."""
    cleaned: list[str] = []
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: cleaned.append("cache"))
    monkeypatch.setattr(
        "cryocast.compatibility.lightning.xpu_accelerator.dist.is_initialized",
        lambda: initialized,
    )
    monkeypatch.setattr(
        "cryocast.compatibility.lightning.xpu_accelerator.dist.destroy_process_group",
        lambda: cleaned.append("distributed"),
    )
    XPUAccelerator().teardown()
    assert cleaned == (["cache", "distributed"] if initialized else ["cache"])


def test_registration_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the named compatibility behavior without device access."""
    calls: list[tuple[str, type, str]] = []
    available = ["cpu"]
    monkeypatch.setattr(
        AcceleratorRegistry,
        "available_accelerators",
        available.copy,
    )

    def register(name: str, kind: type, *, description: str) -> None:
        calls.append((name, kind, description))
        available.append(name)

    monkeypatch.setattr(AcceleratorRegistry, "register", register)
    register_accelerators()
    register_accelerators()
    assert calls == [("xpu", XPUAccelerator, XPUAccelerator.description)]
    assert available == ["cpu", "xpu"]


def test_registration_skips_existing_xpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the named compatibility behavior without device access."""
    monkeypatch.setattr(
        AcceleratorRegistry,
        "available_accelerators",
        lambda: ["cpu", "xpu"],
    )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("An existing accelerator must not be registered again")

    monkeypatch.setattr(AcceleratorRegistry, "register", unexpected)
    register_accelerators()


@pytest.mark.parametrize("operation", ["setup", "stats"])
def test_backend_errors_propagate_without_claiming_a_working_device(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A rejected mocked backend operation must retain its error and device mapping."""
    error = RuntimeError("XPU backend operation failed")
    calls: list[object] = []

    def fail(argument: object) -> None:
        calls.append(argument)
        raise error

    device = torch.device("xpu", 2)
    accelerator = XPUAccelerator()
    if operation == "setup":
        monkeypatch.setattr(torch.xpu, "set_device", fail)
        with pytest.raises(
            RuntimeError, match="XPU backend operation failed"
        ) as caught:
            accelerator.setup_device(device)
        assert calls == [1]
    else:
        monkeypatch.setattr(torch.xpu, "memory_stats", fail)
        with pytest.raises(
            RuntimeError, match="XPU backend operation failed"
        ) as caught:
            accelerator.get_device_stats(device)
        assert calls == [device]
    assert caught.value is error
