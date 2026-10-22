import sys
from unittest.mock import MagicMock

import pytest
import torch

from cryocast.compatibility.torch.patch_open_file_limit import (
    MIN_OPEN_FILE_LIMIT,
    patch_open_file_limit,
)

resource = pytest.importorskip("resource")


@pytest.fixture(autouse=True)
def isolate_sharing_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never change the process-wide sharing strategy while testing resource limits."""
    monkeypatch.setattr(torch.multiprocessing, "set_sharing_strategy", MagicMock())


class TestPatchOpenFileLimit:
    def test_raises_soft_limit_when_below_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        low_soft = MIN_OPEN_FILE_LIMIT // 2
        hard = resource.RLIM_INFINITY
        calls: list = []
        monkeypatch.setattr(resource, "getrlimit", lambda _: (low_soft, hard))
        monkeypatch.setattr(resource, "setrlimit", lambda *args: calls.append(args))
        patch_open_file_limit()
        assert calls == [(resource.RLIMIT_NOFILE, (MIN_OPEN_FILE_LIMIT, hard))]

    def test_does_not_raise_when_already_at_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list = []
        monkeypatch.setattr(
            resource,
            "getrlimit",
            lambda _: (MIN_OPEN_FILE_LIMIT, resource.RLIM_INFINITY),
        )
        monkeypatch.setattr(resource, "setrlimit", lambda *args: calls.append(args))
        patch_open_file_limit()
        assert calls == []

    def test_caps_at_hard_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        low_soft = 256
        low_hard = MIN_OPEN_FILE_LIMIT // 2
        calls: list = []
        monkeypatch.setattr(resource, "getrlimit", lambda _: (low_soft, low_hard))
        monkeypatch.setattr(resource, "setrlimit", lambda *args: calls.append(args))
        patch_open_file_limit()
        assert calls == [(resource.RLIMIT_NOFILE, (low_hard, low_hard))]

    def test_warning_on_oserror(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls: list = []
        monkeypatch.setattr(
            resource,
            "getrlimit",
            lambda _: (_ for _ in ()).throw(OSError("permission denied")),
        )
        monkeypatch.setattr(resource, "setrlimit", lambda *args: calls.append(args))
        patch_open_file_limit()
        assert calls == []
        assert "Could not raise open-file limit" in caplog.text

    def test_sets_file_system_sharing_strategy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list = []
        monkeypatch.setattr(
            resource,
            "getrlimit",
            lambda _: (MIN_OPEN_FILE_LIMIT, resource.RLIM_INFINITY),
        )
        monkeypatch.setattr(torch.multiprocessing, "set_sharing_strategy", calls.append)
        patch_open_file_limit()
        assert calls == ["file_system"]

    def test_sets_file_system_sharing_strategy_even_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list = []
        monkeypatch.setattr(
            resource, "getrlimit", lambda _: (_ for _ in ()).throw(OSError)
        )
        monkeypatch.setattr(torch.multiprocessing, "set_sharing_strategy", calls.append)
        patch_open_file_limit()
        assert calls == ["file_system"]

    @pytest.mark.parametrize("error", [RuntimeError, ValueError])
    def test_warns_if_multiprocessing_sharing_strategy_fails(
        self,
        error: type[Exception],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unavailable sharing strategy warns without failing application startup."""
        monkeypatch.setattr(
            resource,
            "getrlimit",
            lambda _: (MIN_OPEN_FILE_LIMIT, resource.RLIM_INFINITY),
        )

        def fail_strategy(_strategy: str) -> None:
            msg = "sharing strategy not supported"
            raise error(msg)

        monkeypatch.setattr(
            torch.multiprocessing, "set_sharing_strategy", fail_strategy
        )
        patch_open_file_limit()
        assert "Could not set torch sharing strategy" in caplog.text


@pytest.mark.parametrize("stage", ["read", "write"])
@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_resource_error_still_configures_sharing_without_changing_limits(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stage: str,
    error_type: type[Exception],
) -> None:
    """OS-limit errors warn, avoid real OS changes, and still attempt sharing setup."""
    get_limit = MagicMock(return_value=(256, MIN_OPEN_FILE_LIMIT))
    set_limit = MagicMock()
    error = error_type("resource operation rejected")
    (get_limit if stage == "read" else set_limit).side_effect = error
    sharing = MagicMock()
    monkeypatch.setattr(resource, "getrlimit", get_limit)
    monkeypatch.setattr(resource, "setrlimit", set_limit)
    monkeypatch.setattr(torch.multiprocessing, "set_sharing_strategy", sharing)

    patch_open_file_limit()

    get_limit.assert_called_once_with(resource.RLIMIT_NOFILE)
    if stage == "write":
        set_limit.assert_called_once_with(
            resource.RLIMIT_NOFILE, (MIN_OPEN_FILE_LIMIT, MIN_OPEN_FILE_LIMIT)
        )
    else:
        set_limit.assert_not_called()
    sharing.assert_called_once_with("file_system")
    assert "Could not raise open-file limit" in caplog.text
    assert "resource operation rejected" in caplog.text


def test_missing_resource_module_still_configures_sharing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Simulate the optional resource import failing without changing the host OS."""
    sharing = MagicMock()
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setattr(torch.multiprocessing, "set_sharing_strategy", sharing)

    patch_open_file_limit()

    sharing.assert_called_once_with("file_system")
    assert "Could not raise open-file limit" in caplog.text
