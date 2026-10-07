import pytest
import torch

from cryocast.compatibility.torch.patch_open_file_limit import (
    MIN_OPEN_FILE_LIMIT,
    patch_open_file_limit,
)

resource = pytest.importorskip("resource")


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
