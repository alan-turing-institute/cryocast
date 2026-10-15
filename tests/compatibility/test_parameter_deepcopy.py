"""Test MPS deepcopy patch without requiring Apple or Intel GPU hardware."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from cryocast.compatibility.torch.patch_parameter_deepcopy import (
    patch_parameter_deepcopy,
)


def test_cpu_only_system_does_not_patch_deepcopy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the named compatibility behavior without device access."""
    original = nn.Parameter.__deepcopy__
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    patch_parameter_deepcopy()
    assert nn.Parameter.__deepcopy__ is original
    original_param = nn.Parameter(torch.tensor([1.0, 2.0]), requires_grad=False)
    cloned = deepcopy(original_param)
    assert cloned is not original_param
    assert cloned.requires_grad is False
    torch.testing.assert_close(cloned, original_param)


def test_patch_is_idempotent_when_mps_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the named compatibility behavior without device access."""
    original = nn.Parameter.__deepcopy__

    def unpatched(self: nn.Parameter, _memo: dict) -> nn.Parameter:
        return nn.Parameter(self.detach().clone(), self.requires_grad)

    with monkeypatch.context() as patcher:
        patcher.setattr(nn.Parameter, "__deepcopy__", unpatched)
        patcher.setattr(torch.backends.mps, "is_available", lambda: True)
        patch_parameter_deepcopy()
        patched = nn.Parameter.__deepcopy__
        assert patched is not unpatched
        assert getattr(patched, "is_patched", False)
        patch_parameter_deepcopy()
        assert nn.Parameter.__deepcopy__ is patched
    assert nn.Parameter.__deepcopy__ is original


@pytest.mark.parametrize("requires_grad", [True, False])
def test_mps_patch_copies_cpu_parameter_via_mocked_transfer(
    monkeypatch: pytest.MonkeyPatch,
    requires_grad: bool,  # noqa: FBT001
) -> None:
    """The copy must retain values, grad flag, device and separate storage."""
    original = nn.Parameter.__deepcopy__
    transfer_calls: list[str] = []
    real_to = torch.Tensor.to

    def fake_to(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if args and args[0] == "mps":
            transfer_calls.append("mps")
            return tensor
        return real_to(tensor, *args, **kwargs)  # type: ignore[call-overload]

    with monkeypatch.context() as patcher:
        patcher.setattr(torch.backends.mps, "is_available", lambda: True)
        patcher.setattr(nn.Parameter, "__deepcopy__", original)
        patcher.setattr(torch.Tensor, "to", fake_to)
        patch_parameter_deepcopy()
        param = nn.Parameter(torch.tensor([[3.5, -2.0]]), requires_grad=requires_grad)
        cloned = deepcopy(param)
        assert transfer_calls == ["mps"]
        assert cloned.device == param.device
        assert cloned.requires_grad == requires_grad
        assert cloned is not param
        torch.testing.assert_close(cloned, param)
        with torch.no_grad():
            cloned.add_(1)
        assert not torch.equal(cloned, param)
    assert nn.Parameter.__deepcopy__ is original


def test_patch_returns_unchanged_value_for_noncontiguous_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the named compatibility behavior without device access."""
    original = nn.Parameter.__deepcopy__
    actual_to = torch.Tensor.to

    def local_transfer(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        if args and args[0] == "mps":
            return tensor
        return actual_to(tensor, *args, **kwargs)  # type: ignore[call-overload]

    with monkeypatch.context() as patcher:
        patcher.setattr(torch.backends.mps, "is_available", lambda: True)
        patcher.setattr(nn.Parameter, "__deepcopy__", original)
        patcher.setattr(torch.Tensor, "to", local_transfer)
        patch_parameter_deepcopy()
        param = nn.Parameter(torch.arange(12.0).reshape(3, 4).T)
        assert not param.is_contiguous()
        copied = deepcopy(param)
        assert copied.is_contiguous()
        torch.testing.assert_close(copied, param)
    assert nn.Parameter.__deepcopy__ is original
