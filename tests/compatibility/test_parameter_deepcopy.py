"""Test MPS deepcopy patch without requiring Apple or Intel GPU hardware."""

from collections.abc import Generator
from copy import deepcopy

import pytest
import torch
from torch import nn

from cryocast.compatibility.torch.patch_parameter_deepcopy import (
    patch_parameter_deepcopy,
)


@pytest.fixture(autouse=True)
def isolate_parameter_patch(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Use a CPU stand-in so prior CLI imports cannot trigger real MPS transfers."""
    incoming = nn.Parameter.__deepcopy__

    def cpu_copy(parameter: nn.Parameter, _memo: dict) -> nn.Parameter:
        return nn.Parameter(parameter.detach().clone(), parameter.requires_grad)

    with monkeypatch.context() as patcher:
        patcher.setattr(nn.Parameter, "__deepcopy__", cpu_copy)
        yield
    assert nn.Parameter.__deepcopy__ is incoming


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_mocked_copy_preserves_shared_references_and_isolates_gradients(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    *,
    requires_grad: bool,
) -> None:
    """A copied parameter graph keeps aliases, dtype and independent value/grad storage."""
    incoming = nn.Parameter.__deepcopy__
    native_to = torch.Tensor.to
    transfers: list[object] = []

    def local_transfer(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        if args:
            transfers.append(args[0])
        if args and args[0] == "mps":
            return tensor
        return native_to(tensor, *args, **kwargs)  # type: ignore[call-overload]

    parameter = nn.Parameter(
        torch.arange(12, dtype=dtype).reshape(3, 4).T, requires_grad=requires_grad
    )
    original_values = parameter.detach().clone()
    if requires_grad:
        parameter.square().sum().backward()
    original_gradient = None if parameter.grad is None else parameter.grad.clone()
    with monkeypatch.context() as patcher:
        patcher.setattr(torch.backends.mps, "is_available", lambda: True)
        patcher.setattr(nn.Parameter, "__deepcopy__", incoming)
        patcher.setattr(torch.Tensor, "to", local_transfer)
        patch_parameter_deepcopy()
        copied = deepcopy({"first": parameter, "second": parameter})
    first = copied["first"]
    assert isinstance(first, nn.Parameter)
    assert copied["second"] is first
    assert first is not parameter
    assert first.data_ptr() != parameter.data_ptr()
    assert first.dtype == dtype
    assert first.device == parameter.device
    assert first.is_leaf
    assert first.is_contiguous()
    assert first.requires_grad == requires_grad
    assert first.grad is None
    assert transfers == ["mps", parameter.device]
    torch.testing.assert_close(first, original_values, rtol=0, atol=0)
    if requires_grad:
        first.sum().backward()
        torch.testing.assert_close(first.grad, torch.ones_like(first), rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, original_gradient, rtol=0, atol=0)
    with torch.no_grad():
        first.add_(1)
    torch.testing.assert_close(parameter, original_values, rtol=0, atol=0)
    assert nn.Parameter.__deepcopy__ is incoming
