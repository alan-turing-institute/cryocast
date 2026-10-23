"""CPU-only regression tests for the interpolation compatibility patch."""

from collections.abc import Generator
from typing import Any

import pytest
import torch
from torch.nn import functional

from cryocast.compatibility.torch.patch_interpolate_antialias import (
    patch_interpolate_antialias,
)


@pytest.fixture(autouse=True)
def isolate_interpolation_patch(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Exercise installation even if application setup already patched interpolation."""
    incoming = functional.interpolate

    def delegate(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return incoming(tensor, *args, **kwargs)

    # An unmarked delegate isolates patch installation without reloading PyTorch.
    # Restore the exact incoming callable, including any application-owned wrapper.
    with monkeypatch.context() as patcher:
        patcher.setattr(functional, "interpolate", delegate)
        yield
    assert functional.interpolate is incoming


@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_patch_keeps_unrelated_interpolation_modes(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the named compatibility behavior without device access."""
    original = functional.interpolate
    with monkeypatch.context() as patcher:
        patcher.setattr(functional, "interpolate", original)
        patch_interpolate_antialias()
        tensor = torch.arange(16.0).reshape(1, 1, 4, 4)
        result = functional.interpolate(tensor, size=(8, 8), mode=mode)
        expected = original(tensor, size=(8, 8), mode=mode)
        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("antialias", [True, False])
def test_patch_strips_antialias_keyword_but_preserves_other_parameters(
    antialias: bool,  # noqa: FBT001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the named compatibility behavior without device access."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        return tensor

    with monkeypatch.context() as patcher:
        patcher.setattr(functional, "interpolate", original)
        patch_interpolate_antialias()
        tensor = torch.ones((1, 1, 2, 2))
        actual = functional.interpolate(
            tensor, size=(4, 4), mode="bilinear", antialias=antialias
        )
        assert actual is tensor
        assert calls == [((), {"size": (4, 4), "mode": "bilinear"})]


def test_patch_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the named compatibility behavior without device access."""
    original = functional.interpolate
    with monkeypatch.context() as patcher:
        patcher.setattr(functional, "interpolate", original)
        patch_interpolate_antialias()
        first = functional.interpolate
        patch_interpolate_antialias()
        assert functional.interpolate is first
        assert getattr(first, "is_patched", False)


def test_patch_does_not_modify_original_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the named compatibility behavior without device access."""
    original = functional.interpolate
    with monkeypatch.context() as patcher:
        patcher.setattr(functional, "interpolate", original)
        patch_interpolate_antialias()
        assert functional.interpolate is not original
        assert getattr(original, "is_patched", False) is False


def test_patch_preserves_autograd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the named compatibility behavior without device access."""
    original = functional.interpolate
    with monkeypatch.context() as patcher:
        patcher.setattr(functional, "interpolate", original)
        patch_interpolate_antialias()
        tensor = torch.ones((1, 1, 4, 4), requires_grad=True)
        result = functional.interpolate(
            tensor, scale_factor=2, mode="bilinear", antialias=True
        )
        result.sum().backward()
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("use_scale", [False, True])
def test_downsampling_matches_reference_values_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    *,
    align_corners: bool,
    use_scale: bool,
) -> None:
    """The wrapper must match antialias-disabled CPU output and exact derivatives."""
    original = functional.interpolate
    values = (
        torch.arange(140, dtype=torch.float64)
        .reshape(2, 2, 5, 7)
        .sin()
        .transpose(-1, -2)
    )
    actual_input = values.clone().requires_grad_()
    expected_input = values.clone().requires_grad_()
    assert not actual_input.is_contiguous()
    options: dict[str, Any] = {"mode": mode, "align_corners": align_corners}
    if use_scale:
        options.update(scale_factor=0.6, recompute_scale_factor=True)
    else:
        options["size"] = (3, 4)
    expected = original(expected_input, **options, antialias=False)
    with monkeypatch.context() as patcher:
        patcher.setattr(functional, "interpolate", original)
        patch_interpolate_antialias()
        actual = functional.interpolate(actual_input, **options, antialias=True)
    weights = torch.linspace(-1, 1, actual.numel(), dtype=actual.dtype).reshape_as(
        actual
    )
    (actual_gradient,) = torch.autograd.grad(actual, actual_input, grad_outputs=weights)
    (expected_gradient,) = torch.autograd.grad(
        expected, expected_input, grad_outputs=weights
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.detach(), values, rtol=0, atol=0)
    assert actual.dtype == actual_input.dtype
    assert actual.device.type == "cpu"
