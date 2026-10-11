"""CPU-only regression tests for the interpolation compatibility patch."""

import pytest
import torch
from torch.nn import functional

from cryocast.compatibility.torch.patch_interpolate_antialias import (
    patch_interpolate_antialias,
)


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
