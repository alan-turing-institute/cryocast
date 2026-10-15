"""Regression coverage for learnable latent affine scaling and offsets."""

import pytest
import torch
from torch import nn

from cryocast.models.common import Shift


@pytest.mark.parametrize(
    ("scale", "offset", "names"),
    [
        (False, False, []),
        (True, False, ["scale"]),
        (False, True, ["offset"]),
        (True, True, ["scale", "offset"]),
    ],
    ids=["identity", "scale-only", "offset-only", "both"],
)
def test_only_requested_trainable_parameters_exist(
    scale: bool,  # noqa: FBT001
    offset: bool,  # noqa: FBT001
    names: list[str],
) -> None:
    """Disabled transforms must not accidentally allocate trainable state."""
    layer = Shift(scale=scale, offset=offset)
    assert list(dict(layer.named_parameters())) == names
    assert all(parameter.requires_grad for parameter in layer.parameters())
    assert set(layer.state_dict()) == set(names)


@pytest.mark.parametrize(
    ("scale", "offset", "expected"),
    [
        (False, False, [0.0, 1.0, 3.0]),
        (True, False, [0.0, 2.0, 6.0]),
        (False, True, [-2.0, -1.0, 1.0]),
        (True, True, [-1.0, 1.0, 5.0]),
    ],
)
def test_affine_transform_matches_explicit_reference(
    scale: bool,  # noqa: FBT001
    offset: bool,  # noqa: FBT001
    expected: list[float],
) -> None:
    """Each optional parameter applies the exact affine transform."""
    layer = Shift(scale=scale, offset=offset)
    with torch.no_grad():
        if layer.scale is not None:
            layer.scale.fill_(2.0)
        if layer.offset is not None:
            layer.offset.fill_(-2.0 if not scale else -1.0)
    values = torch.tensor([0.0, 1.0, 3.0]).view(1, 1, 1, 3)
    actual = layer(values)
    torch.testing.assert_close(actual.flatten(), torch.tensor(expected))


def test_identity_returns_input_without_allocating_memory() -> None:
    """An unconfigured affine layer should be a true no-op."""
    values = torch.randn(2, 3, 4, 5)
    layer = Shift(scale=False, offset=False)
    assert layer(values) is values


@pytest.mark.parametrize("scale", [False, True], ids=["offset-only", "both"])
def test_offset_does_not_mutate_input_or_break_leaf_gradients(
    scale: bool,  # noqa: FBT001
) -> None:
    """The offset path must support a leaf tensor requiring gradients."""
    layer = Shift(scale=scale, offset=True)
    original = torch.tensor([1.0, -2.0, 3.0]).reshape(1, 1, 1, 3)
    values = original.detach().clone().requires_grad_()
    snapshot = values.detach().clone()
    result = layer(values)
    result.square().sum().backward()

    torch.testing.assert_close(values.detach(), snapshot)
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert layer.offset is not None
    assert layer.offset.grad is not None
    assert torch.isfinite(layer.offset.grad).all()


def test_affine_layer_has_correct_analytical_gradients() -> None:
    """Input, multiplicative and additive gradients must match calculus."""
    layer = Shift(scale=True, offset=True)
    assert layer.scale is not None
    assert layer.offset is not None
    with torch.no_grad():
        layer.scale.fill_(2.0)
        layer.offset.fill_(3.0)

    values = torch.tensor([1.0, 2.0, -1.0]).view(1, 1, 1, 3).requires_grad_()
    layer(values).sum().backward()

    assert values.grad is not None
    torch.testing.assert_close(values.grad, torch.full_like(values, 2.0))
    torch.testing.assert_close(layer.scale.grad, torch.tensor([2.0]))
    torch.testing.assert_close(layer.offset.grad, torch.tensor([3.0]))


@pytest.mark.parametrize(
    ("scale", "offset"), [(True, False), (False, True), (True, True)]
)
def test_affine_transformation_preserves_shape_and_float64_precision(
    scale: bool,  # noqa: FBT001
    offset: bool,  # noqa: FBT001
) -> None:
    """Parameter broadcasting must not change spatial shape or input dtype."""
    layer = Shift(scale=scale, offset=offset)
    values = torch.randn(2, 4, 5, 7, dtype=torch.float64)
    actual = layer(values)
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    assert torch.isfinite(actual).all()


def test_shift_is_a_regular_nn_module() -> None:
    """Trainable affine parameters remain visible to optimisers."""
    layer = Shift(scale=True, offset=True)
    assert isinstance(layer, nn.Module)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = {name: p.detach().clone() for name, p in layer.named_parameters()}
    layer(torch.ones(1, 1, 2, 2)).sum().backward()
    optimizer.step()
    assert all(
        not torch.equal(parameter, before[name])
        for name, parameter in layer.named_parameters()
    )
