"""Regression coverage for learnable latent affine scaling and offsets."""

from io import BytesIO

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


@pytest.mark.parametrize(
    ("scale", "offset"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_affine_nonleaf_view_preserves_sibling_branch_and_exact_gradients(
    *, scale: bool, offset: bool
) -> None:
    """A shared noncontiguous input must retain its values and both gradient paths."""
    layer = Shift(scale=scale, offset=offset).double()
    a, b = (1.75 if scale else 1.0), (-0.5 if offset else 0.0)
    with torch.no_grad():
        if layer.scale is not None:
            layer.scale.fill_(a)
        if layer.offset is not None:
            layer.offset.fill_(b)
    leaf = (
        torch.linspace(-1.0, 2.0, 24, dtype=torch.float64)
        .reshape(2, 1, 3, 4)
        .requires_grad_()
    )
    values = (leaf * 1.5).transpose(-1, -2)
    assert not values.is_contiguous()
    before = values.detach().clone()
    version = values._version
    sibling = values.square().sum()
    result = layer(values)
    expected = a * before + b
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    assert values._version == version
    (result.square().sum() + sibling).backward()
    wanted = (1.5 * (2 * a * expected + 2 * before)).transpose(-1, -2)
    torch.testing.assert_close(leaf.grad, wanted, rtol=1e-12, atol=1e-12)
    if layer.scale is not None:
        torch.testing.assert_close(
            layer.scale.grad, (2 * expected * before).sum().reshape(1)
        )
    if layer.offset is not None:
        torch.testing.assert_close(layer.offset.grad, (2 * expected).sum().reshape(1))


@pytest.mark.parametrize(
    ("scale", "offset"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_affine_input_and_parameter_gradients_match_finite_differences(
    *, scale: bool, offset: bool
) -> None:
    """Check the input and every enabled affine parameter in float64."""
    layer = Shift(scale=scale, offset=offset).double()
    with torch.no_grad():
        if layer.scale is not None:
            layer.scale.fill_(1.3)
        if layer.offset is not None:
            layer.offset.fill_(-0.7)
    names, parameters = tuple(dict(layer.named_parameters())), tuple(layer.parameters())
    values = (
        torch.linspace(-0.6, 0.8, 6, dtype=torch.float64)
        .reshape(1, 1, 2, 3)
        .requires_grad_()
    )

    def evaluate(x: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(
            layer, dict(zip(names, weights, strict=True)), (x,), strict=True
        )

    assert torch.autograd.gradcheck(evaluate, (values, *parameters))


@pytest.mark.parametrize(
    ("scale", "offset"), [(True, False), (False, True), (True, True)]
)
def test_affine_checkpoint_restores_predictions_and_next_adam_update(
    *, scale: bool, offset: bool
) -> None:
    """The out-of-place fix keeps checkpoint keys and optimizer resumption intact."""
    layer = Shift(scale=scale, offset=offset).double()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.02)
    values = torch.linspace(-1, 2, 12, dtype=torch.float64).reshape(1, 1, 3, 4)

    def update(module: Shift, optim: torch.optim.Optimizer) -> None:
        optim.zero_grad(set_to_none=True)
        (module(values) - 0.4).square().mean().backward()
        optim.step()

    update(layer, optimizer)
    buffer = BytesIO()
    torch.save(
        {"model": layer.state_dict(), "optimizer": optimizer.state_dict()}, buffer
    )
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored = Shift(scale=scale, offset=offset).double()
    restored.load_state_dict(state["model"], strict=True)
    resumed = torch.optim.AdamW(restored.parameters(), lr=0.02)
    resumed.load_state_dict(state["optimizer"])
    torch.testing.assert_close(restored(values), layer(values), rtol=0, atol=0)
    update(layer, optimizer)
    update(restored, resumed)
    for name, parameter in layer.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], parameter, rtol=0, atol=0
        )
    for first, second in zip(
        optimizer.state.values(), resumed.state.values(), strict=True
    ):
        assert first.keys() == second.keys()
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
