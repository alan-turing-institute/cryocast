from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from cryocast.models.common import ExponentialRelaxation
from cryocast.models.finalisers import Finaliser
from cryocast.types import DataSpace, SkipAnchorType


class _FinaliserBase:
    """Shared fixtures for Finaliser tests.

    The tests give the finaliser a known raw decoder output, so that the expected forecast
    state can be calculated by hand.
    """

    _OUTPUT_SPACE = DataSpace(name="output", channels=1, shape=(8, 8))
    _N_FORECAST_STEPS = 4

    @classmethod
    def _finaliser(
        cls,
        *,
        method: str = "none",
        restrict_range: str = "sigmoid",
        restrict_range_min: float | None = None,
        restrict_range_max: float | None = None,
        mask_dir: Path | None = None,
        mask_type: str | None = None,
    ) -> Finaliser:
        return Finaliser(
            data_space=cls._OUTPUT_SPACE,
            mask_dir=mask_dir,
            mask_type=mask_type,
            restrict_range=restrict_range,
            restrict_range_min=restrict_range_min,
            restrict_range_max=restrict_range_max,
            skip_connection={"method": method},
        )

    @classmethod
    def _raw(cls, value: float, n_forecast_steps: int = 1) -> torch.Tensor:
        """A spatially uniform raw decoder output with shape [1, T, C, H, W]."""
        return torch.full((1, n_forecast_steps, *cls._OUTPUT_SPACE.chw), value)

    @classmethod
    def _persistence(cls) -> torch.Tensor:
        """A spatially varying persistence field in (0, 1), never exactly 0 or 1."""
        space = cls._OUTPUT_SPACE
        numel = space.channels * space.shape[0] * space.shape[1]
        return torch.linspace(0.05, 0.95, numel).reshape(
            1, 1, space.channels, *space.shape
        )

    @classmethod
    def _steps(cls) -> torch.Tensor:
        """The step number k = 1...N, shaped to broadcast over [B, T, C, H, W]."""
        return torch.arange(1, cls._N_FORECAST_STEPS + 1).view(1, -1, 1, 1, 1)

    @classmethod
    def _top_half_mask(cls, tmp_path: Path) -> Path:
        """Save an active mask that masks the top half (0) and keeps the bottom (1)."""
        mask = np.ones(cls._OUTPUT_SPACE.shape, dtype=np.uint8)
        mask[: cls._OUTPUT_SPACE.shape[0] // 2, :] = 0
        np.save(tmp_path / "active_mask.npy", mask)
        return tmp_path

    @staticmethod
    def _zero(conv_module: nn.Module) -> None:
        conv = cast("nn.Conv2d", conv_module)
        assert conv.bias is not None
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.zero_()


class TestFinaliserBounded(_FinaliserBase):
    @pytest.mark.parametrize("restrict_range", ["sigmoid", "tanh", "clamp"])
    @pytest.mark.parametrize("value", [1e10, -1e10], ids=["large", "small"])
    def test_bounds_values_between_0_and_1(
        self, restrict_range: str, value: float
    ) -> None:
        finaliser = self._finaliser(restrict_range=restrict_range)

        out = finaliser.rollout(self._raw(value))

        assert torch.all(out >= 0.0).item()
        assert torch.all(out <= 1.0).item()

    def test_is_identity_when_mask_skip_and_bound_off(self) -> None:
        finaliser = self._finaliser(restrict_range="none")
        x = torch.randn(2, *self._OUTPUT_SPACE.chw)

        assert torch.equal(finaliser(x), x)


class TestFinaliserMask(_FinaliserBase):
    @pytest.mark.parametrize(
        ("mask_type", "mask_filename"),
        [("active", "active_mask.npy"), ("land", "land_mask.npy")],
        ids=["active", "land"],
    )
    @pytest.mark.parametrize(
        "restrict_range", ["none", "sigmoid"], ids=["unbounded", "sigmoid"]
    )
    def test_mask_loads_and_zeros_masked_cells(
        self, tmp_path: Path, mask_type: str, mask_filename: str, restrict_range: str
    ) -> None:
        """Masking is applied AFTER bounding, so masked cells are 0, not sigmoid(0)=0.5."""
        mask = np.ones(self._OUTPUT_SPACE.shape, dtype=np.uint8)
        mask[:4, :] = 0
        np.save(tmp_path / mask_filename, mask)
        finaliser = self._finaliser(
            restrict_range=restrict_range, mask_dir=tmp_path, mask_type=mask_type
        )
        unmasked_finaliser = self._finaliser(restrict_range=restrict_range)
        raw = torch.randn(2, 1, *self._OUTPUT_SPACE.chw)

        out = finaliser.rollout(raw)
        reference = unmasked_finaliser.rollout(raw)

        assert finaliser.mask.mask.shape == self._OUTPUT_SPACE.shape
        # Every masked cell must be exactly zero, with or without bounding...
        assert torch.all(out[..., :4, :] == 0).item()
        # ...and unmasked cells are untouched.
        active = out[..., 4:, :]
        assert torch.equal(active, reference[..., 4:, :])
        if restrict_range == "sigmoid":
            assert torch.all((active > 0) & (active < 1)).item()

    def test_use_mask_without_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="mask is requested"):
            self._finaliser(mask_dir=tmp_path / "does_not_exist", mask_type="active")

    def test_mask_type_none_creates_no_buffer(self) -> None:
        finaliser = self._finaliser(mask_type=None)
        assert not hasattr(finaliser.mask, "mask")

    def test_unknown_mask_type_raises(self) -> None:
        """A typo in mask_type fails loudly rather than silently disabling masking."""
        with pytest.raises(ValueError, match="not a valid MaskType"):
            self._finaliser(mask_type="activ")


class TestFinaliserSkipConnection(_FinaliserBase):
    def test_additive_skip_recovers_persistence(self) -> None:
        finaliser = self._finaliser(method="additive")
        persistence = self._persistence()

        out = finaliser.rollout(self._raw(0.0, n_forecast_steps=2), anchor=persistence)

        assert torch.allclose(out, persistence.expand(-1, 2, -1, -1, -1), atol=1e-4)

    def test_additive_skip_can_decrease_below_persistence(self) -> None:
        # A negative residual must be reachable, or the decoder could only ever push
        # the prediction above persistence, never below it.
        finaliser = self._finaliser(method="additive")
        persistence = self._persistence()

        out = finaliser.rollout(self._raw(-0.5), anchor=persistence)

        assert torch.all(out < persistence)

    def test_gated_skip_recovers_half_persistence(self) -> None:
        finaliser = self._finaliser(method="gated")
        # Zero the gate's conv so its pre-activation is 0 everywhere
        # gate = sigmoid(0) = 0.5, so output will be:
        # 0.5 * bounded_main + 0.5 * persistence ~= 0.5 * persistence.
        assert finaliser.skip_connection is not None
        self._zero(finaliser.skip_connection.gate.block[0])
        persistence = self._persistence()

        out = finaliser.rollout(self._raw(-20.0), anchor=persistence)

        assert torch.allclose(out, 0.5 * persistence, atol=1e-4)

    def test_convolutional_skip_is_clamped_to_valid_range(self) -> None:
        finaliser = self._finaliser(method="convolutional")
        # Force the fusion's final conv to output a large constant, well outside [0, 1].
        assert finaliser.skip_connection is not None
        final_conv = cast("nn.Conv2d", finaliser.skip_connection.fusion[-1])
        assert final_conv.bias is not None
        with torch.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.fill_(10.0)

        out = finaliser.rollout(self._raw(-20.0), anchor=self._persistence())

        assert torch.allclose(out, torch.ones_like(out))

    @pytest.mark.parametrize("anchor_type", list(SkipAnchorType))
    def test_skip_connection_without_anchor_raises(
        self, anchor_type: SkipAnchorType
    ) -> None:
        finaliser = self._finaliser(method="additive")
        with pytest.raises(ValueError, match="no anchor was provided"):
            finaliser.rollout(self._raw(0.0), anchor_type=anchor_type)


class TestFinaliserSkipAnchor(_FinaliserBase):
    """The anchor_type argument to Finaliser.rollout()."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="default"),
            pytest.param({"anchor_type": SkipAnchorType.INITIAL}, id="explicit"),
        ],
    )
    def test_initial_reuses_the_same_anchor(self, kwargs: dict[str, Any]) -> None:
        finaliser = self._finaliser(method="additive", restrict_range="none")
        persistence = self._persistence()

        out = finaliser.rollout(
            self._raw(0.1, self._N_FORECAST_STEPS), anchor=persistence, **kwargs
        )

        expected = (persistence + 0.1).clamp(0, 1).expand_as(out)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_previous_accumulates_a_constant_tendency(self) -> None:
        finaliser = self._finaliser(method="additive", restrict_range="none")
        persistence = self._persistence()

        out = finaliser.rollout(
            self._raw(0.1, self._N_FORECAST_STEPS),
            anchor=persistence,
            anchor_type=SkipAnchorType.PREVIOUS,
        )

        # Each step adds 0.1 to the clamped output of the step before it. Values only
        # increase, so clamping at an intermediate step gives the same result as
        # clamping only at the end.
        expected = (persistence + 0.1 * self._steps()).clamp(0, 1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_previous_anchor_is_the_clamped_output(self) -> None:
        finaliser = self._finaliser(method="additive", restrict_range="none")
        raw = torch.cat([self._raw(2.0), self._raw(-0.25)], dim=1)

        out = finaliser.rollout(
            raw, anchor=self._persistence(), anchor_type=SkipAnchorType.PREVIOUS
        )

        # The first step pushes every cell above 1, so it is clamped to 1. The second
        # step subtracts from the clamped value (1), not from the unclamped value.
        assert torch.allclose(out[:, 0], torch.ones_like(out[:, 0]))
        assert torch.allclose(out[:, 1], torch.full_like(out[:, 1], 0.75))

    def test_gated_previous_compounds_the_gate(self) -> None:
        # With a zeroed gate (0.5) and main output ~0, each step halves its anchor.
        finaliser = self._finaliser(method="gated")
        assert finaliser.skip_connection is not None
        self._zero(finaliser.skip_connection.gate.block[0])
        persistence = self._persistence()

        out = finaliser.rollout(
            self._raw(-20.0, self._N_FORECAST_STEPS),
            anchor=persistence,
            anchor_type=SkipAnchorType.PREVIOUS,
        )

        expected = persistence * 0.5 ** self._steps()
        assert torch.allclose(out, expected, atol=1e-4)

    @pytest.mark.parametrize("method", ["additive", "gated", "convolutional"])
    def test_previous_matches_a_manual_loop(self, method: str) -> None:
        finaliser = self._finaliser(method=method)
        raw = torch.randn(1, self._N_FORECAST_STEPS, *self._OUTPUT_SPACE.chw)
        persistence = self._persistence()

        out = finaliser.rollout(
            raw, anchor=persistence, anchor_type=SkipAnchorType.PREVIOUS
        )

        anchor = persistence[:, 0]
        for t in range(self._N_FORECAST_STEPS):
            anchor = finaliser(raw[:, t], anchor=anchor)
            assert torch.allclose(out[:, t], anchor, atol=1e-6)

    def test_previous_without_skip_connection_matches_initial(self) -> None:
        finaliser = self._finaliser(method="none")
        raw = torch.randn(1, self._N_FORECAST_STEPS, *self._OUTPUT_SPACE.chw)

        initial = finaliser.rollout(raw)
        previous = finaliser.rollout(raw, anchor_type=SkipAnchorType.PREVIOUS)

        assert torch.equal(initial, previous)

    def test_previous_keeps_masked_cells_at_zero(self, tmp_path: Path) -> None:
        finaliser = self._finaliser(
            method="additive",
            restrict_range="none",
            mask_dir=self._top_half_mask(tmp_path),
            mask_type="active",
        )
        persistence = self._persistence()

        out = finaliser.rollout(
            self._raw(0.1, self._N_FORECAST_STEPS),
            anchor=persistence,
            anchor_type=SkipAnchorType.PREVIOUS,
        )

        # Masked cells are 0 at every step, although each anchor adds a tendency...
        assert torch.all(out[..., :4, :] == 0).item()
        # ...and active cells still accumulate the tendency.
        expected = (persistence + 0.1 * self._steps()).clamp(0, 1)
        assert torch.allclose(out[..., 4:, :], expected[..., 4:, :], atol=1e-6)

    def test_previous_gradients_flow_through_every_step(self) -> None:
        finaliser = self._finaliser(method="additive", restrict_range="none")
        persistence = self._persistence().requires_grad_()

        out = finaliser.rollout(
            self._raw(0.0, self._N_FORECAST_STEPS),
            anchor=persistence,
            anchor_type=SkipAnchorType.PREVIOUS,
        )
        out[:, -1].sum().backward()

        # The last step depends on the initial anchor through the chain of steps.
        assert persistence.grad is not None
        assert torch.all(persistence.grad != 0).item()


class TestFinaliserRelaxation(_FinaliserBase):
    """The relaxation of a Finaliser and the equilibrium argument to rollout().

    The relaxed PREVIOUS-anchor rollout and masking are tested through the model in
    test_physical_rollout.py (TestClimatologyRelaxation).
    """

    _RELAXATION = ExponentialRelaxation(tau=2.0)

    def test_initial_relaxes_every_step_from_the_same_anchor(self) -> None:
        finaliser = self._finaliser(method="additive", restrict_range="none")
        finaliser.relaxation = self._RELAXATION
        persistence = self._persistence()
        equilibrium = 0.3

        out = finaliser.rollout(
            self._raw(0.0, self._N_FORECAST_STEPS),
            anchor=persistence,
            anchor_type=SkipAnchorType.INITIAL,
            equilibrium=self._raw(equilibrium, self._N_FORECAST_STEPS),
        )

        # Each step is one relaxation step away from the observation
        expected = persistence + self._RELAXATION.rate * (equilibrium - persistence)
        assert torch.allclose(out, expected.expand_as(out), atol=1e-6)

    def test_without_equilibrium_does_not_relax(self) -> None:
        finaliser = self._finaliser(method="additive", restrict_range="none")
        raw = self._raw(0.1, self._N_FORECAST_STEPS)
        persistence = self._persistence()

        plain = finaliser.rollout(
            raw, anchor=persistence, anchor_type=SkipAnchorType.PREVIOUS
        )
        finaliser.relaxation = self._RELAXATION
        relaxed = finaliser.rollout(
            raw, anchor=persistence, anchor_type=SkipAnchorType.PREVIOUS
        )

        assert torch.equal(relaxed, plain)


class TestFinaliserCustomOutputRange(_FinaliserBase):
    """Check that restrict_range_min/max are configured correctly."""

    def test_defaults_to_unit_range(self) -> None:
        finaliser = self._finaliser()
        assert finaliser.range_min == 0
        assert finaliser.range_max == 1

    def test_custom_range_is_respected_without_skip_connection(self) -> None:
        finaliser = self._finaliser(
            restrict_range="tanh", restrict_range_min=-5.0, restrict_range_max=5.0
        )
        assert finaliser.range_min == -5.0
        assert finaliser.range_max == 5.0

        out = finaliser.rollout(self._raw(100.0))

        # tanh saturates towards +1, rescaled into [-5, 5] -> +5 everywhere.
        assert torch.allclose(out, 5.0 * torch.ones_like(out), atol=1e-3)

    def test_final_clamp_uses_custom_range_with_skip_connection(self) -> None:
        finaliser = self._finaliser(
            method="convolutional",
            restrict_range="tanh",
            restrict_range_min=-5.0,
            restrict_range_max=5.0,
        )
        assert finaliser.skip_connection is not None
        final_conv = cast("nn.Conv2d", finaliser.skip_connection.fusion[-1])
        assert final_conv.bias is not None
        with torch.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.fill_(100.0)  # Far outside [-5, 5]

        persistence = torch.zeros(1, 1, *self._OUTPUT_SPACE.chw)
        out = finaliser.rollout(self._raw(-20.0), anchor=persistence)

        assert torch.allclose(out, 5.0 * torch.ones_like(out))

    def test_additive_residual_bound_covers_asymmetric_configured_range(self) -> None:
        # An asymmetric configured range [0, 2] should still give a symmetric [-2, 2]
        # residual bound (max of the absolute endpoints), not [0, 2].
        finaliser = self._finaliser(
            method="additive",
            restrict_range="tanh",
            restrict_range_min=0.0,
            restrict_range_max=2.0,
        )
        persistence = torch.full((1, 1, *self._OUTPUT_SPACE.chw), 1.0)

        out = finaliser.rollout(self._raw(-100.0), anchor=persistence)

        # tanh(-100) ~= -1, rescaled by the residual bound of 2 -> residual ~= -2.
        # persistence (1.0) + residual (-2) clamped to the configured [0, 2] -> 0.
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-3)
