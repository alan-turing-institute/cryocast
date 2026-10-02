from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from icenet_mp.losses import LeadTimeWeightedLoss
from icenet_mp.losses.build_loss import build_loss


class TestBuildLoss:
    def test_no_exponent_returns_unwrapped_loss(self) -> None:
        loss_fn = build_loss(
            OmegaConf.create({"_target_": "torch.nn.HuberLoss", "delta": 0.5})
        )
        assert isinstance(loss_fn, torch.nn.HuberLoss)
        assert loss_fn.delta == pytest.approx(0.5)

    def test_none_exponent_returns_unwrapped_loss(self) -> None:
        # The key must also be stripped before instantiation, as HuberLoss would
        # reject an unexpected `lead_time_exponent` keyword argument
        loss_fn = build_loss(
            OmegaConf.create(
                {
                    "_target_": "torch.nn.HuberLoss",
                    "delta": 0.5,
                    "lead_time_exponent": None,
                }
            )
        )
        assert isinstance(loss_fn, torch.nn.HuberLoss)

    @pytest.mark.parametrize(
        "exponent", [-1.0, 0.0, 2.0], ids=["negative", "zero", "positive"]
    )
    def test_exponent_wraps_loss(self, exponent: float) -> None:
        loss_fn = build_loss(
            OmegaConf.create(
                {
                    "_target_": "torch.nn.HuberLoss",
                    "delta": 0.5,
                    "lead_time_exponent": exponent,
                }
            )
        )
        assert isinstance(loss_fn, LeadTimeWeightedLoss)
        assert loss_fn.exponent == pytest.approx(exponent)
        assert isinstance(loss_fn.wrapped_loss, torch.nn.HuberLoss)
        assert loss_fn.wrapped_loss.delta == pytest.approx(0.5)

    def test_accepts_plain_mapping(self) -> None:
        loss_fn = build_loss({"_target_": "torch.nn.MSELoss", "lead_time_exponent": 1})
        assert isinstance(loss_fn, LeadTimeWeightedLoss)
        assert isinstance(loss_fn.wrapped_loss, torch.nn.MSELoss)

    def test_rejects_non_module_target(self) -> None:
        with pytest.raises(TypeError, match=r"'builtins\.dict' created a dict"):
            build_loss(OmegaConf.create({"_target_": "builtins.dict"}))

    def test_rejects_double_wrap(self) -> None:
        cfg = OmegaConf.create(
            {
                "_target_": "icenet_mp.losses.lead_time_weighted_loss.LeadTimeWeightedLoss",
                "wrapped_loss": {"_target_": "torch.nn.MSELoss"},
                "lead_time_exponent": 2.0,
            }
        )
        with pytest.raises(TypeError, match="would wrap the loss twice"):
            build_loss(cfg)

    @pytest.mark.parametrize(
        ("extra_keys", "expected_type"),
        [
            ({}, torch.nn.HuberLoss),
            ({"lead_time_exponent": None}, torch.nn.HuberLoss),
            ({"lead_time_exponent": 2.0}, LeadTimeWeightedLoss),
        ],
        ids=["no-exponent", "none-exponent", "exponent"],
    )
    def test_struct_mode(
        self, extra_keys: dict[str, Any], expected_type: type[torch.nn.Module]
    ) -> None:
        cfg = OmegaConf.create(
            {"_target_": "torch.nn.HuberLoss", "delta": 0.5, **extra_keys}
        )
        OmegaConf.set_struct(cfg, value=True)
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, expected_type)
        # The input config is not modified, so checkpoints round-trip the exponent
        assert OmegaConf.to_container(cfg) == {
            "_target_": "torch.nn.HuberLoss",
            "delta": 0.5,
            **extra_keys,
        }
