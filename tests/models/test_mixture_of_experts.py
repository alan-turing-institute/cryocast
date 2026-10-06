from importlib.resources import files

import pytest
import torch
import yaml
from omegaconf import DictConfig

from cryocast.models.processors import MixtureOfExpertsProcessor
from cryocast.types import DataSpace


def _processor(experts: list[dict[str, object]]) -> MixtureOfExpertsProcessor:
    """Build a small MoE processor for unit tests."""
    return MixtureOfExpertsProcessor(
        experts=[DictConfig(config) for config in experts],
        data_space=DataSpace(name="latent", channels=2, shape=(8, 8)),
        n_forecast_steps=2,
        n_history_steps=2,
        target_channel_offset=0,
        gate_hidden_channels=4,
    )


def test_moe_expert_weights_are_normalised_per_sample() -> None:
    """Return one softmax distribution over experts for each sample."""
    processor = _processor(
        [
            {"_target_": "cryocast.models.processors.NullProcessor"},
            {"_target_": "cryocast.models.processors.NullProcessor"},
        ]
    )
    x = torch.randn(3, 2, 2, 8, 8)

    weights = processor.expert_weights(x)

    assert weights.shape == (3, 2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    assert torch.all(weights >= 0)


def test_moe_rollout_preserves_forecast_shape() -> None:
    """Blend expert forecasts without changing the processor tensor contract."""
    processor = _processor(
        [
            {"_target_": "cryocast.models.processors.NullProcessor"},
            {"_target_": "cryocast.models.processors.NullProcessor"},
        ]
    )
    x = torch.randn(3, 2, 2, 8, 8)

    output = processor.rollout(x)

    assert output.prediction.shape == (3, 2, 2, 8, 8)


def test_moe_backpropagates_to_gate_and_trainable_expert() -> None:
    """Keep routing and expert parameters trainable end to end."""
    processor = _processor(
        [
            {"_target_": "cryocast.models.processors.NullProcessor"},
            {
                "_target_": "cryocast.models.processors.VitProcessor",
                "depth": 1,
                "dropout": 0.0,
                "emb_dim": 8,
                "heads": 2,
                "mlp_dim": 16,
                "patch_size": 2,
            },
        ]
    )
    x = torch.randn(2, 2, 2, 8, 8, requires_grad=True)

    prediction = processor.rollout(x).prediction
    prediction.square().mean().backward()

    gate_gradients = [
        parameter.grad
        for parameter in processor.gate.parameters()
        if parameter.grad is not None
    ]
    expert_gradients = [
        parameter.grad
        for parameter in processor.experts[1].parameters()
        if parameter.grad is not None
    ]
    assert gate_gradients
    assert expert_gradients
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gate_gradients)
    assert any(
        torch.count_nonzero(gradient).item() > 0 for gradient in expert_gradients
    )


def test_moe_requires_at_least_one_expert() -> None:
    """Reject an empty mixture at construction time."""
    with pytest.raises(ValueError, match="at least one expert"):
        _processor([])


def test_moe_model_config_keeps_expert_configs_non_recursive() -> None:
    """Keep nested expert configs intact until shared processor arguments are known."""
    config_path = files("cryocast.config") / "model" / "cnn_moe_cnn.yaml"
    config = yaml.safe_load(config_path.read_text())

    processor = config["processor"]
    assert processor["_recursive_"] is False
    assert len(processor["experts"]) == 2
