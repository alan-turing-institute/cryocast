"""Real-network and checkpoint checks for the optional MoE model."""

from importlib.resources import files
from pathlib import Path
from typing import Any

import lightning
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from cryocast.models import EncodeProcessDecode
from cryocast.models.processors import MixtureOfExpertsProcessor


def _components() -> dict[str, DictConfig]:
    encoder = {"_target_": "cryocast.models.encoders.NaiveLinearEncoder"}
    return {
        "encoders": DictConfig(
            {"latent_space": [32, 32], "sic": encoder, "weather": encoder}
        ),
        "decoder": DictConfig(
            {"_target_": "cryocast.models.decoders.NaiveLinearDecoder"}
        ),
        "processor": DictConfig(
            {
                "_target_": "cryocast.models.processors.MixtureOfExpertsProcessor",
                "_recursive_": False,
                "gate_hidden_channels": 4,
                "experts": [
                    {
                        "_target_": "cryocast.models.processors.UNetProcessor",
                        "start_out_channels": 4,
                        "norm_type": "groupnorm",
                    },
                    {
                        "_target_": "cryocast.models.processors.VitProcessor",
                        "emb_dim": 8,
                        "heads": 2,
                        "mlp_dim": 16,
                        "patch_size": 4,
                        "depth": 1,
                        "dropout": 0.0,
                    },
                ],
            }
        ),
    }


def _model() -> EncodeProcessDecode:
    components = _components()
    return EncodeProcessDecode(
        name="moe-integration",
        encoders=components["encoders"],
        processor=components["processor"],
        decoder=components["decoder"],
        input_spaces=[
            {"name": "weather", "channels": 1, "shape": [32, 32]},
            {"name": "sic", "channels": 2, "shape": [32, 32]},
        ],
        output_space={"name": "sic", "channels": 1, "shape": [32, 32]},
        n_history_steps=2,
        n_forecast_steps=3,
        target_variable_indices=[1],
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        metrics=[],
        optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-3}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
        hemisphere="north",
    )


@pytest.mark.parametrize("seed", [0, 490])
def test_real_experts_train_and_full_model_restores(tmp_path: Path, seed: int) -> None:
    """Learn through both real experts and the gate, then safely restore the model."""
    torch.manual_seed(seed)
    model = _model().train()
    inputs = {
        "weather": torch.randn(2, 2, 1, 32, 32),
        "sic": torch.rand(2, 2, 2, 32, 32),
    }
    target = torch.rand(2, 3, 1, 32, 32)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    prediction = model(inputs)
    assert prediction.shape == target.shape
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert isinstance(model.processor, MixtureOfExpertsProcessor)
    for component in [
        model.processor.gate,
        *model.processor.experts,
        *model.encoders,
        model.decoder,
    ]:
        gradients = [p.grad for p in component.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(inputs)
    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "moe.ckpt"
    torch.save(checkpoint, path)
    assert torch.serialization.get_unsafe_globals_in_checkpoint(path) == []
    components = _components()
    restored = EncodeProcessDecode.load_from_checkpoint(
        path,
        encoders=components["encoders"],
        processor=components["processor"],
        decoder=components["decoder"],
        metrics=[],
        weights_only=True,
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.state_dict(), model.state_dict(), rtol=0, atol=0
    )


def test_optional_configuration_matches_encoder_decoder_controls() -> None:
    """Changing the processor must not also change encoder/decoder baselines."""
    with initialize_config_dir(
        config_dir=str(files("cryocast.config")), version_base=None
    ):
        mixture = compose(config_name="sample", overrides=["model=cnn_moe_cnn"])
        control = compose(config_name="sample", overrides=["model=cnn_unet_cnn"])
    for key in ("encoders", "decoder"):
        assert OmegaConf.to_container(mixture.model[key]) == OmegaConf.to_container(
            control.model[key]
        )
    for key in ("data", "variables", "window", "loss", "train", "evaluate"):
        assert OmegaConf.to_container(mixture[key]) == OmegaConf.to_container(
            control[key]
        )
    assert mixture.model.processor._recursive_ is False
    assert len(mixture.model.processor.experts) == 2


def test_whole_trajectory_mixture_does_not_claim_physical_step_support() -> None:
    """A whole-forecast mixture does not imply a separate physical-step policy."""
    model = _model()
    model.rollout_space = type(model.rollout_space)("physical")
    inputs = {
        "weather": torch.zeros(2, 2, 1, 32, 32),
        "sic": torch.zeros(2, 2, 2, 32, 32),
    }
    with pytest.raises(ValueError, match="does not implement an autoregressive"):
        model(inputs)
