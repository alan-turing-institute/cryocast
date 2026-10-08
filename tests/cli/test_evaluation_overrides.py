"""Evaluation-only model override visibility and sampler regression tests."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig

from cryocast.cli.evaluation_overrides import (
    _requested_model_paths,
    apply_evaluation_model_overrides,
)
from cryocast.cli.hydra import HydraInvocation
from cryocast.model_service import ModelService
from cryocast.models.processors import DiffusionProcessor
from cryocast.types import DataSpace


class FakeDiffusionProcessor:
    """Small stand-in for the real, separately tested diffusion sampler API."""

    def __init__(
        self, *, timesteps: int = 150, ddim_steps: int = 150, eta: float = 1.0
    ) -> None:
        """Track sampling changes without running a diffusion network."""
        self.timesteps = timesteps
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.calls: list[tuple[int | None, float]] = []

    def set_sampler(self, *, ddim_steps: int | None, eta: float) -> None:
        """Record and validate evaluation-only sampling configuration."""
        n_steps = self.timesteps if ddim_steps is None else ddim_steps
        if not 1 <= n_steps <= self.timesteps:
            msg = "invalid ddim_steps"
            raise ValueError(msg)
        if not 0 <= eta <= 1:
            msg = "invalid eta"
            raise ValueError(msg)
        self.ddim_steps = n_steps
        self.eta = eta
        self.calls.append((ddim_steps, eta))


def _service(
    *,
    processor: object | None = None,
    stored_name: str = "trained-model",
    stored_ddim_steps: int | None = None,
    stored_eta: float = 1.0,
) -> ModelService:
    """Build a checkpoint-loaded service without touching the filesystem."""
    service = ModelService.__new__(ModelService)
    service.config_ = DictConfig(
        {
            "model": {
                "name": stored_name,
                "processor": {
                    "_target_": "cryocast.models.processors.DiffusionProcessor",
                    "timesteps": 150,
                    "ddim_steps": stored_ddim_steps,
                    "eta": stored_eta,
                },
            }
        }
    )
    service.model_ = MagicMock()
    service.model_.processor = processor
    return service


def _request(
    *,
    model_name: str = "trained-model",
    ddim_steps: int | None = None,
    eta: float = 1.0,
) -> DictConfig:
    """Create an independently composed evaluation-side model config."""
    return DictConfig(
        {
            "model": {
                "name": model_name,
                "processor": {
                    "_target_": "cryocast.models.processors.DiffusionProcessor",
                    "timesteps": 150,
                    "ddim_steps": ddim_steps,
                    "eta": eta,
                },
            }
        }
    )


def test_model_path_parser_handles_hydra_add_delete_and_group_overrides() -> None:
    """Recover only explicitly changed model paths, not unrelated settings."""
    paths = _requested_model_paths(
        (
            "++model.processor.ddim_steps=50",
            "+model.processor.eta=0.0",
            "~model.decoder",
            "model=cnn_ddim_cnn",
            "evaluate.trainer.devices=1",
            "window.batch_size=4",
        )
    )
    assert paths == {
        "model.processor.ddim_steps": False,
        "model.processor.eta": False,
        "model.decoder": True,
        "model": False,
    }


def test_implicit_default_config_does_not_warn_about_checkpoint_differences(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The standard sample config is not an explicit model override."""
    service = _service()
    request = _request(model_name="sample-model")
    with caplog.at_level(logging.WARNING):
        apply_evaluation_model_overrides(
            service, request, HydraInvocation("sample", ())
        )

    assert service.config.model.name == "trained-model"
    assert not caplog.records


@pytest.mark.parametrize(
    ("path", "requested_name"),
    [
        ("model.name=other-model", "other-model"),
        ("model=cnn_ddim_cnn", "other-model"),
    ],
)
def test_explicit_incompatible_model_override_warns_without_swapping_checkpoint(
    path: str, requested_name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Never instantiate a different architecture from CLI model overrides."""
    service = _service()
    with caplog.at_level(logging.WARNING):
        apply_evaluation_model_overrides(
            service,
            _request(model_name=requested_name),
            HydraInvocation("sample", (path,)),
        )

    assert "Ignoring evaluation override" in caplog.text
    assert "checkpoint" in caplog.text
    assert service.config.model.name == "trained-model"


def test_custom_config_file_warns_when_its_model_differs_from_checkpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Detect alternative model groups introduced through --config-name."""
    service = _service()
    with caplog.at_level(logging.WARNING):
        apply_evaluation_model_overrides(
            service,
            _request(model_name="different-model"),
            HydraInvocation("experiment.local", ()),
        )

    assert "--config-name=experiment.local" in caplog.text
    assert "model.name" in caplog.text
    assert service.config.model.name == "trained-model"


def test_custom_config_with_same_model_has_no_false_positive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Do not warn when a custom config happens to match the checkpoint model."""
    service = _service()
    with caplog.at_level(logging.WARNING):
        apply_evaluation_model_overrides(
            service, _request(), HydraInvocation("experiment.local", ())
        )
    assert not caplog.records


@pytest.mark.parametrize(
    ("overrides", "candidate", "initial_ddim", "initial_eta", "expected"),
    [
        (
            ("++model.processor.ddim_steps=50", "++model.processor.eta=0.0"),
            (50, 0.0),
            150,
            1.0,
            (50, 0.0),
        ),
        (
            ("model.processor.eta=0.25",),
            (None, 0.25),
            50,
            0.0,
            (50, 0.25),
        ),
        (
            ("model.processor.ddim_steps=null",),
            (None, 1.0),
            50,
            1.0,
            (150, 1.0),
        ),
    ],
    ids=["both", "eta-only", "return-to-ddpm"],
)
def test_diffusion_sampler_overrides_apply_without_replacing_saved_model(
    monkeypatch: pytest.MonkeyPatch,
    overrides: tuple[str, ...],
    candidate: tuple[int | None, float],
    initial_ddim: int,
    initial_eta: float,
    expected: tuple[int, float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Switch sampling in-place, retaining any non-overridden sampler setting."""
    monkeypatch.setattr(
        "cryocast.cli.evaluation_overrides.DiffusionProcessor",
        FakeDiffusionProcessor,
    )
    processor = FakeDiffusionProcessor(ddim_steps=initial_ddim, eta=initial_eta)
    service = _service(processor=processor)
    request = _request(ddim_steps=candidate[0], eta=candidate[1])

    with caplog.at_level(logging.INFO):
        apply_evaluation_model_overrides(
            service, request, HydraInvocation("sample", overrides)
        )

    assert (processor.ddim_steps, processor.eta) == expected
    assert len(processor.calls) == 1
    assert "Applied evaluation-only diffusion sampler overrides" in caplog.text
    assert service.config.model.processor.timesteps == 150
    if "model.processor.ddim_steps" in " ".join(overrides):
        assert service.config.model.processor.ddim_steps == candidate[0]
    if "model.processor.eta" in " ".join(overrides):
        assert service.config.model.processor.eta == candidate[1]


def test_non_diffusion_sampler_override_is_explicitly_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Do not imply DDIM switches can change arbitrary processor classes."""
    service = _service(processor=SimpleNamespace())
    with caplog.at_level(logging.WARNING):
        apply_evaluation_model_overrides(
            service,
            _request(ddim_steps=50),
            HydraInvocation("sample", ("model.processor.ddim_steps=50",)),
        )

    assert "Ignoring evaluation override" in caplog.text
    assert "model.processor.ddim_steps" in caplog.text
    assert service.config.model.processor.ddim_steps is None


def test_model_architecture_overrides_do_not_get_applied_with_sampler(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Apply safe sampling but warn that changing diffusion steps is unsupported."""
    monkeypatch.setattr(
        "cryocast.cli.evaluation_overrides.DiffusionProcessor",
        FakeDiffusionProcessor,
    )
    processor = FakeDiffusionProcessor()
    service = _service(processor=processor)
    request = _request(ddim_steps=10)
    request.model.processor.timesteps = 10
    with caplog.at_level(logging.WARNING):
        apply_evaluation_model_overrides(
            service,
            request,
            HydraInvocation(
                "sample",
                ("model.processor.timesteps=10", "model.processor.ddim_steps=10"),
            ),
        )

    assert processor.ddim_steps == 10
    assert processor.timesteps == 150
    assert service.config.model.processor.timesteps == 150
    assert "Ignoring evaluation override 'model.processor.timesteps'" in caplog.text


def test_deleted_sampler_override_warns_and_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Deletion is not the same as explicitly requesting DDPM sampling."""
    monkeypatch.setattr(
        "cryocast.cli.evaluation_overrides.DiffusionProcessor",
        FakeDiffusionProcessor,
    )
    processor = FakeDiffusionProcessor()
    service = _service(processor=processor)
    request = _request()
    del request.model.processor.eta
    with caplog.at_level(logging.WARNING):
        apply_evaluation_model_overrides(
            service, request, HydraInvocation("sample", ("~model.processor.eta",))
        )

    assert processor.calls == []
    assert "Ignoring evaluation override 'model.processor.eta'" in caplog.text


def test_invalid_explicit_sampler_setting_fails_instead_of_silently_ignoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve sampler validation for nonsensical inference-time requests."""
    monkeypatch.setattr(
        "cryocast.cli.evaluation_overrides.DiffusionProcessor",
        FakeDiffusionProcessor,
    )
    service = _service(processor=FakeDiffusionProcessor())
    with pytest.raises(ValueError, match="invalid ddim_steps"):
        apply_evaluation_model_overrides(
            service,
            _request(ddim_steps=200),
            HydraInvocation("sample", ("model.processor.ddim_steps=200",)),
        )


def test_real_diffusion_sampler_switch_does_not_modify_training_weights() -> None:
    """Exercise the actual processor and prove sampler switches keep all weights."""
    combined = DataSpace(name="combined", channels=4, shape=(16, 16))
    target = DataSpace(name="target", channels=2, shape=(16, 16))
    processor = DiffusionProcessor(
        data_space=combined,
        data_space_target=target,
        target_channel_offset=1,
        n_forecast_steps=1,
        n_history_steps=1,
        timesteps=4,
        ddim_steps=None,
        eta=1.0,
        start_out_channels=8,
        time_embed_dim=256,
        dropout_rate=0.0,
        normalization="none",
        loss=torch.nn.MSELoss(),
    )
    service = _service(processor=processor, stored_ddim_steps=None)
    state_before = {
        name: value.clone() for name, value in processor.state_dict().items()
    }
    request = _request(ddim_steps=2, eta=0.0)

    apply_evaluation_model_overrides(
        service,
        request,
        HydraInvocation(
            "sample", ("model.processor.ddim_steps=2", "model.processor.eta=0.0")
        ),
    )

    assert processor.ddim_steps == 2
    assert processor.eta == 0.0
    assert not processor.uses_ddpm_sampler
    assert service.config.model.processor.ddim_steps == 2
    assert service.config.model.processor.eta == 0.0
    for name, value in state_before.items():
        torch.testing.assert_close(processor.state_dict()[name], value)
