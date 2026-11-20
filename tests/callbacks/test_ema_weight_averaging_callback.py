from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import AveragedModel

from cryocast.callbacks.ema_weight_averaging_callback import EMAWeightAveragingCallback
from cryocast.models.diffusion import LatentNormaliser


class ParameterlessLightningModule(LightningModule):
    """A LightningModule with no parameters, to exercise the empty-module guard."""


class NormalisedLightningModule(LightningModule):
    """A LightningModule with one trainable layer and a LatentNormaliser."""

    def __init__(self) -> None:
        """Initialise the layer and the normaliser."""
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.normaliser = LatentNormaliser(2)


def _averaged(callback: EMAWeightAveragingCallback) -> NormalisedLightningModule:
    """Return the averaged copy of the module held by the callback."""
    assert callback._average_model is not None
    return cast("NormalisedLightningModule", callback._average_model.module)


class TestInit:
    """Tests for EMAWeightAveragingCallback construction."""

    def test_stores_schedule_parameters(self) -> None:
        """Store the configured update schedule on the instance."""
        callback = EMAWeightAveragingCallback(
            decay_rate=0.99, every_n_epochs=2, every_n_steps=3
        )

        assert callback.every_n_epochs == 2
        assert callback.every_n_steps == 3


class TestOnTrainBatchEnd:
    """Tests for on_train_batch_end."""

    def test_skips_parameterless_module(
        self, mock_trainer: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not delegate to the parent hook when the module has no parameters."""
        callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_steps=1)
        parent_hook = MagicMock()
        monkeypatch.setattr(WeightAveraging, "on_train_batch_end", parent_hook)
        module = ParameterlessLightningModule()

        callback.on_train_batch_end(mock_trainer, module)

        parent_hook.assert_not_called()

    def test_delegates_for_parameterised_module(
        self,
        mock_trainer: MagicMock,
        linear_lightning_module: LightningModule,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Delegate to the parent hook when the module has parameters."""
        callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_steps=1)
        parent_hook = MagicMock()
        monkeypatch.setattr(WeightAveraging, "on_train_batch_end", parent_hook)
        module = linear_lightning_module

        callback.on_train_batch_end(mock_trainer, module)

        parent_hook.assert_called_once()

    def test_forwards_args_and_kwargs_to_parent_hook(
        self,
        mock_trainer: MagicMock,
        linear_lightning_module: LightningModule,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Forward positional and keyword arguments to the parent hook unchanged."""
        callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_steps=1)
        parent_hook = MagicMock()
        monkeypatch.setattr(WeightAveraging, "on_train_batch_end", parent_hook)
        module = linear_lightning_module
        batch = {"sic": torch.zeros(1)}

        callback.on_train_batch_end(mock_trainer, module, batch, 5, unused=True)

        parent_hook.assert_called_once_with(mock_trainer, module, batch, 5, unused=True)


class TestOnTrainEpochEnd:
    """Tests for on_train_epoch_end."""

    def test_skips_parameterless_module(
        self, mock_trainer: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not delegate to the parent hook when the module has no parameters."""
        callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_epochs=1)
        parent_hook = MagicMock()
        monkeypatch.setattr(WeightAveraging, "on_train_epoch_end", parent_hook)
        module = ParameterlessLightningModule()

        callback.on_train_epoch_end(mock_trainer, module)

        parent_hook.assert_not_called()

    def test_delegates_for_parameterised_module(
        self,
        mock_trainer: MagicMock,
        linear_lightning_module: LightningModule,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Delegate to the parent hook when the module has parameters."""
        callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_epochs=1)
        parent_hook = MagicMock()
        monkeypatch.setattr(WeightAveraging, "on_train_epoch_end", parent_hook)
        module = linear_lightning_module

        callback.on_train_epoch_end(mock_trainer, module)

        parent_hook.assert_called_once_with(mock_trainer, module)


class TestShouldUpdate:
    """Tests for should_update."""

    @pytest.mark.parametrize(
        ("step_idx", "epoch_idx", "expected"),
        [
            (2, None, False),
            (3, None, True),
            (6, None, True),
            (None, 1, False),
            (None, 2, True),
            (None, 4, True),
            (None, None, False),
            (0, None, False),
            (None, 0, False),
            (3, 1, False),
            (4, 2, True),
        ],
        ids=[
            "step-not-multiple",
            "step-multiple",
            "step-multiple-x2",
            "epoch-not-multiple",
            "epoch-multiple",
            "epoch-multiple-x2",
            "neither-given",
            "step-zero",
            "epoch-zero",
            "both-given-epoch-not-multiple-wins",
            "both-given-epoch-multiple-wins",
        ],
    )
    def test_updates_on_configured_interval(
        self, step_idx: int | None, epoch_idx: int | None, *, expected: bool
    ) -> None:
        """Update on the configured step/epoch interval, epoch taking precedence."""
        callback = EMAWeightAveragingCallback(
            decay_rate=0.99,
            every_n_steps=3,
            every_n_epochs=2,
        )

        assert (
            callback.should_update(step_idx=step_idx, epoch_idx=epoch_idx) is expected
        )


class TestExcludedBuffers:
    """Tests for copying buffers that should not be averaged."""

    @pytest.mark.parametrize("hook", ["on_train_batch_end", "on_train_epoch_end"])
    def test_copies_normaliser_buffers_instead_of_averaging(
        self, mock_trainer: MagicMock, monkeypatch: pytest.MonkeyPatch, hook: str
    ) -> None:
        """Keep the averaged normaliser statistics the same as in training."""
        callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_steps=1)
        monkeypatch.setattr(WeightAveraging, hook, MagicMock())
        module = NormalisedLightningModule()
        callback._average_model = AveragedModel(module, use_buffers=True)
        module.normaliser.running_mean.fill_(3.0)
        module.normaliser.running_var.fill_(4.0)
        module.normaliser.num_batches_tracked.fill_(100)

        getattr(callback, hook)(mock_trainer, module)

        averaged = _averaged(callback).normaliser
        for name, buffer in module.normaliser.named_buffers():
            assert torch.equal(averaged.get_buffer(name), buffer)

    def test_does_not_copy_parameters(
        self, mock_trainer: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Leave averaged parameters to the parent callback."""
        callback = EMAWeightAveragingCallback(decay_rate=0.99, every_n_steps=1)
        monkeypatch.setattr(WeightAveraging, "on_train_batch_end", MagicMock())
        module = NormalisedLightningModule()
        callback._average_model = AveragedModel(module, use_buffers=True)
        before = _averaged(callback).layer.weight.clone()
        with torch.no_grad():
            module.layer.weight.fill_(7.0)

        callback.on_train_batch_end(mock_trainer, module)

        assert torch.equal(_averaged(callback).layer.weight, before)
