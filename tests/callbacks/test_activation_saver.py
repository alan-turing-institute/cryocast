import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from lightning import LightningModule
from torch import nn

from cryocast.callbacks import ActivationSaver


class MinimalReusedLayerModel(nn.Module):
    """Minimal model that calls the same layer twice in one forward pass."""

    def __init__(self) -> None:
        """Initialise the repeated-layer test model."""
        super().__init__()
        self.shared = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.shared.weight.copy_(2 * torch.eye(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared layer twice."""
        return self.shared(self.shared(x))


class MinimalRolloutModel(nn.Module):
    """Minimal model that calls `self.processor` twice, simulating two rollout steps."""

    def __init__(self) -> None:
        """Initialise the two-step rollout test model."""
        super().__init__()
        self.processor = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.processor.weight.copy_(2 * torch.eye(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the processor twice, as if performing a two-step rollout."""
        first_step = self.processor(x)
        return self.processor(first_step)


class TestActivationSaver:
    def test_attach_rejects_unknown_layer(self, tmp_path: Path) -> None:
        """Reject unknown layer paths."""
        saver = ActivationSaver(["missing"], tmp_path)
        model = nn.Sequential(nn.Linear(2, 2))

        with pytest.raises(ValueError, match="missing"):
            saver.attach(model)

    def test_layer_hook_keeps_first_fire_per_batch(
        self, tmp_path: Path, mock_trainer: MagicMock, mock_module: MagicMock
    ) -> None:
        """Keep only the first activation when a layer fires twice."""
        saver = ActivationSaver(["shared"], tmp_path)
        model = MinimalReusedLayerModel()
        saver.attach(model)
        saver.on_test_batch_start(mock_trainer, mock_module, {}, 0)

        inputs = torch.tensor([[1.0, 3.0]])
        output = model(inputs)

        assert torch.equal(output, 4 * inputs)
        assert torch.equal(saver._current_activations["shared"], 2 * inputs)
        saver.detach()

    def test_batch_payload_and_metadata_are_written(
        self, tmp_path: Path, mock_trainer: MagicMock, mock_module: MagicMock
    ) -> None:
        """Write batch payloads and evaluation metadata."""
        saver = ActivationSaver(["0"], tmp_path, save_inputs=True)
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        saver.attach(model)

        sic_batch = torch.tensor([[1.0, 2.0]])
        batch = {
            "sic": sic_batch,
            "metadata": "not-a-tensor",
        }
        saver.on_test_batch_start(mock_trainer, mock_module, batch, 7)
        model(sic_batch)
        saver.on_test_batch_end(mock_trainer, mock_module, None, batch, 7)

        payload_path = tmp_path / "batch_00007.pt"
        payload = torch.load(payload_path, weights_only=False)

        assert payload["batch_idx"] == 7
        assert payload["layer_paths"] == ["0"]
        assert set(payload["activations"]) == {"0"}
        assert set(payload["inputs"]) == {"sic"}
        assert torch.equal(payload["inputs"]["sic"], sic_batch)

        saver.on_test_end(mock_trainer, mock_module)
        metadata = json.loads((tmp_path / "metadata.json").read_text())

        assert metadata["layer_paths"] == ["0"]
        assert metadata["save_inputs"] is True
        assert metadata["batch_file_template"] == "batch_{batch_idx:05d}.pt"
        assert saver._handles == []

    def test_empty_layer_list_is_disabled(
        self, tmp_path: Path, mock_trainer: MagicMock, mock_module: MagicMock
    ) -> None:
        """Leave the callback disabled when no layers are configured."""
        output_dir = tmp_path / "activations"
        saver = ActivationSaver([], output_dir)

        saver.on_test_start(mock_trainer, mock_module)
        saver.on_test_batch_start(mock_trainer, mock_module, {}, 0)
        saver.on_test_batch_end(mock_trainer, mock_module, None, {}, 0)
        saver.on_test_end(mock_trainer, mock_module)

        assert not output_dir.exists()

    def test_processor_rollout_counter_keeps_first_step_only(
        self, tmp_path: Path, mock_trainer: MagicMock, mock_module: MagicMock
    ) -> None:
        """Increment the rollout counter per processor call and keep only step 0."""
        saver = ActivationSaver(["processor"], tmp_path)
        model = MinimalRolloutModel()
        saver.attach(model)
        saver.on_test_batch_start(mock_trainer, mock_module, {}, 0)

        inputs = torch.tensor([[1.0, 3.0]])
        model(inputs)

        assert torch.equal(saver._current_activations["processor"], 2 * inputs)
        saver.detach()

    def test_batch_end_warns_about_uncaptured_layers(
        self,
        tmp_path: Path,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when a configured layer never fired its hook this batch."""
        saver = ActivationSaver(["0"], tmp_path)
        model = nn.Sequential(nn.Linear(2, 2))
        saver.attach(model)

        saver.on_test_batch_start(mock_trainer, mock_module, {}, 0)
        # The model is never called, so "0"'s forward hook never fires.
        with caplog.at_level(logging.WARNING):
            saver.on_test_batch_end(mock_trainer, mock_module, None, {}, 0)

        assert "no activation captured for layers ['0']" in caplog.text
        saver.detach()

    def test_on_test_start_attaches_hooks_when_enabled(
        self,
        tmp_path: Path,
        mock_trainer: MagicMock,
        linear_lightning_module: LightningModule,
    ) -> None:
        """Attach hooks automatically when on_test_start runs and layers are configured."""
        saver = ActivationSaver(["layer"], tmp_path)

        saver.on_test_start(mock_trainer, linear_lightning_module)

        assert len(saver._handles) > 0
        saver.detach()

    def test_detach_stops_capturing_new_activations(
        self, tmp_path: Path, mock_trainer: MagicMock, mock_module: MagicMock
    ) -> None:
        """Stop capturing activations once the hooks have been detached."""
        saver = ActivationSaver(["shared"], tmp_path)
        model = MinimalReusedLayerModel()
        saver.attach(model)
        saver.on_test_batch_start(mock_trainer, mock_module, {}, 0)
        model(torch.tensor([[1.0, 3.0]]))
        assert saver._current_activations

        saver.detach()
        saver._current_activations = {}
        model(torch.tensor([[1.0, 3.0]]))

        assert saver._current_activations == {}
