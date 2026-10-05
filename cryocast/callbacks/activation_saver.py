"""Forward-hook manager for capturing model intermediate activations.

Activations are captured during `trainer.test(...)` (i.e. the `imp evaluate`
command) which invokes `LightningModule.test_step`.

For an `EncodeProcessDecode` model the processor's internal forward runs once
per forecast step inside `BaseProcessor.rollout`, so the same hooked layer
fires multiple times per outer forward. To keep outputs small we persist only the **first
rollout step**. Encoder modules similarly fire once per history step inside
`BaseEncoder.rollout`; the first fire per batch is the one we save.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import nn

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


class ActivationSaver(Callback):
    """Register forward hooks and save captured activations to disk per batch.

    The ActivationSaver plays two roles:

    1. A set of PyTorch forward hooks attached to named submodules of the
       model (encoders, `processor.conv*`, decoder stages, ...). Each hook
       stores a detached CPU copy of its module's output tensor for the first
       rollout step of each test batch.
    2. A Lightning `Callback` whose `on_test_batch_end` flushes the buffered
       activations (and, optionally, a copy of the raw batch tensors) to a
       single `batch_{idx:05d}.pt` file.

    Args:
        layer_paths: Dotted module paths resolvable via `model.get_submodule`
            (equivalently, keys of `model.named_modules()`).
        output_dir: Directory for per-batch `.pt` files and a `metadata.json`.
        save_inputs: If True, also save the raw batch tensors alongside the
            activations.

    """

    BATCH_FILE_TEMPLATE = "batch_{batch_idx:05d}.pt"
    METADATA_FILE = "metadata.json"

    def __init__(
        self,
        layer_paths: Sequence[str],
        output_dir: Path | str,
        *,
        save_inputs: bool = True,
    ) -> None:
        """Initialise an ActivationSaver bound to a specific model."""
        super().__init__()
        self.layer_paths = list(layer_paths)
        self.output_dir = Path(output_dir)
        self.save_inputs = save_inputs

        self._enabled = len(self.layer_paths) > 0
        self._handles: list[RemovableHandle] = []
        self._rollout_idx: int = -1
        self._current_batch_idx: int = -1
        self._current_activations: dict[str, torch.Tensor] = {}
        self._current_inputs: dict[str, torch.Tensor] = {}

    def attach(self, model: nn.Module) -> None:
        """Resolve layer paths on the model and register all forward hooks.

        Raises:
            ValueError: If any requested layer path does not resolve to a
                submodule on the model.

        """
        named_modules = dict(model.named_modules())
        for name in self.layer_paths:
            logger.debug("Available activation layer: %s", name)
        missing = [path for path in self.layer_paths if path not in named_modules]
        if missing:
            available = sorted(name for name in named_modules if name)
            preview = ", ".join(available)
            msg = (
                f"Activation layers not found on model: {missing}. "
                f"Example available modules: {preview}"
            )
            raise ValueError(msg)

        # Reset rollout counter at the start of every outer forward pass.
        self._handles.append(
            model.register_forward_pre_hook(self.rollout_counter_reset)
        )

        # Increment rollout counter each time `processor.forward` runs.
        processor = getattr(model, "processor", None)
        if isinstance(processor, nn.Module):
            self._handles.append(
                processor.register_forward_pre_hook(self.rollout_counter_increment)
            )

        for path in self.layer_paths:
            module = named_modules[path]
            self._handles.append(
                module.register_forward_hook(self._make_layer_hook(path))
            )
            logger.info("Registered ActivationSaver hook on layer '%s'", path)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ActivationSaver attached: %d layer(s), output_dir=%s",
            len(self.layer_paths),
            self.output_dir,
        )

    def detach(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        logger.debug("ActivationSaver detached")

    def rollout_counter_increment(
        self,
        module: nn.Module,  # noqa: ARG002
        inputs: tuple[Any, ...],  # noqa: ARG002
    ) -> None:
        """Increment the rollout counter for this batch at each forward."""
        self._rollout_idx += 1

    def rollout_counter_reset(
        self,
        module: nn.Module,  # noqa: ARG002
        inputs: tuple[Any, ...],  # noqa: ARG002
    ) -> None:
        """Reset the processor rollout counter at the start of each batch."""
        self._rollout_idx = -1

    def _make_layer_hook(self, path: str):  # noqa: ANN202
        def _hook(
            module: nn.Module,  # noqa: ARG001
            module_input: tuple[Any, ...],  # noqa: ARG001
            module_output: Any,  # noqa: ANN401
        ) -> None:
            # Gate 1: skip any processor rollout step beyond the first.
            if self._rollout_idx > 0:
                return
            # Gate 2: keep only the first fire of each layer per batch.
            # `BaseEncoder.rollout` calls `self(...)` per history step, so
            # encoder-module hooks fire multiple times at _rollout_idx == -1.
            if path in self._current_activations:
                return
            if isinstance(module_output, torch.Tensor):
                self._current_activations[path] = module_output.detach().cpu()

        return _hook

    def on_test_batch_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Clear per-batch buffers and snapshot the raw batch tensors."""
        if self._enabled:
            self._current_batch_idx = batch_idx
            self._current_activations = {}
            self._current_inputs = {}
            if self.save_inputs and isinstance(batch, dict):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        self._current_inputs[key] = value.detach().cpu()

    def on_test_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        outputs: Any,  # noqa: ARG002, ANN401
        batch: Any,  # noqa: ARG002, ANN401
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Persist captured activations for this batch to a single `.pt` file."""
        if self._enabled:
            uncaptured = [
                path
                for path in self.layer_paths
                if path not in self._current_activations
            ]
            if uncaptured:
                logger.warning(
                    "Batch %d: no activation captured for layers %s; "
                    "the forward hook did not fire during the first rollout step.",
                    batch_idx,
                    uncaptured,
                )

            payload: dict[str, Any] = {
                "batch_idx": batch_idx,
                "layer_paths": self.layer_paths,
                "activations": self._current_activations,
            }
            if self.save_inputs:
                payload["inputs"] = self._current_inputs

            out_path = self.output_dir / self.BATCH_FILE_TEMPLATE.format(
                batch_idx=batch_idx
            )
            torch.save(payload, out_path)
            logger.debug(
                "Saved activations for batch %d to %s (%d layers).",
                batch_idx,
                out_path,
                len(self._current_activations),
            )

    def on_test_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Write a metadata file and remove forward hooks when the test loop ends."""
        if self._enabled:
            metadata_path = self.output_dir / self.METADATA_FILE
            metadata_path.write_text(
                json.dumps(
                    {
                        "layer_paths": self.layer_paths,
                        "save_inputs": self.save_inputs,
                        "batch_file_template": self.BATCH_FILE_TEMPLATE,
                        "note": "Only activations from the first processor rollout step are saved.",
                    },
                    indent=2,
                )
            )
            logger.info("ActivationSaver metadata written to %s", metadata_path)
            self.detach()

    def on_test_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        """Called when the test begins."""
        if not self._enabled:
            logger.info(
                "No layer paths specified for ActivationSaver; skipping hook attachment."
            )
            return
        self.attach(pl_module)
