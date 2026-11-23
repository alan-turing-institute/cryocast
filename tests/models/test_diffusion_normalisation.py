from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cryocast.callbacks.ema_weight_averaging_callback import EMAWeightAveragingCallback
from cryocast.models import EncodeProcessDecode
from cryocast.models.diffusion import LatentNormaliser
from cryocast.models.processors import DiffusionProcessor
from cryocast.types import DataSpace

CHANNELS = 2
SHAPE = (16, 16)


def _processor(
    *, normalise_latents: bool, use_autoregressive: bool = True
) -> DiffusionProcessor:
    return DiffusionProcessor(
        data_space=DataSpace(name="combined", channels=CHANNELS, shape=SHAPE),
        n_forecast_steps=2,
        n_history_steps=2,
        timesteps=4,
        start_out_channels=8,
        time_embed_dim=256,
        dropout_rate=0.0,
        use_autoregressive=use_autoregressive,
        normalise_latents=normalise_latents,
        loss=torch.nn.MSELoss(),
    )


def _fit_statistics(processor: DiffusionProcessor, *latents: torch.Tensor) -> None:
    """Measure the latent statistics of a processor on one batch."""
    normaliser = processor.latent_normaliser
    assert normaliser is not None
    normaliser.n_batches = 1
    normaliser.update(*latents)


def _model(cfg_input_space: DictConfig) -> EncodeProcessDecode:
    """Return a small encode-process-decode model with a normalised diffusion processor."""
    return EncodeProcessDecode(
        name="normalised-diffusion",
        encoders=DictConfig(
            {
                "latent_space": SHAPE,
                cfg_input_space["name"]: {
                    "_target_": "cryocast.models.encoders.NaiveLinearEncoder"
                },
            }
        ),
        processor=DictConfig(
            {
                "_target_": "cryocast.models.processors.DiffusionProcessor",
                "timesteps": 4,
                "start_out_channels": 8,
                "dropout_rate": 0.0,
                "normalise_latents": True,
                "loss": {"_target_": "torch.nn.MSELoss"},
            }
        ),
        decoder=DictConfig({"_target_": "cryocast.models.decoders.NaiveLinearDecoder"}),
        hemisphere="north",
        input_spaces=[cfg_input_space],
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        metrics=[],
        n_forecast_steps=1,
        n_history_steps=2,
        output_space=DictConfig(
            {"channels": 1, "name": cfg_input_space["name"], "shape": SHAPE}
        ),
        optimizer=DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-2}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
        target_variable_indices=[2],
    )


class _DataModule(LightningDataModule):
    """Two training batches of random inputs, with future inputs for the latent loss."""

    def __init__(self, cfg_input_space: DictConfig) -> None:
        """Make the samples."""
        super().__init__()
        name, channels = cfg_input_space["name"], cfg_input_space["channels"]
        generator = torch.Generator().manual_seed(438)
        self.samples = [
            {
                name: 3.0 + torch.rand(2, channels, *SHAPE, generator=generator),
                f"future_{name}": 3.0
                + torch.rand(1, channels, *SHAPE, generator=generator),
                "target": torch.rand(1, 1, *SHAPE, generator=generator),
            }
            for _ in range(4)
        ]

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Return the training batches."""
        return DataLoader(self.samples, batch_size=2)  # type: ignore[arg-type]


class _RecordStatistics(Callback):
    """Record the normaliser buffers at the start of every training batch."""

    def __init__(self) -> None:
        """Start with no records."""
        self.records: list[dict[str, torch.Tensor]] = []

    def on_train_batch_start(
        self, _trainer: Trainer, pl_module: LightningModule, *_args: Any
    ) -> None:
        """Record a copy of the normaliser buffers."""
        normaliser = _normaliser(pl_module)
        self.records.append({k: v.clone() for k, v in normaliser.state_dict().items()})


def _normaliser(module: torch.nn.Module) -> LatentNormaliser:
    """Return the latent normaliser of a model."""
    normaliser = cast("EncodeProcessDecode", module).processor.latent_normaliser
    assert isinstance(normaliser, LatentNormaliser)
    return normaliser


def _fit(
    model: EncodeProcessDecode,
    datamodule: LightningDataModule,
    tmp_path: Path,
    callbacks: list[Callback],
) -> None:
    """Train the model for a few steps on the CPU."""
    trainer = Trainer(
        accelerator="cpu",
        callbacks=callbacks,
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        logger=False,
        max_steps=3,
    )
    trainer.fit(model, datamodule=datamodule)


class TestLatentStatisticsFittedBeforeTraining:
    def test_statistics_are_fixed_from_the_first_step_and_kept_by_ema(
        self, cfg_input_space: DictConfig, tmp_path: Path
    ) -> None:
        model = _model(cfg_input_space)
        record = _RecordStatistics()
        ema = EMAWeightAveragingCallback(decay_rate=0.9, every_n_steps=1)

        _fit(model, _DataModule(cfg_input_space), tmp_path, [record, ema])

        # Fitted over n_batches batches before the first step, going round the two
        # training batches as often as needed
        first = record.records[0]
        assert float(first["num_batches_tracked"]) == _normaliser(model).n_batches
        assert not torch.equal(
            first["running_var"], torch.ones_like(first["running_var"])
        )
        # Unchanged by training, and by EMA, whose weights are copied back at the end
        for buffers in [*record.records[1:], _normaliser(model).state_dict()]:
            for name, value in buffers.items():
                assert torch.equal(value, first[name])
        assert ema._average_model is not None
        averaged = _normaliser(ema._average_model.module)
        for name, value in averaged.state_dict().items():
            assert torch.equal(value, first[name])

    def test_fitted_statistics_are_not_measured_again(
        self, cfg_input_space: DictConfig, tmp_path: Path
    ) -> None:
        model = _model(cfg_input_space)
        normaliser = _normaliser(model)
        normaliser.running_mean.fill_(7.0)
        normaliser.num_batches_tracked.fill_(normaliser.n_batches)

        _fit(model, _DataModule(cfg_input_space), tmp_path, [])

        assert (normaliser.running_mean == 7.0).all()

    def test_fitting_without_datamodule_raises(
        self, cfg_input_space: DictConfig, tmp_path: Path
    ) -> None:
        model = _model(cfg_input_space)
        trainer = Trainer(
            accelerator="cpu",
            default_root_dir=tmp_path,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=0,
            logger=False,
            max_steps=1,
        )
        with pytest.raises(RuntimeError, match="needs a datamodule"):
            trainer.fit(model, _DataModule(cfg_input_space).train_dataloader())


class TestDiffusionProcessorNormalisation:
    def test_disabled_by_default(self) -> None:
        processor = DiffusionProcessor(
            data_space=DataSpace(name="combined", channels=CHANNELS, shape=SHAPE),
            n_forecast_steps=2,
            n_history_steps=2,
            timesteps=4,
            start_out_channels=8,
            time_embed_dim=256,
            loss=torch.nn.MSELoss(),
        )
        assert processor.latent_normaliser is None
        assert not any("latent_normaliser" in key for key in processor.state_dict())

    @pytest.mark.parametrize(
        "use_autoregressive", [True, False], ids=["autoregressive", "parallel"]
    )
    def test_training_diffuses_normalised_latents(
        self,
        make_latents: Callable[..., torch.Tensor],
        per_channel_stats: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        monkeypatch: pytest.MonkeyPatch,
        *,
        use_autoregressive: bool,
    ) -> None:
        processor = _processor(
            normalise_latents=True, use_autoregressive=use_autoregressive
        )
        clean: list[torch.Tensor] = []
        conditions: list[torch.Tensor] = []
        q_sample = processor.diffusion.q_sample
        forward = processor.model.forward

        def record_clean(
            x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
        ) -> torch.Tensor:
            clean.append(x_start.detach())
            return q_sample(x_start, t, noise)

        def record_condition(
            y: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
        ) -> torch.Tensor:
            conditions.append(cond.detach())
            return forward(y, t, cond)

        monkeypatch.setattr(processor.diffusion, "q_sample", record_clean)
        monkeypatch.setattr(processor.model, "forward", record_condition)
        x, y = make_latents(seed=1), make_latents(seed=2)
        _fit_statistics(processor, x, y)

        output = processor.rollout(x, y)

        # The statistics come from this batch, so both inputs are close to unit scale
        assert int(processor.latent_normaliser.num_batches_tracked) == 1  # type: ignore[union-attr]
        for tensor in (clean[0], conditions[0]):
            per_channel = tensor.unflatten(1, (-1, CHANNELS))
            mean, std = per_channel_stats(per_channel)
            assert mean.abs().max() < 0.2
            assert (std - 1).abs().max() < 0.2
        assert output.loss is not None
        assert torch.isfinite(output.loss)

    def test_training_without_fitted_statistics_raises(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        processor = _processor(normalise_latents=True)
        with pytest.raises(RuntimeError, match="must be measured before training"):
            processor.rollout(make_latents(seed=1), make_latents(seed=2))

    def test_metrics_prediction_is_in_original_units(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        processor = _processor(normalise_latents=True)
        x, y = make_latents(seed=1), make_latents(seed=2)
        _fit_statistics(processor, x, y)
        prediction = processor.rollout(x, y).prediction
        # Untrained predictions are noisy, but must be on the scale of the latents
        assert prediction[:, :, 0].mean() > 2.0
        assert prediction[:, :, 1].mean() < 0.0

    @pytest.mark.parametrize(
        "use_autoregressive", [True, False], ids=["autoregressive", "parallel"]
    )
    def test_inference_returns_original_units(
        self,
        make_latents: Callable[..., torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
        *,
        use_autoregressive: bool,
    ) -> None:
        processor = _processor(
            normalise_latents=True, use_autoregressive=use_autoregressive
        )
        _fit_statistics(processor, make_latents(seed=1), make_latents(seed=2))
        processor.eval()
        histories: list[torch.Tensor] = []

        def zero_sample(y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            histories.append(cond)
            return torch.zeros_like(y)

        monkeypatch.setattr(processor, "_run_reverse_diffusion", zero_sample)
        x = make_latents(seed=3)
        prediction = processor.rollout(x).prediction

        # A zero sample in normalised units is the running mean in original units
        normaliser = processor.latent_normaliser
        assert normaliser is not None
        expected = normaliser.running_mean.view(1, 1, -1, 1, 1).expand_as(prediction)
        torch.testing.assert_close(prediction, expected)
        # The history passed to the sampler is normalised
        torch.testing.assert_close(histories[0], normaliser.normalise(x).flatten(1, 2))

    def test_normalised_and_plain_processors_load_their_own_checkpoints(
        self, make_latents: Callable[..., torch.Tensor]
    ) -> None:
        trained = _processor(normalise_latents=True)
        _fit_statistics(trained, make_latents(seed=1), make_latents(seed=2))
        restored = _processor(normalise_latents=True)
        restored.load_state_dict(trained.state_dict())
        assert restored.latent_normaliser is not None
        assert trained.latent_normaliser is not None
        torch.testing.assert_close(
            restored.latent_normaliser.running_mean,
            trained.latent_normaliser.running_mean,
        )
        plain = _processor(normalise_latents=False)
        plain.load_state_dict(_processor(normalise_latents=False).state_dict())
