import logging
import re
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from icenet_mp.losses import LeadTimeWeightedLoss
from icenet_mp.models.common.normalisations import ChannelNorm2D
from icenet_mp.models.processors import (
    BaseProcessor,
    ConvLSTMProcessor,
    DiffusionProcessor,
    NullProcessor,
    UNetProcessor,
    VitProcessor,
)
from icenet_mp.types import DataSpace, ProcessorOutput


class TestBaseProcessor:
    def test_rollout_requires_forward_implementation(self) -> None:
        latent_space = DataSpace(name="latent", channels=3, shape=(4, 4))
        processor = BaseProcessor(
            data_space=latent_space,
            n_forecast_steps=2,
            n_history_steps=2,
        )
        with pytest.raises(
            NotImplementedError,
            match=r"If you are using the default forward method, you must implement rollout.",
        ):
            processor.rollout(
                torch.randn(1, 2, latent_space.channels, *latent_space.shape)
            )


class TestRolloutSlidingWindow:
    """Regression tests for issue #272.

    Every prediction should be conditioned on the full window of n_history_steps
    timesteps, not a single timestep, and once n_forecast_steps > n_history_steps
    the window should slide forward using the most recently produced timestep
    rather than replaying stale original history.
    """

    def test_forward_receives_full_concatenated_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        latent_space = DataSpace(name="latent", channels=2, shape=(4, 4))
        n_history_steps, n_forecast_steps = 3, 5
        processor = NullProcessor(
            data_space=latent_space,
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
        )
        # Record the input of every forward call, delegating to the real forward
        calls: list[torch.Tensor] = []
        original_forward = processor.forward

        def recording_forward(x: torch.Tensor) -> torch.Tensor:
            calls.append(x)
            return original_forward(x)

        monkeypatch.setattr(processor, "forward", recording_forward)
        x = torch.randn(1, n_history_steps, latent_space.channels, *latent_space.shape)
        processor.rollout(x)

        assert len(calls) == n_forecast_steps
        for call in calls:
            # Every call must see the whole window, not a single timestep.
            assert call.shape == (
                1,
                latent_space.channels * n_history_steps,
                *latent_space.shape,
            )

        # The first call's window is exactly the original history, oldest to newest.
        expected_first_window = torch.cat(
            [x[:, idx_t, :, :, :] for idx_t in range(n_history_steps)], dim=1
        )
        assert torch.equal(calls[0], expected_first_window)

        # Later windows drop the oldest timestep and append the newest prediction. The
        # NullProcessor predicts its newest input timestep, which starts as h2.
        _, h1, h2 = x.unbind(dim=1)
        assert torch.equal(calls[1], torch.cat([h1, h2, h2], dim=1))
        assert torch.equal(calls[2], torch.cat([h2, h2, h2], dim=1))

    def test_null_processor_persistence_not_leapfrog(self) -> None:
        """Check NullProcessor.rollout reduces to true persistence.

        Before the fix, NullProcessor.rollout with n_forecast_steps > n_history_steps
        replayed the original history in a fixed cycle (e.g. [h0, h1, h2, h0, h1, h2])
        instead of repeating the most recent timestep. It should now reduce to true
        persistence: every forecast timestep equals the last known history timestep.
        """
        latent_space = DataSpace(name="latent", channels=1, shape=(1, 1))
        n_history_steps, n_forecast_steps = 3, 6
        processor = NullProcessor(
            data_space=latent_space,
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
        )
        history_values = [0.10, 0.15, 0.20]
        x = torch.tensor(history_values).reshape(1, n_history_steps, 1, 1, 1)

        result = processor.rollout(x)
        forecast = result.prediction.reshape(-1).tolist()

        assert forecast == pytest.approx([history_values[-1]] * n_forecast_steps)


@pytest.mark.parametrize("test_batch_size", [1, 2], ids=lambda b: f"batch{b}")
@pytest.mark.parametrize(
    "test_latent_chw", [(128, 32, 32), (3, 100, 200)], ids=["128x32x32", "3x100x200"]
)
@pytest.mark.parametrize("test_n_forecast_steps", [1, 2], ids=lambda n: f"forecast{n}")
@pytest.mark.parametrize("test_n_history_steps", [1, 2], ids=lambda n: f"history{n}")
class TestNullProcessor:
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        processor = NullProcessor(
            data_space=latent_space,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
        )
        result = processor.rollout(
            torch.randn(
                test_batch_size,
                test_n_history_steps,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert isinstance(result, ProcessorOutput)
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            latent_space.channels,
            *latent_space.shape,
        )
        assert result.loss is None


class TestUNetProcessor:
    @pytest.mark.parametrize(
        ("test_kernel_size", "test_start_out_channels", "match"),
        [
            (-1, 32, r"Kernel size must be greater than 0."),
            (0, 32, r"Kernel size must be greater than 0."),
            (1, -1, r"Start out channels must be greater than 0."),
            (1, 0, r"Start out channels must be greater than 0."),
        ],
        ids=[
            "kernel_size-negative",
            "kernel_size-zero",
            "start_out_channels-negative",
            "start_out_channels-zero",
        ],
    )
    def test_rejects_non_positive_hyperparameters(
        self, test_kernel_size: int, test_start_out_channels: int, match: str
    ) -> None:
        latent_space = DataSpace(name="latent", channels=3, shape=(32, 32))
        with pytest.raises(ValueError, match=match):
            UNetProcessor(
                data_space=latent_space,
                kernel_size=test_kernel_size,
                n_forecast_steps=1,
                n_history_steps=1,
                start_out_channels=test_start_out_channels,
            )

    @pytest.mark.parametrize(
        ("test_norm_type", "expected_norm"),
        [
            ("batchnorm", nn.BatchNorm2d),
            ("channelnorm", ChannelNorm2D),
            ("groupnorm", nn.GroupNorm),
        ],
        ids=["batchnorm", "channelnorm", "groupnorm"],
    )
    def test_norm_type_used_throughout(
        self, test_norm_type: str, expected_norm: type[nn.Module]
    ) -> None:
        latent_space = DataSpace(name="latent", channels=3, shape=(32, 32))
        processor = UNetProcessor(
            data_space=latent_space,
            kernel_size=1,
            n_forecast_steps=1,
            n_history_steps=1,
            norm_type=test_norm_type,
            start_out_channels=8,
        )
        norm_classes = (nn.BatchNorm2d, ChannelNorm2D, nn.GroupNorm)
        norms = [m for m in processor.modules() if isinstance(m, norm_classes)]
        assert norms
        assert all(isinstance(m, expected_norm) for m in norms)

    def test_rejects_unknown_norm_type(self) -> None:
        latent_space = DataSpace(name="latent", channels=3, shape=(32, 32))
        with pytest.raises(ValueError, match=r"Unknown norm_type: unknown"):
            UNetProcessor(
                data_space=latent_space,
                n_forecast_steps=1,
                n_history_steps=1,
                norm_type="unknown",
                start_out_channels=8,
            )

    @pytest.mark.parametrize(
        "test_latent_hw",
        [(100, 200), (32, 40), (16, 16), (16, 32)],
        ids=["100x200", "32x40", "16x16", "16x32"],
    )
    def test_rejects_latent_shape_not_divisible_by_16(
        self, test_latent_hw: tuple[int, int]
    ) -> None:
        latent_space = DataSpace(name="latent", channels=3, shape=test_latent_hw)
        processor = UNetProcessor(
            data_space=latent_space,
            kernel_size=1,
            n_forecast_steps=1,
            n_history_steps=1,
            start_out_channels=7,
        )
        height, width = test_latent_hw
        msg = (
            f"Latent space height ({height}) and width ({width}) must each be "
            "divisible by 16 with a factor more than 1."
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            processor.rollout(torch.randn(1, 1, 3, height, width))

    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=lambda b: f"batch{b}")
    @pytest.mark.parametrize(
        "test_latent_chw", [(128, 32, 32), (3, 32, 64)], ids=["128x32x32", "3x32x64"]
    )
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 2], ids=lambda n: f"forecast{n}"
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 2], ids=lambda n: f"history{n}"
    )
    @pytest.mark.parametrize(
        "test_start_out_channels", [7, 32], ids=lambda c: f"start_out{c}"
    )
    @pytest.mark.parametrize("test_kernel_size", [1, 3], ids=lambda k: f"kernel{k}")
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_kernel_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_start_out_channels: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        processor = UNetProcessor(
            data_space=latent_space,
            kernel_size=test_kernel_size,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            start_out_channels=test_start_out_channels,
        )
        result = processor.rollout(
            torch.randn(
                test_batch_size,
                test_n_history_steps,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert isinstance(result, ProcessorOutput)
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            latent_space.channels,
            *latent_space.shape,
        )


class TestVitProcessor:
    def test_rejects_non_square_input(self) -> None:
        latent_space = DataSpace(name="latent", channels=4, shape=(16, 32))
        with pytest.raises(ValueError, match="height and width"):
            VitProcessor(
                data_space=latent_space,
                n_forecast_steps=1,
                n_history_steps=1,
            )

    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=lambda b: f"batch{b}")
    @pytest.mark.parametrize(
        "test_latent_chw", [(4, 16, 16), (8, 32, 32)], ids=["4x16x16", "8x32x32"]
    )
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 2], ids=lambda n: f"forecast{n}"
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 2], ids=lambda n: f"history{n}"
    )
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent",
            channels=test_latent_chw[0],
            shape=test_latent_chw[1:],
        )
        processor = VitProcessor(
            data_space=latent_space,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            depth=1,
            emb_dim=16,
            heads=4,
            mlp_dim=32,
            patch_size=4,
        )
        result = processor.rollout(
            torch.randn(
                test_batch_size,
                test_n_history_steps,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert isinstance(result, ProcessorOutput)
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            latent_space.channels,
            *latent_space.shape,
        )


class TestDDPMProcessor:
    C_TARGET = 2
    LATENT_CHW = (4, 16, 16)

    def _make_processor(
        self,
        *,
        n_forecast_steps: int,
        n_history_steps: int,
        use_autoregressive: bool,
        target_channel_offset: int = 0,
        loss: DictConfig | torch.nn.Module | None = None,
    ) -> DiffusionProcessor:
        combined = DataSpace(
            name="combined", channels=self.LATENT_CHW[0], shape=self.LATENT_CHW[1:]
        )
        target = DataSpace(
            name="target", channels=self.C_TARGET, shape=self.LATENT_CHW[1:]
        )
        return DiffusionProcessor(
            data_space=combined,
            data_space_target=target,
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
            timesteps=2,
            start_out_channels=8,
            time_embed_dim=256,
            dropout_rate=0.0,
            use_autoregressive=use_autoregressive,
            target_channel_offset=target_channel_offset,
            loss=torch.nn.MSELoss() if loss is None else loss,
        )

    @pytest.mark.parametrize(
        "test_target_channel_offset",
        [-1, LATENT_CHW[0]],
        ids=["negative", "equal-to-combined-channels"],
    )
    def test_rejects_out_of_bounds_target_slice(
        self, test_target_channel_offset: int
    ) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            self._make_processor(
                n_forecast_steps=1,
                n_history_steps=1,
                use_autoregressive=False,
                target_channel_offset=test_target_channel_offset,
            )

    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=lambda b: f"batch{b}")
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 2], ids=lambda n: f"forecast{n}"
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 2], ids=lambda n: f"history{n}"
    )
    @pytest.mark.parametrize(
        "test_use_autoregressive", [True, False], ids=["autoregressive", "direct"]
    )
    # C_TARGET=2 inside 4 combined channels: offsets 0 and 1 are both valid
    @pytest.mark.parametrize(
        "test_target_channel_offset", [0, 1], ids=lambda o: f"offset{o}"
    )
    def test_inference_forward_shape(
        self,
        *,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,
        test_target_channel_offset: int,
    ) -> None:
        processor = self._make_processor(
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
            target_channel_offset=test_target_channel_offset,
        )
        x = torch.randn(test_batch_size, test_n_history_steps, *self.LATENT_CHW)
        with torch.no_grad():
            result = processor.rollout(x)

        assert isinstance(result, ProcessorOutput)
        assert result.loss is None
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            *self.LATENT_CHW,
        )

    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=lambda b: f"batch{b}")
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 2], ids=lambda n: f"forecast{n}"
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 2], ids=lambda n: f"history{n}"
    )
    @pytest.mark.parametrize(
        "test_use_autoregressive", [True, False], ids=["autoregressive", "direct"]
    )
    def test_training_returns_loss_that_backprops(
        self,
        *,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,
    ) -> None:
        processor = self._make_processor(
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
        )
        x = torch.randn(test_batch_size, test_n_history_steps, *self.LATENT_CHW)
        y = torch.randn(
            test_batch_size,
            test_n_forecast_steps,
            self.C_TARGET,
            *self.LATENT_CHW[1:],
        )
        result = processor.rollout(x, y)

        assert isinstance(result, ProcessorOutput)
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            *self.LATENT_CHW,
        )
        assert result.loss is not None
        assert result.loss.ndim == 0

        result.loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in processor.model.parameters()
        )

    # n_forecast_steps=3 differs from C_TARGET=2 so a swapped (C, T) unflatten
    # changes the number of lead times seen by the wrapped loss
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 3], ids=lambda n: f"forecast{n}"
    )
    @pytest.mark.parametrize(
        "test_use_autoregressive", [True, False], ids=["autoregressive", "direct"]
    )
    def test_training_supports_lead_time_weighted_loss(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        test_n_forecast_steps: int,
        test_use_autoregressive: bool,
    ) -> None:
        """Restore forecast time for lead-time weighted latent diffusion loss."""
        batch_size, n_history_steps = 2, 1
        processor = self._make_processor(
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=n_history_steps,
            use_autoregressive=test_use_autoregressive,
            loss=OmegaConf.create(
                {"_target_": "torch.nn.MSELoss", "lead_time_exponent": 2.0}
            ),
        )
        assert isinstance(processor.loss_fn, LeadTimeWeightedLoss)
        assert isinstance(processor.loss_fn._wrapped_loss, torch.nn.MSELoss)
        assert processor.loss_fn.exponent == pytest.approx(2.0)
        # Record the per-step NCHW slices that the wrapped loss receives
        wrapped_loss = processor.loss_fn._wrapped_loss
        original_forward = wrapped_loss.forward
        calls: list[tuple[torch.Size, torch.Size]] = []

        def recording_forward(
            prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            calls.append((prediction.shape, target.shape))
            return original_forward(prediction, target)

        monkeypatch.setattr(wrapped_loss, "forward", recording_forward)
        x = torch.randn(batch_size, n_history_steps, *self.LATENT_CHW)
        y = torch.randn(
            batch_size,
            test_n_forecast_steps,
            self.C_TARGET,
            *self.LATENT_CHW[1:],
        )

        result = processor.rollout(x, y)

        expected_steps = 1 if test_use_autoregressive else test_n_forecast_steps
        assert len(calls) == expected_steps
        expected_shape = (batch_size, self.C_TARGET, *self.LATENT_CHW[1:])
        assert all(shapes == (expected_shape,) * 2 for shapes in calls)
        assert result.loss is not None
        assert result.loss.ndim == 0
        assert torch.isfinite(result.loss)

    @pytest.mark.parametrize(
        ("test_use_autoregressive", "test_lead_time_exponent", "expect_warning"),
        [(True, 2.0, True), (False, 2.0, False), (True, None, False)],
        ids=["autoregressive-weighted", "direct-weighted", "autoregressive-unweighted"],
    )
    def test_warns_when_lead_time_weighting_ignored(
        self,
        caplog: pytest.LogCaptureFixture,
        *,
        test_use_autoregressive: bool,
        test_lead_time_exponent: float | None,
        expect_warning: bool,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="icenet_mp.models.processors"):
            self._make_processor(
                n_forecast_steps=2,
                n_history_steps=1,
                use_autoregressive=test_use_autoregressive,
                loss=OmegaConf.create(
                    {
                        "_target_": "torch.nn.MSELoss",
                        "lead_time_exponent": test_lead_time_exponent,
                    }
                ),
            )
        warned = any("has no effect" in r.getMessage() for r in caplog.records)
        assert warned is expect_warning

    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=lambda b: f"batch{b}")
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 2], ids=lambda n: f"forecast{n}"
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 2], ids=lambda n: f"history{n}"
    )
    @pytest.mark.parametrize(
        "test_use_autoregressive", [True, False], ids=["autoregressive", "direct"]
    )
    def test_non_target_channels_persist_from_last_frame(
        self,
        *,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,
    ) -> None:
        processor = self._make_processor(
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
        )
        x = torch.randn(test_batch_size, test_n_history_steps, *self.LATENT_CHW)
        with torch.no_grad():
            result = processor.rollout(x)

        s = processor.target_channel_offset
        assert s is not None
        c_target = processor.c_target
        non_target_idx = [
            i for i in range(self.LATENT_CHW[0]) if not (s <= i < s + c_target)
        ]
        last_frame = x[:, -1]

        for t_step in range(test_n_forecast_steps):
            assert torch.equal(
                result.prediction[:, t_step, non_target_idx],
                last_frame[:, non_target_idx],
            )

    @pytest.mark.parametrize(
        ("test_train_sampler", "test_infer_sampler"),
        [((None, 1.0), (1, 0.0)), ((1, 0.0), (None, 1.0)), ((2, 0.0), (1, 0.5))],
        ids=["ddpm-to-ddim", "ddim-to-ddpm", "ddim-to-other-ddim"],
    )
    def test_checkpoint_can_be_sampled_with_different_settings(
        self,
        tmp_path: Path,
        test_train_sampler: tuple[int | None, float],
        test_infer_sampler: tuple[int | None, float],
    ) -> None:
        """Weights saved with one (ddim_steps, eta) load and sample with another."""
        train_ddim_steps, train_eta = test_train_sampler
        infer_ddim_steps, infer_eta = test_infer_sampler

        # Train with one sampler setting; take one optimiser step so the saved
        # weights differ from a fresh initialisation.
        trained = self._make_processor(
            n_forecast_steps=1, n_history_steps=1, use_autoregressive=False
        )
        trained.set_sampler(ddim_steps=train_ddim_steps, eta=train_eta)
        x = torch.randn(2, 1, *self.LATENT_CHW)
        y = torch.randn(2, 1, self.C_TARGET, *self.LATENT_CHW[1:])
        optimizer = torch.optim.SGD(trained.parameters(), lr=0.1)
        loss = trained.rollout(x, y).loss
        assert loss is not None
        loss.backward()
        optimizer.step()

        checkpoint_path = tmp_path / "processor.pt"
        torch.save(trained.state_dict(), checkpoint_path)

        # Load into a processor with a different sampler setting. Strict loading
        # fails if the sampler setting added or removed any saved keys.
        restored = self._make_processor(
            n_forecast_steps=1, n_history_steps=1, use_autoregressive=False
        )
        restored.set_sampler(ddim_steps=infer_ddim_steps, eta=infer_eta)
        restored.load_state_dict(torch.load(checkpoint_path))

        restored_state = restored.state_dict()
        for name, tensor in trained.state_dict().items():
            torch.testing.assert_close(restored_state[name], tensor)

        expected_ddim_steps = (
            restored.timesteps if infer_ddim_steps is None else infer_ddim_steps
        )
        assert restored.ddim_steps == expected_ddim_steps
        assert restored.eta == infer_eta

        with torch.no_grad():
            result = restored.rollout(x)
        assert result.prediction.shape == (2, 1, *self.LATENT_CHW)


@pytest.mark.parametrize("test_batch_size", [1, 2])
@pytest.mark.parametrize("test_latent_chw", [(4, 16, 16)])
@pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
@pytest.mark.parametrize("test_n_history_steps", [1, 2])
@pytest.mark.parametrize("test_use_autoregressive", [True, False])
class TestDDIMProcessor:
    C_TARGET = 2
    TIMESTEPS = 4
    DDIM_STEPS = 2

    def _make_processor(
        self,
        *,
        latent_chw: tuple[int, int, int],
        n_forecast_steps: int,
        n_history_steps: int,
        use_autoregressive: bool,
        target_channel_offset: int = 0,
        ddim_steps: int | None = None,
        eta: float = 0.0,
        timesteps: int | None = None,
    ) -> DiffusionProcessor:
        combined = DataSpace(
            name="combined", channels=latent_chw[0], shape=latent_chw[1:]
        )
        target = DataSpace(name="target", channels=self.C_TARGET, shape=latent_chw[1:])
        return DiffusionProcessor(
            data_space=combined,
            data_space_target=target,
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
            timesteps=timesteps if timesteps is not None else self.TIMESTEPS,
            ddim_steps=ddim_steps if ddim_steps is not None else self.DDIM_STEPS,
            eta=eta,
            start_out_channels=8,
            time_embed_dim=256,
            dropout_rate=0.0,
            use_autoregressive=use_autoregressive,
            target_channel_offset=target_channel_offset,
            loss=torch.nn.MSELoss(),
        )

    def test_inference_forward_shape(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        processor = self._make_processor(
            latent_chw=test_latent_chw,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
        )
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )
        with torch.no_grad():
            result = processor.rollout(x)

        assert isinstance(result, ProcessorOutput)
        assert result.loss is None
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )

    def test_training_returns_loss_and_shape(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        processor = self._make_processor(
            latent_chw=test_latent_chw,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
        )
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )
        y = torch.randn(
            test_batch_size,
            test_n_forecast_steps,
            self.C_TARGET,
            *test_latent_chw[1:],
        )
        result = processor.rollout(x, y)

        assert isinstance(result, ProcessorOutput)
        assert result.loss is not None
        assert result.loss.ndim == 0
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )

    def test_non_target_channels_persist_from_last_frame(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        processor = self._make_processor(
            latent_chw=test_latent_chw,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
        )
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )
        with torch.no_grad():
            result = processor.rollout(x)

        s = processor.target_channel_offset
        assert s is not None
        c_target = processor.c_target
        non_target_idx = [
            i for i in range(test_latent_chw[0]) if not (s <= i < s + c_target)
        ]
        last_frame = x[:, -1]

        for t_step in range(test_n_forecast_steps):
            torch.testing.assert_close(
                result.prediction[:, t_step, non_target_idx],
                last_frame[:, non_target_idx],
            )

    def test_training_loss_backprops(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        processor = self._make_processor(
            latent_chw=test_latent_chw,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
        )
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )
        y = torch.randn(
            test_batch_size,
            test_n_forecast_steps,
            self.C_TARGET,
            *test_latent_chw[1:],
        )
        result = processor.rollout(x, y)
        assert result.loss is not None
        result.loss.backward()

        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in processor.model.parameters()
        )

    def test_rejects_ddim_steps_larger_than_timesteps(
        self,
        test_batch_size: int,  # noqa: ARG002
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        with pytest.raises(ValueError, match=r"ddim_steps=\d+ must be in the range"):
            self._make_processor(
                latent_chw=test_latent_chw,
                n_forecast_steps=test_n_forecast_steps,
                n_history_steps=test_n_history_steps,
                use_autoregressive=test_use_autoregressive,
                ddim_steps=self.TIMESTEPS + 1,
            )

    def test_rejects_ddim_steps_zero(
        self,
        test_batch_size: int,  # noqa: ARG002
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        with pytest.raises(ValueError, match=r"ddim_steps=\d+ must be in the range"):
            self._make_processor(
                latent_chw=test_latent_chw,
                n_forecast_steps=test_n_forecast_steps,
                n_history_steps=test_n_history_steps,
                use_autoregressive=test_use_autoregressive,
                ddim_steps=0,
            )

    def test_rejects_negative_eta(
        self,
        test_batch_size: int,  # noqa: ARG002
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        with pytest.raises(ValueError, match=r"eta=.+ must be in the range"):
            self._make_processor(
                latent_chw=test_latent_chw,
                n_forecast_steps=test_n_forecast_steps,
                n_history_steps=test_n_history_steps,
                use_autoregressive=test_use_autoregressive,
                eta=-0.1,
            )

    def test_eta_zero_sampling_is_deterministic(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        processor = self._make_processor(
            latent_chw=test_latent_chw,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
            eta=0.0,
        )
        processor.eval()
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )
        with torch.no_grad():
            torch.manual_seed(0)
            a = processor.rollout(x).prediction
            torch.manual_seed(0)
            b = processor.rollout(x).prediction
        torch.testing.assert_close(a, b)

    def test_ddim_steps_equal_to_timesteps_runs(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        """DDIM must accept ddim_steps == timesteps (walk every trained rung)."""
        processor = self._make_processor(
            latent_chw=test_latent_chw,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
            ddim_steps=self.TIMESTEPS,
        )
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )
        with torch.no_grad():
            result = processor.rollout(x)
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            test_latent_chw[0],
            *test_latent_chw[1:],
        )

    def test_rejects_eta_greater_than_one(
        self,
        test_batch_size: int,  # noqa: ARG002
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        with pytest.raises(ValueError, match=r"eta=.+ must be in the range"):
            self._make_processor(
                latent_chw=test_latent_chw,
                n_forecast_steps=test_n_forecast_steps,
                n_history_steps=test_n_history_steps,
                use_autoregressive=test_use_autoregressive,
                eta=1.1,
            )

    def test_ddim_timestep_sequence_is_evenly_spaced(
        self,
        test_batch_size: int,  # noqa: ARG002
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        """The DDIM timestep subset must be evenly-spaced from timesteps-1 down to 0."""
        processor = self._make_processor(
            latent_chw=test_latent_chw,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
            timesteps=10,
            ddim_steps=5,
        )
        torch.testing.assert_close(
            processor._ddim_timesteps,
            torch.tensor([9, 6, 4, 2, 0], dtype=torch.long),
        )


class TestConvLSTMProcessor:
    @pytest.mark.parametrize(
        ("test_hidden_channels", "test_n_layers", "test_dropout", "match"),
        [
            (0, 2, 0.0, r"hidden_channels must be greater than 0."),
            (4, 0, 0.0, r"n_layers must be greater than 0."),
            (4, 2, -0.1, r"dropout must be in the range \[0, 1\)."),
            (4, 2, 1.0, r"dropout must be in the range \[0, 1\)."),
        ],
        ids=[
            "hidden_channels-zero",
            "n_layers-zero",
            "dropout-negative",
            "dropout-one",
        ],
    )
    def test_rejects_invalid_hyperparameters(
        self,
        test_hidden_channels: int,
        test_n_layers: int,
        test_dropout: float,
        match: str,
    ) -> None:
        latent_space = DataSpace(name="latent", channels=2, shape=(4, 4))
        with pytest.raises(ValueError, match=match):
            ConvLSTMProcessor(
                data_space=latent_space,
                dropout=test_dropout,
                hidden_channels=test_hidden_channels,
                n_forecast_steps=2,
                n_history_steps=2,
                n_layers=test_n_layers,
            )

    def test_rejects_even_kernel_size(self) -> None:
        latent_space = DataSpace(name="latent", channels=2, shape=(4, 4))
        with pytest.raises(
            ValueError, match=r"kernel_size must be a positive odd integer."
        ):
            ConvLSTMProcessor(
                data_space=latent_space,
                kernel_size=2,
                n_forecast_steps=2,
                n_history_steps=2,
            )

    @pytest.mark.parametrize("test_batch_size", [1, 2], ids=lambda b: f"batch{b}")
    @pytest.mark.parametrize(
        "test_n_forecast_steps", [1, 4], ids=lambda n: f"forecast{n}"
    )
    @pytest.mark.parametrize(
        "test_n_history_steps", [1, 3], ids=lambda n: f"history{n}"
    )
    @pytest.mark.parametrize("test_n_layers", [1, 2], ids=lambda n: f"layers{n}")
    def test_forward_shape(
        self,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_n_layers: int,
    ) -> None:
        latent_space = DataSpace(name="latent", channels=4, shape=(8, 8))
        processor = ConvLSTMProcessor(
            data_space=latent_space,
            hidden_channels=6,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            n_layers=test_n_layers,
        )
        result = processor.rollout(
            torch.randn(
                test_batch_size,
                test_n_history_steps,
                latent_space.channels,
                *latent_space.shape,
            )
        )
        assert isinstance(result, ProcessorOutput)
        assert result.loss is None
        assert result.prediction.shape == (
            test_batch_size,
            test_n_forecast_steps,
            latent_space.channels,
            *latent_space.shape,
        )

    def test_backpropagates_through_history(self) -> None:
        latent_space = DataSpace(name="latent", channels=2, shape=(6, 6))
        processor = ConvLSTMProcessor(
            data_space=latent_space,
            hidden_channels=4,
            n_forecast_steps=2,
            n_history_steps=3,
            n_layers=2,
        )
        x = torch.randn(
            2, 3, latent_space.channels, *latent_space.shape, requires_grad=True
        )

        processor.rollout(x).prediction.square().mean().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, parameter in processor.named_parameters():
            assert parameter.grad is not None, f"{name} did not receive a gradient"
            assert torch.isfinite(parameter.grad).all(), (
                f"{name} has a non-finite gradient"
            )

    def test_zero_residual_head_reduces_to_persistence(self) -> None:
        latent_space = DataSpace(name="latent", channels=2, shape=(5, 5))
        processor = ConvLSTMProcessor(
            data_space=latent_space,
            hidden_channels=4,
            n_forecast_steps=4,
            n_history_steps=3,
            n_layers=1,
            residual=True,
        )
        nn.init.zeros_(processor.output_projection.weight)
        assert processor.output_projection.bias is not None
        nn.init.zeros_(processor.output_projection.bias)
        x = torch.randn(1, 3, latent_space.channels, *latent_space.shape)

        prediction = processor.rollout(x).prediction

        torch.testing.assert_close(prediction, x[:, -1:].expand_as(prediction))
