import re

import pytest
import torch
from omegaconf import OmegaConf

from icenet_mp.losses import TimeWeightedLoss
from icenet_mp.models.processors import (
    BaseProcessor,
    DDPMProcessor,
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
    ) -> DDPMProcessor:
        combined = DataSpace(
            name="combined", channels=self.LATENT_CHW[0], shape=self.LATENT_CHW[1:]
        )
        target = DataSpace(
            name="target", channels=self.C_TARGET, shape=self.LATENT_CHW[1:]
        )
        return DDPMProcessor(
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
            loss=torch.nn.MSELoss(),
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
    def test_training_supports_time_weighted_loss(
        self,
        test_batch_size: int,
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        """Restore forecast time for time-weighted latent diffusion loss."""
        processor = self._make_processor(
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
            use_autoregressive=test_use_autoregressive,
        )
        processor.loss_fn = TimeWeightedLoss(torch.nn.MSELoss())
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            *self.LATENT_CHW,
        )
        y = torch.randn(
            test_batch_size,
            test_n_forecast_steps,
            self.C_TARGET,
            *self.LATENT_CHW[1:],
        )

        result = processor.rollout(x, y)

        assert result.loss is not None
        assert result.loss.ndim == 0
        assert torch.isfinite(result.loss)

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


class TestDDPMProcessorLossConfig:
    def test_lead_time_exponent_rejected(self) -> None:
        latent = DataSpace(name="combined", channels=4, shape=(16, 16))
        loss = OmegaConf.create(
            {"_target_": "torch.nn.MSELoss", "lead_time_exponent": 1.0}
        )
        with pytest.raises(ValueError, match="lead_time_exponent"):
            DDPMProcessor(
                data_space=latent,
                data_space_target=latent,
                n_forecast_steps=1,
                n_history_steps=1,
                loss=loss,
            )
