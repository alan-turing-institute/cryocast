import re

import pytest
import torch

from icenet_mp.models.processors import (
    BaseProcessor,
    DDPMProcessor,
    NullProcessor,
    UNetProcessor,
    VitProcessor,
)
from icenet_mp.types import DataSpace, ProcessorOutput


@pytest.mark.parametrize("test_batch_size", [1, 2])
@pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (3, 100, 200)])
@pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
@pytest.mark.parametrize("test_n_history_steps", [1, 2])
class TestBaseProcessor:
    def test_rollout(
        self,
        test_batch_size: int,
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
    ) -> None:
        latent_space = DataSpace(
            name="latent", channels=test_latent_chw[0], shape=test_latent_chw[1:]
        )
        processor = BaseProcessor(
            data_space=latent_space,
            n_forecast_steps=test_n_forecast_steps,
            n_history_steps=test_n_history_steps,
        )
        with pytest.raises(
            NotImplementedError,
            match=r"If you are using the default forward method, you must implement rollout.",
        ):
            processor.rollout(
                torch.randn(
                    test_batch_size,
                    test_n_history_steps,
                    *test_latent_chw,
                )
            )


class RecordingProcessor(BaseProcessor):
    """A test double that records the input it receives on each forward call."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialise a RecordingProcessor."""
        super().__init__(**kwargs)
        self.calls: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(x)
        return x[:, -self.data_space.channels :, :, :]


class TestRolloutSlidingWindow:
    """Regression tests for issue #272.

    Every prediction should be conditioned on the full window of n_history_steps
    timesteps, not a single timestep, and once n_forecast_steps > n_history_steps
    the window should slide forward using the most recently produced timestep
    rather than replaying stale original history.
    """

    def test_forward_receives_full_concatenated_window(self) -> None:
        latent_space = DataSpace(name="latent", channels=2, shape=(4, 4))
        n_history_steps, n_forecast_steps = 3, 5
        processor = RecordingProcessor(
            data_space=latent_space,
            n_forecast_steps=n_forecast_steps,
            n_history_steps=n_history_steps,
        )
        x = torch.randn(1, n_history_steps, latent_space.channels, *latent_space.shape)
        processor.rollout(x)

        assert len(processor.calls) == n_forecast_steps
        for call in processor.calls:
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
        torch.testing.assert_close(processor.calls[0], expected_first_window)

        # Later windows drop the oldest timestep and append the newest prediction. The
        # RecordingProcessor predicts its newest input timestep, which starts as h2.
        _, h1, h2 = x.unbind(dim=1)
        assert torch.equal(processor.calls[1], torch.cat([h1, h2, h2], dim=1))
        assert torch.equal(processor.calls[2], torch.cat([h2, h2, h2], dim=1))

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


@pytest.mark.parametrize("test_batch_size", [1, 2])
@pytest.mark.parametrize("test_latent_chw", [(128, 32, 32), (3, 100, 200)])
@pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
@pytest.mark.parametrize("test_n_history_steps", [1, 2])
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
    @pytest.mark.parametrize("test_kernel_size", [-1, 0])
    def test_rejects_non_positive_kernel_size(self, test_kernel_size: int) -> None:
        latent_space = DataSpace(name="latent", channels=3, shape=(32, 32))
        with pytest.raises(ValueError, match=r"Kernel size must be greater than 0."):
            UNetProcessor(
                data_space=latent_space,
                kernel_size=test_kernel_size,
                n_forecast_steps=1,
                n_history_steps=1,
                start_out_channels=8,
            )

    @pytest.mark.parametrize("test_start_out_channels", [-1, 0])
    def test_rejects_non_positive_start_out_channels(
        self, test_start_out_channels: int
    ) -> None:
        latent_space = DataSpace(name="latent", channels=3, shape=(32, 32))
        with pytest.raises(
            ValueError, match=r"Start out channels must be greater than 0."
        ):
            UNetProcessor(
                data_space=latent_space,
                kernel_size=3,
                n_forecast_steps=1,
                n_history_steps=1,
                start_out_channels=test_start_out_channels,
            )

    @pytest.mark.parametrize(
        "test_latent_hw",
        [(100, 200), (32, 40), (16, 16), (16, 32)],
        ids=lambda hw: f"hw={hw}",
    )
    def test_rejects_latent_not_a_multiple_of_16_above_16(
        self, test_latent_hw: tuple[int, int]
    ) -> None:
        latent_space = DataSpace(name="latent", channels=3, shape=test_latent_hw)
        processor = UNetProcessor(
            data_space=latent_space,
            kernel_size=3,
            n_forecast_steps=1,
            n_history_steps=1,
            start_out_channels=8,
        )
        x = torch.randn(1, 1, latent_space.channels, *latent_space.shape)
        height, width = test_latent_hw
        msg = f"Latent space height ({height}) and width ({width}) must each be divisible by 16 with a factor more than 1."
        with pytest.raises(ValueError, match=re.escape(msg)):
            processor.rollout(x)

    @pytest.mark.parametrize("test_batch_size", [1, 2])
    @pytest.mark.parametrize("test_kernel_size", [1, 3])
    @pytest.mark.parametrize(
        "test_latent_chw", [(3, 32, 32), (4, 32, 64)], ids=lambda chw: f"chw={chw}"
    )
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
    @pytest.mark.parametrize("test_n_history_steps", [1, 2])
    @pytest.mark.parametrize("test_start_out_channels", [7, 16])
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
        x = torch.randn(
            test_batch_size,
            test_n_history_steps,
            latent_space.channels,
            *latent_space.shape,
        )

        result = processor.rollout(x)

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

    @pytest.mark.parametrize("test_batch_size", [1, 2])
    @pytest.mark.parametrize("test_latent_chw", [(4, 16, 16), (8, 32, 32)])
    @pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
    @pytest.mark.parametrize("test_n_history_steps", [1, 2])
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


@pytest.mark.parametrize("test_batch_size", [1, 2])
@pytest.mark.parametrize("test_latent_chw", [(4, 16, 16)])
@pytest.mark.parametrize("test_n_forecast_steps", [1, 2])
@pytest.mark.parametrize("test_n_history_steps", [1, 2])
@pytest.mark.parametrize("test_use_autoregressive", [True, False])
class TestDDPMProcessor:
    C_TARGET = 2

    def _make_processor(
        self,
        *,
        latent_chw: tuple[int, int, int],
        n_forecast_steps: int,
        n_history_steps: int,
        use_autoregressive: bool,
        target_channel_offset: int = 0,
    ) -> DDPMProcessor:
        combined = DataSpace(
            name="combined", channels=latent_chw[0], shape=latent_chw[1:]
        )
        target = DataSpace(name="target", channels=self.C_TARGET, shape=latent_chw[1:])
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

    def test_rejects_out_of_bounds_target_slice(
        self,
        test_batch_size: int,  # noqa: ARG002
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            self._make_processor(
                latent_chw=test_latent_chw,
                n_forecast_steps=test_n_forecast_steps,
                n_history_steps=test_n_history_steps,
                use_autoregressive=test_use_autoregressive,
                target_channel_offset=test_latent_chw[
                    0
                ],  # start == c_combined (out of range)
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

    def test_rejects_negative_target_slice(
        self,
        test_batch_size: int,  # noqa: ARG002
        test_latent_chw: tuple[int, int, int],
        test_n_forecast_steps: int,
        test_n_history_steps: int,
        test_use_autoregressive: bool,  # noqa: FBT001
    ) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            self._make_processor(
                latent_chw=test_latent_chw,
                n_forecast_steps=test_n_forecast_steps,
                n_history_steps=test_n_history_steps,
                use_autoregressive=test_use_autoregressive,
                target_channel_offset=-1,
            )

    def test_nonzero_target_channel_offset_inference_shape(
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
            target_channel_offset=1,  # C_TARGET=2, latent_chw[0]=4 -> valid, non-zero
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
