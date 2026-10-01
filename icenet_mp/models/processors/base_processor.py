from collections import deque
from collections.abc import Iterable

from torch import cat, nn, stack

from icenet_mp.types import DataSpace, ProcessorOutput, TensorNCHW, TensorNTCHW


class BaseProcessor(nn.Module):
    """Processor that converts latent input into latent output.

    Input space:
        TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)

    Output space:
        TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
    """

    def __init__(
        self,
        *,
        computes_loss_in_latent_space: bool = False,
        data_space: DataSpace,
        data_space_target: DataSpace | None = None,
        n_forecast_steps: int,
        n_history_steps: int,
        target_channel_offset: int | None = None,
    ) -> None:
        """Initialise a BaseProcessor."""
        super().__init__()
        self.computes_loss_in_latent_space = computes_loss_in_latent_space
        self.data_space = data_space
        self.data_space_target = data_space_target or data_space
        self.n_forecast_steps = n_forecast_steps
        self.n_history_steps = n_history_steps
        self.target_channel_offset = target_channel_offset
        # The latent spatial dimensions (H, W) for the inputs and target must match
        if self.data_space_target.shape != self.data_space.shape:
            msg = (
                f"Expected data_space_target.shape {self.data_space_target.shape} "
                f"to match data_space.shape {self.data_space.shape}"
            )
            raise ValueError(msg)

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: predict the next timestep from a window of history/forecast timesteps.

        Args:
            x: TensorNCHW with shape (batch_size, n_channels, latent_height, latent_width),
               i.e. the current window of n_history_steps timesteps concatenated along channels,
               ordered oldest to newest.

        Returns:
            TensorNCHW with shape (batch_size, n_channels, latent_height, latent_width),
            i.e. the single next predicted timestep.

        """
        msg = "If you are using the default forward method, you must implement rollout."
        raise NotImplementedError(msg)

    def rollout(self, x: TensorNTCHW, y: TensorNTCHW | None = None) -> ProcessorOutput:  # noqa: ARG002
        """Process in latent space across multiple timesteps.

        The default implementation slides a window of the n_history_steps most recent
        timesteps along, concatenating them along the channel dimension and calling
        `self.forward` once per forecast step to predict the next timestep, which is
        then appended to the window (dropping the oldest timestep) so that every
        prediction is conditioned on all of the currently-available history rather
        than a single timestep. This matches issue #272: earlier versions called
        `self.forward` on one history/forecast timestep at a time, which meant each
        forecast day only ever saw a single timestep of context and, once
        n_forecast_steps > n_history_steps, replayed stale original history timesteps
        in a fixed cycle instead of the most recently produced information.

        Override this method to handle the NTCHW tensors directly or to compute a custom
        loss using the target tensor `y` (e.g. for diffusion models). Set `loss` on the
        returned `ProcessorOutput` to supply a custom training loss; the caller will use
        it directly and skip its own loss computation.

        Args:
            x: Encoded input TensorNTCHW with (batch_size, n_history_steps, n_latent_channels_total, latent_height, latent_width)
            y: during training: Encoded target TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_target, latent_height, latent_width)
                where n_latent_channels_target = self.data_space_target.channels (<= n_latent_channels_total);
                otherwise: None

        Returns:
            ProcessorOutput with:
                prediction: TensorNTCHW with (batch_size, n_forecast_steps, n_latent_channels_total, latent_height, latent_width)
                loss: an optional processor-specific loss

        """
        # The current window of n_history_steps timesteps, oldest to newest
        window: deque[TensorNCHW] = deque(x.unbind(dim=1))

        # Slide the window forward, predicting one timestep at a time from the whole
        # window and then dropping the oldest timestep to make room for the prediction.
        outputs: list[TensorNCHW] = []
        for _ in range(self.n_forecast_steps):
            next_step = self.step(window)
            outputs.append(next_step)
            window.popleft()
            window.append(next_step)

        return ProcessorOutput(prediction=stack(outputs, dim=1))

    def step(self, window: Iterable[TensorNCHW]) -> TensorNCHW:
        """Predict the next timestep from a window of timesteps.

        Folds the window along the channel dimension, oldest to newest, and calls
        `self.forward` once to generate a single prediction for the next timestep.

        Args:
            window: n_history_steps TensorNCHW timesteps, oldest to newest, each with
                shape (batch_size, n_channels, latent_height, latent_width).

        Returns:
            TensorNCHW with shape (batch_size, n_channels, latent_height, latent_width),
            i.e. the single next predicted timestep.

        """
        return self(cat(list(window), dim=1))
