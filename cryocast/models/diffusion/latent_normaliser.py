import torch
from torch import nn


class LatentNormaliser(nn.Module):
    """Normalise latents to zero mean and unit variance per channel.

    The per-channel mean and variance are running estimates, updated from each
    training batch and fixed outside training, like the statistics in batch
    normalisation. They are stored as buffers, so a trained model is sampled with
    the same values it was trained with. The first updates use a plain average
    over the batches seen so far, before switching to an exponential moving
    average with the given momentum.
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor

    def __init__(
        self, channels: int, *, momentum: float = 0.01, eps: float = 1e-5
    ) -> None:
        """Initialise the normaliser.

        Args:
            channels: Number of latent channels.
            momentum: Weight of each new batch in the running statistics, once
                more than ``1 / momentum`` batches have been seen. Must be in
                (0, 1].
            eps: Added to the variance to avoid dividing by zero.

        """
        super().__init__()
        if not 0.0 < momentum <= 1.0:
            msg = f"momentum must be in (0, 1], got {momentum}."
            raise ValueError(msg)
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, *latents: torch.Tensor) -> None:
        """Update the running statistics from [N, T, C, H, W] latents.

        Does nothing outside training.
        """
        if not self.training:
            return
        values = torch.cat(
            [latent.detach().transpose(0, 2).flatten(1) for latent in latents], dim=1
        ).float()  # (C, N * T * H * W)
        weight = max(self.momentum, 1.0 / float(self.num_batches_tracked + 1))
        self.running_mean.lerp_(values.mean(dim=1), weight)
        self.running_var.lerp_(values.var(dim=1, unbiased=False), weight)
        self.num_batches_tracked += 1

    def _shape(self, stat: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-channel statistic over an [N, T, C, H, W] tensor."""
        return stat.to(latent.dtype).view(1, 1, -1, 1, 1)

    def normalise(self, latent: torch.Tensor) -> torch.Tensor:
        """Return [N, T, C, H, W] latents with zero mean and unit variance."""
        std = (self.running_var + self.eps).sqrt()
        return (latent - self._shape(self.running_mean, latent)) / self._shape(
            std, latent
        )

    def denormalise(self, latent: torch.Tensor) -> torch.Tensor:
        """Return [N, T, C, H, W] latents in their original units."""
        std = (self.running_var + self.eps).sqrt()
        return latent * self._shape(std, latent) + self._shape(
            self.running_mean, latent
        )
