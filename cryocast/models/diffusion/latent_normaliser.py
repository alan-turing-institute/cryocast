import torch
from torch import nn


class LatentNormaliser(nn.Module):
    """Normalise latents to zero mean and unit variance per channel.

    The per-channel mean and variance are measured over the first ``n_batches``
    training batches. After that they are fixed, as they are outside training. They
    are stored as buffers, so a trained model is sampled with the same values it was
    trained with.

    EMA weight averaging copies these buffers instead of averaging them (see
    ``exclude_from_weight_averaging``), so the averaged model always uses the same
    statistics as training.
    """

    # Tells EMAWeightAveragingCallback to copy these buffers instead of averaging them
    exclude_from_weight_averaging = True

    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor

    def __init__(
        self, channels: int, *, n_batches: int = 100, eps: float = 1e-5
    ) -> None:
        """Initialise the normaliser.

        Args:
            channels: Number of latent channels.
            n_batches: Number of training batches to average the statistics over,
                before fixing them. Must be at least 1.
            eps: Added to the variance to avoid dividing by zero.

        """
        super().__init__()
        if n_batches < 1:
            msg = f"n_batches must be at least 1, got {n_batches}."
            raise ValueError(msg)
        self.n_batches = n_batches
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, *latents: torch.Tensor) -> None:
        """Update the running statistics from [N, T, C, H, W] latents.

        Does nothing outside training, or once ``n_batches`` batches have been seen.
        """
        if not self.training or self.num_batches_tracked >= self.n_batches:
            return
        values = torch.cat(
            [latent.detach().transpose(0, 2).flatten(1) for latent in latents], dim=1
        ).float()  # (C, N * T * H * W)
        # Average the mean and the mean of squares over batches, so the variance
        # also includes the differences between batch means (e.g. seasons)
        weight = 1.0 / float(self.num_batches_tracked + 1)
        batch_mean = values.mean(dim=1)
        batch_sq_mean = values.var(dim=1, unbiased=False) + batch_mean**2
        sq_mean = (self.running_var + self.running_mean**2).lerp_(batch_sq_mean, weight)
        self.running_mean.lerp_(batch_mean, weight)
        self.running_var.copy_((sq_mean - self.running_mean**2).clamp_(min=0.0))
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
