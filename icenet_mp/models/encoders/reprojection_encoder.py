import logging
from functools import cache
from typing import Any

import numpy as np
import torch
from torch import nn

from icenet_mp.geotools import nearest_neighbour_indices
from icenet_mp.types import ArrayHWV, DataSpace, TensorNCHW

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class ReprojectionEncoder(BaseEncoder):
    """Encoder that reprojects data from a source projection to a target projection.

    Each cell in the target projection takes its value from the nearest neighbour in the
    source.

    Input space:
        TensorNTCHW with (batch_size, n_timeslices, input_channels, input_height, input_width)

    Latent space:
        TensorNTCHW with (batch_size, n_timeslices, input_channels, latent_height, latent_width)
    """

    def __init__(
        self, *, data_space_in: DataSpace, project_to: str, **kwargs: Any
    ) -> None:
        """Initialise a ReprojectionEncoder."""
        super().__init__(
            output_channels=data_space_in.channels,
            data_space_in=data_space_in,
            **kwargs,
        )

        # Check details of the input data space
        self.project_from = self.data_space_in.name
        input_lats = len(self.latitudes.get(self.project_from, []))
        input_lons = len(self.longitudes.get(self.project_from, []))
        input_area = self.data_space_in.area
        if input_lats != input_area or input_lons != input_area:
            msg = (
                f"Input dataset '{self.project_from}' has {input_lats} latitudes and "
                f"{input_lons} longitudes but {input_area} of each are needed."
            )
            raise ValueError(msg)

        # Check details of the output data space to project to
        self.project_to = project_to
        output_lats = len(self.latitudes.get(self.project_to, []))
        output_lons = len(self.longitudes.get(self.project_to, []))
        output_area = self.data_space_out.area
        if output_lats != output_area or output_lons != output_area:
            msg = (
                f"Output dataset '{self.project_to}' has {output_lats} latitudes and "
                f"{output_lons} longitudes but {output_area} of each are needed."
            )
            raise ValueError(msg)

        # Add a cached method for calculating nearest neighbours
        # Adding this as a decorator would lead to memory leaks
        self.cached_nearest_neighbours = cache(self.nearest_neighbours)

        # Normalise the input across height and width separately for each channel
        self.norm = nn.BatchNorm2d(self.data_space_in.channels)

    def nearest_neighbours(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine the nearest neighbour input cell for each cell in the output grid.

        Returns:
            Tuple of (nn_indices_h, nn_indices_w) where each is a tensor of shape
            [output_height, output_width] containing, for each output cell, the index in
            the H and W dimensions of the nearest neighbour input cell that should be
            used as the source.

        """
        # Get lat/lon values for each cell in the input and output grids
        input_latlons: ArrayHWV = np.stack(
            (self.latitudes[self.project_from], self.longitudes[self.project_from]),
            axis=-1,
        ).reshape(*self.data_space_in.shape, 2)
        output_latlons: ArrayHWV = np.stack(
            (self.latitudes[self.project_to], self.longitudes[self.project_to]),
            axis=-1,
        ).reshape(*self.data_space_out.shape, 2)

        # Calculate nearest neighbour indices and return them as tensors
        nn_indices_h, nn_indices_w = nearest_neighbour_indices(
            input_latlons, output_latlons
        )
        return (
            torch.tensor(nn_indices_h, dtype=torch.long, device=device),
            torch.tensor(nn_indices_w, dtype=torch.long, device=device),
        )

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        """Forward step: encode input space into latent space by splitting into patches.

        Args:
            x: TensorNCHW with (batch_size, input_channels, input_height, input_width)

        Returns:
            TensorNCHW with (batch_size, latent_channels, latent_height, latent_width)

        """
        return self.norm(x[:, :, *self.cached_nearest_neighbours(x.device)])
