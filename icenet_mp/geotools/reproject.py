import logging
import time
from itertools import product

import numpy as np
from haversine import haversine_vector

from icenet_mp.types import ArrayHWV, ArrayIndices2D

logger = logging.getLogger(__name__)


def nearest_neighbour_indices(
    input_latlons: ArrayHWV, output_latlons: ArrayHWV
) -> tuple[ArrayIndices2D, ArrayIndices2D]:
    """Calculate the nearest neighbour input cell for each cell in the output grid.

    Args:
        input_latlons: Array of shape [input_h, input_w, 2] containing the latitudes and
            longitudes of each cell in the input grid.
        output_latlons: Array of shape [output_h, output_w, 2] containing the latitudes
            and longitudes of each cell in the output grid.

    Returns:
        Tuple of (nn_indices_h, nn_indices_w) where each is a numpy array of shape
        [output_height, output_width] containing, for each output cell, the index in
        the H and W dimensions of the nearest neighbour input cell that should be
        used as the source.

    """
    # We record the time taken in reprojection as this can be slow
    start = time.perf_counter()

    # Start by validating the shapes of the input and output lat/lon arrays
    if input_latlons.ndim != 3 or input_latlons.shape[2] != 2:  # noqa: PLR2004
        msg = f"Input lat/lons must have shape [input_h, input_w, 2], but got shape {input_latlons.shape}"
        raise ValueError(msg)
    input_h, input_w = int(input_latlons.shape[0]), int(input_latlons.shape[1])
    if output_latlons.ndim != 3 or output_latlons.shape[2] != 2:  # noqa: PLR2004
        msg = f"Output lat/lons must have shape [output_h, output_w, 2], but got shape {output_latlons.shape}"
        raise ValueError(msg)
    output_h, output_w = int(output_latlons.shape[0]), int(output_latlons.shape[1])
    logger.warning(
        "Calculating reprojection from input grid (%d x %d) to output grid (%d x %d)...",
        input_h,
        input_w,
        output_h,
        output_w,
    )

    # We want to find the closest input grid point for each output grid point. If we
    # try to fully vectorise the call, generating a single array of shape
    # [n_output_latlons, n_input_latlons], then we will run out of memory.
    # Instead we loop over the output points, using argmin to reduce to the closest
    # source point for each output point. We then look up the source grid indices
    # for that source point and store each of the height and width indices in an
    # array of [output_h, output_w]. This allows easy application of the index
    # lookup during the forward pass.
    input_indices = list(product(range(input_h), range(input_w)))
    input_latlons_flat = input_latlons.reshape(-1, 2)
    output_latlons_flat = output_latlons.reshape(-1, 2)
    closest_src_point_indices = np.array(
        [
            input_indices[
                np.argmin(
                    haversine_vector(input_latlons_flat, output_latlon, comb=True)
                )
            ]
            for output_latlon in output_latlons_flat
        ]
    )  # [output_h * output_w, 2]

    # Construct grids of shape [output_h, output_w] that give the indices of the
    # closest input point for each output point
    nn_indices_h = closest_src_point_indices[:, 0].reshape(output_h, output_w)
    nn_indices_w = closest_src_point_indices[:, 1].reshape(output_h, output_w)

    logger.info(
        "Reprojection calculation took %.2f seconds", time.perf_counter() - start
    )
    return (nn_indices_h, nn_indices_w)
