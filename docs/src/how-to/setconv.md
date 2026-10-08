# Convert irregular observations to a grid with SetConv

SetConv creates gridded features from point observations, even if the observation
locations change over time. This is useful for exploratory integration of
non-gridded measurements such as Argo floats.

CryoCast provides two PyTorch components in `cryocast.models.common`:

- `SetConv`: Gaussian kernel aggregation producing one observation-density channel
  followed by density-normalised feature channels.
- `SetConvCNN`: the same aggregation followed by a two-layer CNN, producing
  configurable gridded feature channels that can be passed to a gridded model.

The existing `cryocast.models.encoders.SetConvEncoder` remains the fixed-coordinate,
great-circle-distance encoder for the current gridded pipeline. In contrast,
`SetConv` and `SetConvCNN` accept moving point positions on each call and expect
planar projected coordinates. The two APIs are intentionally distinct.

This is a **model component**, not yet a native Argo ingestion pathway. Existing
`CommonDataModule` and `BaseEncoder` interfaces still use gridded `NTCHW` data.
A future data adapter must supply the point values, positions and masks, and
project observation coordinates into the same metric space as the query grid.
No existing training/data configuration is changed by using these modules.

## Inputs and outputs

| Argument | Shape | Meaning |
| --- | --- | --- |
| `values` | `(B, T, N, C)` | Per-observation measurement values |
| `positions` | `(B, T, N, 2)` | Per-observation projected `(x, y)` coordinates |
| `grid_positions` | `(H, W, 2)` | Query coordinates, shared across batches and time |
| `valid_mask` (optional) | `(B, T, N)` | Boolean mask excluding invalid or padded points |
| `SetConv` output | `(B, T, C + 1, H, W)` | Density, then normalised channels |
| `SetConvCNN` output | `(B, T, C_out, H, W)` | Learned gridded features |

The time dimension can be omitted from both input arrays and mask for a
single-time `(B, N, C)` operation. The output then has `(B, C_out, H, W)`
(or `(B, C + 1, H, W)` for `SetConv`).

`N` is a padded maximum across the batch; a boolean mask excludes padded
observations. Masked positions/values may be NaN, but valid observations must
be finite. An empty/all-masked point set yields zero density and zero
normalised signals. The CNN may still produce nonzero outputs through biases.

## Example

```python
import torch

from cryocast.models.common import SetConvCNN

rows, cols = torch.meshgrid(
    torch.linspace(0, 1, 8),
    torch.linspace(0, 1, 12),
    indexing="ij",
)
grid_xy = torch.stack((cols, rows), dim=-1)

# Two time steps, two sensors. The sensors move between timesteps.
positions = torch.tensor(
    [[[[0.1, 0.3], [0.7, 0.8]], [[0.2, 0.3], [0.8, 0.8]]]]
)
values = torch.tensor([[[[0.2], [0.8]], [[0.4], [0.9]]]])
valid = torch.tensor([[[True, True], [True, False]]])

adapter = SetConvCNN(
    in_channels=1,
    out_channels=8,
    lengthscale=0.15,
    hidden_channels=16,
    chunk_size=96,
)
gridded = adapter(values, positions, grid_xy, valid)
assert gridded.shape == (1, 2, 8, 8, 12)
```

## Calculation and numerical considerations

For query point `q` and valid observations `(p_i, v_i)`, SetConv computes

\[
w_i(q) = \exp\left(-\frac{\lVert q-p_i\rVert^2}{2\ell^2}\right)
\]

and returns `density = sum(w_i)` together with
`signal = sum(w_i * v_i) / max(density, eps)` for each measurement channel.
Signals become zero when the density is below `eps`, so unsupported cells do
not contain arbitrary values. The length scale `ell` is positive by construction
and learnable by default.

The kernel is permutation-invariant with respect to observation order.
The positions are evaluated on **every call**, including for each time step;
there is no cached station arrangement. `chunk_size` caps the number of grid
queries evaluated at once, reducing temporary weight memory to roughly
`O(B * T * chunk_size * N)` instead of `O(B * T * H * W * N)`.

Use a *projected*, locally meaningful planar coordinate system (such as
EASE2 with consistently converted units). Applying this Euclidean distance
kernel directly to latitude/longitude degrees, especially near the poles or
across the antimeridian, is not geographically correct. The numerical
`lengthscale` uses the same units as the projected coordinates and the query
grid.

This provides a baseline for SetConv vs direct off-grid attention. It does not
claim an observed forecast-skill improvement without a separate, controlled
training/evaluation comparison.
