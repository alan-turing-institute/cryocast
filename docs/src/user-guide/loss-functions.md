# Loss functions

## Selecting a loss

The training loss is a Hydra config group. The default is set in `icenet_mp/config/base.yaml`
(`loss: amse`) and can be overridden on any command line:

```bash
imp train --config-name <config> loss=mse
imp train --config-name <config> loss=huber loss.delta=0.1
imp train --config-name <config> loss.mode=hybrid loss.spectral_weight=0.1 loss.lead_time_exponent=2
```

Each option corresponds to a file in `icenet_mp/config/loss/`, whose header comments carry the full
parameter documentation; this page summarises how to choose between them.

## Supported losses

| `loss=` | what it is | pros | cons |
|---|---|---|---|
| `huber` | quadratic below `delta`, linear above | smooth near zero, robust to outliers, loss stays in target units above `delta` | needs `delta` chosen; below `delta` it inherits MSE's blur incentive (see AMSE) |
| `mse` | squared residual | smooth everywhere; strong gradient on large errors | outlier-sensitive; strongest amplitude-damping (blur) incentive |
| `mae` | absolute residual | fully outlier-robust; constant gradient | non-smooth at zero; weak signal on small errors |
| `rmse` | `sqrt(MSE + eps)` | interpretable in target units | same optimum as MSE; gradient rescaled by the running loss value |
| `smooth_l1` | Huber variant with `beta` transition | as Huber; PyTorch-native parameterisation | as Huber |
| `amse` (default) | spectral anti-blur loss (Subich et al. 2025, arXiv:2501.19374, flat-grid adaptation) | removes the "double penalty": matching the target's power spectrum per scale band is optimal at any coherence, so partially-predictable fine scales are no longer rewarded for being damped | more expensive than pointwise losses (FFT per step); `hybrid` mode introduces `spectral_weight` to calibrate |

## Why an anti-blur loss exists (the double penalty, in two sentences)

For any squared-error-type loss, the loss-optimal amplitude of a spatial scale the
model cannot predict perfectly equals its coherence with the target — at 50 %
coherence the optimal move is to halve that scale's amplitude. Spatial blur is
therefore the *optimum* of pointwise training, not a failure of it; AMSE modifies the
per-scale decomposition so that preserving the target's spectrum is optimal instead
(full derivation in the header of `icenet_mp/losses/amse_loss.py`).

### AMSE options

* `loss.mode=hybrid` (default): standard Huber plus `spectral_weight` × the AMSE
  excess. At `spectral_weight=0` this is exactly the Huber control. Calibrate
  `spectral_weight` so the spectral term carries ~10 % of the total loss at
  initialisation (the untrained model's Huber/excess ratio differs per architecture).
* `loss.mode=pure`: the AMSE objective alone (no `spectral_weight`).
* `loss.wavenumber_weight=fastnet` (default off): re-weights the spectral penalty
  toward fine scales by `max(N_k * k^sqrt(3), 1)` (FastNet, arXiv:2509.17601) at
  unchanged total, moving the penalty onto ice-edge scales.
* `loss.static_ref_path=<path.npy>` (default off): subtracts a static reference field
  from both prediction and target before the spectral terms, so static
  climatological structure does not dilute the anti-blur signal.

Both optional flags are bit-for-bit inert when off.

## Weighting by lead time

Every loss YAML has a `lead_time_exponent` key, which defaults to `null` (no lead-time weighting).
Setting `loss.lead_time_exponent=X` makes forecast days further into the future contribute more.
The loss is evaluated separately at each lead time and the per-day values are combined with weights `w_t = (t + 1) ** lead_time_exponent`.
These are rescaled to have mean 1 so the overall magnitude of the loss is comparable to unweighted runs.

```bash
imp train --config-name <config> loss.lead_time_exponent=1             # default loss, linear scaling
imp train --config-name <config> loss=huber loss.lead_time_exponent=2  # Huber, quadratic scaling
```

- `lead_time_exponent=0` gives uniform weighting, which is identical to the unweighted loss for
  mean-reduced pointwise losses (`mse`, `mae`, `huber`, `smooth_l1`). For losses that reduce
  non-linearly over the whole tensor (`rmse`, `amse`) it is a per-day average instead, so it is
  close to, but not exactly, the unweighted loss.
- For autoregressive DDPM, training optimises one forecast step at a time, so lead-time weighting
  has no effect on the training loss (a warning is logged when the model is built); validation and
  test losses are computed over the full rollout and are weighted. For parallel DDPM, the weights
  apply per lead time to the v-prediction loss rather than to SIC error.
- In multistage training, the encoder and decoder stages train on a single time step, so
  lead-time weighting has no effect on them (a warning is logged). It applies in the processor
  stage, which forecasts the full set of lead times.
