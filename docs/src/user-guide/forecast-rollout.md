# Forecast rollout

An `EncodeProcessDecode` model makes a forecast for several lead times from one window of observations.
This process is the rollout.
Three options in the model config control the rollout:

| Option | Values | Default | Effect |
|--------|--------|---------|--------|
| `rollout_space` | `latent`, `physical` | `latent` | Where the forecast loop runs |
| `skip_anchor` | `initial`, `previous` | `initial` for `latent`, `previous` for `physical` | The frame that the skip connection adds to each lead time. A `physical` rollout accepts only `previous`. |
| `relaxation_tau` | a positive number of lead times, or `null` | `null` (off) | Relax each lead time towards climatology |

All three options are top-level keys of the model config, next to `encoders`, `processor`, `decoder` and `finaliser`.
If you do not set `skip_anchor` and `relaxation_tau`, the model uses the defaults in the table.
Checkpoints that do not record these options use the same defaults.

## Rollout space

The rollout space sets where the model advances the forecast from one lead time to the next.

| Value | Each forecast | Requirements |
|-------|---------------|--------------|
| `latent` | The encoders encode the input window once. The processor makes all lead times in latent space. The decoder decodes all lead times together. | None |
| `physical` | For each lead time, the encoders encode the window, the processor makes one step, and the decoder decodes it. The forecast goes into the window for the next lead time. | The target dataset must also be a model input. The processor must implement `forward()` for one step. `skip_anchor` must be `previous`. |

A physical rollout encodes the window once for each lead time, so it is slower than a latent rollout.
In a physical rollout, the inputs that are not the target dataset keep their last observed frame for all lead times.

The `cnn_vit_cnn_residual` model config uses a physical rollout:

```yaml
rollout_space: physical
```

## Skip anchor

The `skip_connection` of the `finaliser` combines the decoder output with an anchor frame.
With an `additive` skip connection, the decoder output is a change from the anchor:

```
x_k = clamp(anchor_k + change_k, 0, 1)
```

Here, `x_k` is the forecast for lead time `k`, and `change_k` is the decoder output after the range restriction.
For an `additive` skip connection, the range restriction lets `change_k` be negative.
The `skip_anchor` option sets the anchor for each lead time:

| Value | Anchor for lead time 1 | Anchor for lead time `k > 1` | The decoder output is |
|-------|------------------------|------------------------------|-----------------------|
| `initial` | The last observed frame | The last observed frame | The total change since the last observation |
| `previous` | The last observed frame | The forecast for lead time `k - 1` | The change over one lead time |

With `previous`, the anchor changes at each lead time, so the changes add together over the rollout.
If the decoder predicts no change, both values give persistence.

### Why `previous` can help

With `initial`, the decoder must predict a larger change at each longer lead time.
With `previous`, the decoder predicts the change over one lead time at all lead times.
This is the same task for each lead time, so it is easier to learn.

In [multistage training](../how-to/train-multistage.md), the decoder stage trains the decoder with the frame from the day before the target as its anchor.
Thus the decoder learns the change over one day.
A latent rollout with `initial` then uses the same decoder with the last observed frame as its anchor for every lead time.
With `previous`, the forecast uses the decoder in the same way as the decoder stage trained it.

A physical rollout always uses `previous`.
If you set `skip_anchor: initial` with `rollout_space: physical`, the model raises a `ValueError` when you make it.

!!! note
    `skip_anchor` has no effect if the finaliser has no skip connection (`skip_connection: none`).

With `previous` in a latent rollout, the decoder still decodes all lead times together.
Only the finaliser runs once for each lead time, so the extra cost is small.

## Climatology relaxation

A forecast that goes beyond the lead times used in training can move far from realistic values.
Climatology relaxation moves each lead time a small fraction of the way towards climatology:

```
x_k ← x_k + λ · (c_k − x_k),  with  λ = 1 − exp(−1 / tau)
```

| Symbol | Meaning |
|--------|---------|
| `x_k` | The forecast for lead time `k`, after the skip connection |
| `c_k` | The climatology for the date of lead time `k` |
| `tau` | The e-folding time of the anomaly from climatology, in lead times |
| `λ` | The fraction of the anomaly that the model removes at each lead time |

Each relaxed value is between the forecast and the climatology.
Thus it stays in the valid range.
The finaliser applies the steps in this sequence: range restriction, skip connection, relaxation, mask.
Thus masked cells are always exactly 0.

### Damped anomaly persistence

The anomaly is the difference between the forecast and climatology.
If the decoder predicts no change and climatology is constant, the anomaly decreases exponentially:

```
x_k = c + exp(−k / tau) · (x_0 − c)
```

Here, `x_0` is the last observed frame.
This forecast is damped anomaly persistence, a standard reference forecast for long lead times.
A trained model adds its own change at each lead time, and relaxation removes a fraction of the anomaly.

The table shows the fraction of the initial anomaly that stays after `k` daily lead times, if the decoder predicts no change:

| `tau` | `λ` | `k = 7` | `k = 14` | `k = 30` | `k = 90` |
|-------|-----|---------|----------|----------|----------|
| 10 | 0.095 | 0.50 | 0.25 | 0.05 | 0.00 |
| 30 | 0.033 | 0.79 | 0.63 | 0.37 | 0.05 |
| 90 | 0.011 | 0.93 | 0.86 | 0.72 | 0.37 |

A small `tau` gives a strong pull towards climatology.
A large `tau` gives a weak pull, and the effect is small at short lead times.

### Requirements

- Relaxation needs `skip_anchor: previous`.
  With an `initial` anchor, the relaxation of one lead time does not carry over to the next lead time.
  The model raises a `ValueError` when you make it.
- In a latent rollout, relaxation needs a finaliser skip connection.
  Without a skip connection, the relaxation of one lead time does not carry over to the next lead time.
  The model raises a `ValueError` when you make it.
  A physical rollout re-encodes each relaxed lead time, so it does not need a skip connection.
- Relaxation needs a `climatology` entry in each batch.
  The data module makes the climatology only from the training period of the target dataset.
  Thus the climatology does not contain information from the forecast period.
  If the batch has no climatology, the forward pass raises a `ValueError`.

## When to use each option

| Situation | Recommended settings |
|-----------|---------------------|
| You must reproduce an existing checkpoint or leaderboard result | Do not set `skip_anchor` or `relaxation_tau` |
| Multistage model with a pre-trained decoder | `skip_anchor: previous`, so the forecast uses the decoder as the decoder stage trained it |
| Forecast for lead times longer than the training lead times, or a forecast bias that increases with lead time | `skip_anchor: previous` and `relaxation_tau` |
| Physical rollout | Keep the default anchor (`previous`). Add `relaxation_tau` for long lead times. |

To select `tau`, compare values on the validation split (2022–2023).

!!! warning
    Do not select `tau` with the test year (2024).
    Do not use 2025 data. This year is the embargoed holdout.

Compare a new setting with the same model without it, on RMSE, IIEE and signed SIE error at each lead time.
See [Metrics](metrics.md) for these metrics.

## Configure the options

The model configs do not set these options.
To enable them, add them to the `model` section of your `*.local.yaml` config:

```yaml
skip_anchor: previous
relaxation_tau: 30.0
```

You can also set them on the command line.
The keys are not in the model configs, so put a `+` before each key:

```bash
uv run cryocast train --config-name <your-config>.local +model.skip_anchor=previous +model.relaxation_tau=30
```

!!! warning
    Do not omit the `+`. Without it, Hydra cannot add the key and stops with a `ConfigCompositionException`.

To turn relaxation off, remove the `relaxation_tau` key or set it to `null`.

## Processors with a fixed horizon

Most processors make one lead time at a time, and use it as the input for the next lead time.
You can run them for more lead times than in training.

Two processors make all lead times in one step:

- `GSTAProcessor` (the `cnn_gsta_cnn` and `dc_gsta_dc` model configs).
  The last gated attention block has `n_forecast_steps × latent channels` output channels.
- The diffusion processors with `use_autoregressive: false`.
  The UNet denoises all lead times together, as `n_forecast_steps × latent channels`.

These processors have a fixed horizon:

- The size of the last layer comes from `n_forecast_steps`.
  If you make the model with a different `n_forecast_steps`, the checkpoint does not load.
  Thus the model cannot forecast past the horizon that it was trained on.
- A longer horizon puts more lead times into the same layers.
  Increase the capacity with the horizon, for example `hid_channels` for `GSTAProcessor`.
  Do not copy the settings from a short-horizon run to a long-horizon run without a test.
- With `n_blocks: 1`, `GSTAProcessor` has one layer from the history to all lead times, and no hidden channels.

All diffusion model configs (`cnn_ddpm_cnn`, `cnn_ddim_cnn`, `dc_ddpm_dc`, `dc_ddim_dc`) set `use_autoregressive: true`.
Thus they do not have a fixed horizon.

## Memory use for long rollouts

The decoder decodes all lead times together, at the full output resolution.
For training, it keeps its activations for each lead time until the backward pass.
Thus decoder memory increases linearly with `n_forecast_steps`, and a long rollout can use all GPU memory.

To decrease decoder memory, set `decode_chunk_size` in the decoder config:

```bash
uv run cryocast train --config-name <your-config>.local +model.decoder.decode_chunk_size=10
```

The decoder then decodes this number of frames at a time.
In training, it does not keep its activations. It runs each chunk again in the backward pass.
The forecasts and gradients do not change, but each training step is slower.
A smaller chunk size uses less memory.

!!! note
    `decode_chunk_size` cannot be used with a decoder that has batch normalisation (`norm_type: batchnorm`).
    The first rollout then raises a `ValueError`.

A processor that predicts one lead time at a time also keeps its activations for each lead time.
This applies to `UNetProcessor`, `VitProcessor`, `NullProcessor` and `ConvLSTMProcessor`.
To recompute them in the backward pass in the same way, set `recompute_steps` in the processor config:

```bash
uv run cryocast train --config-name <your-config>.local +model.processor.recompute_steps=true
```

`GSTAProcessor` and the diffusion processors ignore `recompute_steps`.

!!! note
    `recompute_steps` cannot be used with a processor that has batch normalisation (`norm_type: batchnorm`).
    The first rollout then raises a `ValueError`.
    The `UNetProcessor` default is `batchnorm`, but the `UNetProcessor` model configs set `norm_type: groupnorm`.

### Physical rollout

A physical rollout does not use `decode_chunk_size` or the processor `recompute_steps`.
For each lead time, it encodes the input window, runs one processor step, decodes one frame and applies the finaliser.
For training, it keeps the activations of the encoders, the processor and the decoder for each lead time.
Thus a physical rollout uses the most memory for each lead time.

To recompute each lead time in the backward pass, set `recompute_steps` in the model config:

```bash
uv run cryocast train --config-name <your-config>.local +model.recompute_steps=true
```

The model then keeps only the input window of each lead time.
The forecasts and gradients do not change, but each training step is slower.

!!! note
    The model `recompute_steps` cannot be used with a latent rollout or with batch normalisation.
    The model then raises a `ValueError` when you create it.
    `CNNEncoder` and `CNNDecoder` use `batchnorm` by default.
    For `cnn_vit_cnn_residual`, set `norm_type: groupnorm` for each encoder in your config, for example:

    ```bash
    uv run cryocast train --config-name <your-config>.local +model.recompute_steps=true \
      +model.encoders.sic-osisaf.norm_type=groupnorm +model.encoders.era5.norm_type=groupnorm
    ```

    A model with `groupnorm` encoders cannot load a checkpoint from a model with `batchnorm` encoders.
