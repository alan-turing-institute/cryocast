# Forecast rollout

An `EncodeProcessDecode` model makes a forecast for several lead times from one window of observations.
This process is the rollout.
Three options in the model config control the rollout:

| Option | Values | Default | Effect |
|--------|--------|---------|--------|
| `rollout_space` | `latent`, `physical` | `latent` | Where the forecast loop runs |
| `skip_anchor` | `initial`, `previous` | `initial` for `latent`, `previous` for `physical` | The frame that the skip connection adds to each lead time |
| `relaxation_tau` | a positive number of lead times, or `null` | `null` (off) | Relax each lead time towards climatology |

All three options are top-level keys of the model config, next to `encoders`, `processor`, `decoder` and `finaliser`.
If you do not set `skip_anchor` and `relaxation_tau`, the model uses the defaults in the table.
Checkpoints that do not record these options use the same defaults.

## Rollout space

The rollout space sets where the model advances the forecast from one lead time to the next.

| Value | Each forecast | Requirements |
|-------|---------------|--------------|
| `latent` | The encoders encode the input window once. The processor makes all lead times in latent space. The decoder decodes all lead times together. | None |
| `physical` | For each lead time, the encoders encode the window, the processor makes one step, and the decoder decodes it. The forecast goes into the window for the next lead time. | The target dataset must also be a model input. The processor must implement `forward()` for one step. |

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

!!! note
    `skip_anchor` has no effect if the finaliser has no skip connection (`skip_connection: none`).
    In this case, the model writes a warning to the log.

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

### Relaxation for a trained model

Relaxation has no weights, so you can add it to a trained model, change `tau`, or turn it off when you load a checkpoint.
The `model.relaxation_tau` value in the current config replaces the value in the checkpoint:

```bash
uv run cryocast evaluate --checkpoint PATH_TO_A_CHECKPOINT +model.relaxation_tau=30
```

To turn relaxation off, set `++model.relaxation_tau=null`.
The change also applies when you resume training from the checkpoint.

`skip_anchor` always comes from the checkpoint, because it changes the inputs that the model learned from.
Thus you can add relaxation only to a model that was trained with `skip_anchor: previous`.
This is the default for a physical rollout.
For a latent rollout, train the model with `skip_anchor: previous` first.

## When to use each option

| Situation | Recommended settings |
|-----------|---------------------|
| You must reproduce an existing checkpoint or leaderboard result | Do not set `skip_anchor` or `relaxation_tau` |
| Multistage model with a pre-trained decoder | `skip_anchor: previous`, so the forecast uses the decoder as the decoder stage trained it |
| Forecast for lead times longer than the training lead times, or a forecast bias that increases with lead time | `skip_anchor: previous` and `relaxation_tau` |
| Physical rollout | Keep the default anchor (`previous`). Add `relaxation_tau` for long lead times. |
| Ablation of the anchor in a physical rollout | `skip_anchor: initial`, without `relaxation_tau` |

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
