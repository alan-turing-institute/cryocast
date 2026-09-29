# Run an evaluation job

This guide walks through running an evaluation on a trained checkpoint: launching the job, enabling visualisations, saving predictions, and finding the outputs.

## Prerequisites

Make sure IceNet-MP is [installed](../user-guide/installation.md) and that you have a trained checkpoint — either from a [training run](train.md) or downloaded from shared storage.

## 1. Get a checkpoint

### From a training run

Checkpoints are saved to `${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints/<name>.ckpt` after training. Pick the checkpoint you want to evaluate.

### From shared storage (HPC)

Pre-trained checkpoints are available on Baskerville, DAWN, and Isambard-AI.
Ask a team member for the path.

## 2. Create a local config

If you do not already have a local config from a training run, create one at `icenet_mp/config/<your-name>.local.yaml`.
See [Train a model — Create a local config](train.md#2-create-a-local-config) for details.

## 3. Run evaluate

See the [`evaluate` command reference](../user-guide/commands.md#evaluate) for full option details, then run:

```bash
uv run imp evaluate --config-name <your-name>.local --checkpoint PATH_TO_CHECKPOINT
```

### Enabling visualisations

By default, all visualisations are enabled (see `icenet_mp/config/evaluate/callbacks/plotting.yaml`). To disable forecast plots, set `make_static_plots` and `make_video_plots` to `false` in your local config:

```yaml
evaluate:
  callbacks:
    plotting:
      make_static_plots: false
      make_video_plots: false
```

Plots of the raw input data are also enabled by default. To disable them, set:

```yaml
evaluate:
  callbacks:
    plotting:
      make_input_plots: false
```

### Saving predictions as NetCDF

Pass `--save-predictions` to write the model output from the configured test period to a NetCDF file:

```bash
uv run imp evaluate \
  --config-name <your-name>.local \
  --checkpoint PATH_TO_CHECKPOINT \
  --save-predictions
```

The file is written incrementally during evaluation, so the full test period does not need to be held in memory. It is saved as `predictions.nc` inside the run's directory (alongside checkpoints and other run artifacts, and under W&B's `files/` in the run's dashboard when using W&B). It contains `forecast_reference_time`, `lead_time`, `valid_time`, latitude/longitude coordinates, and one data variable per prediction target, together with the matching ground truth under an `_observed` suffix. Sea-ice concentration is exported as `ice_conc` (prediction) and `ice_conc_observed` (ground truth) in its original source scale with CF `sea_ice_area_fraction` metadata.

If the target dataset has generated masks, the static `land_mask` and `active_mask` variables are included on the `(y, x)` grid (1 = ocean/active, 0 = land/inactive) and linked to the data variables via `ancillary_variables`. Masked models output zero over land, so use these masks to distinguish land from open water.

Prediction export uses the existing `data.split.test` date ranges. To export a smaller date range, change the test split in the config rather than running a separate prediction pass.

NetCDF export currently supports single-process evaluation. If the evaluation config uses multiple devices, set `evaluate.trainer.devices=1` for the export run.

## 4. Check results in W&B

Once evaluation completes, the run appears in the W&B project `evaluate` under the `turing-seaice` entity at [wandb.ai](https://wandb.ai).

| Key | Contents |
|-----|----------|
| `output_static` | Static images of forecast output. |
| `output_video` | Animated forecast output. |
| `input_static` | Static images of the raw input data (if `make_input_plots: true`). |
| `input_video` | Animated raw input data (if `make_input_plots: true`). |
| `Custom Charts` | Per-forecast-day metrics, allowing skill to be assessed at longer lead times. |

### How to interpret forecast days and example videos

The per-forecast-day charts and the example videos summarise different dimensions of the same evaluation run.

For each valid start date in the configured evaluation period, the model produces predictions for each of the next `n_forecast_steps` timesteps.
The value of each metric is averaged across all start dates to give a single value.
In other words, the value at, for example, forecast day 3, tells you the average value of that metric at 3 days into the future across all start dates in the evaluation period.

The example videos use a small subset of representative start dates spread across the evaluation period.
Each video contains the full forecast sequence for the selected start date.
For example, when running with `n_forecast_steps=7`, each video will have 7 frames, one for each forecast day.

- `Custom Charts` show performance as a function of forecast day, aggregated across the evaluation period.
- `output_video` shows a small number of representative forecast sequences from within that period.

This distinction is useful when checking whether skill degrades with lead time without generating a video for every possible forecast start date.
