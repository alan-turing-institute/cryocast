# Configuration

## Your local config file

Create a file in `icenet_mp/config` named `<chosen-name>.local.yaml`.
Local config files should inherit from `base.yaml` and override only what you need:

```yaml
defaults:
  - base
  - _self_

base_path: /local/path/to/my/data
```

Run any command with your config using:

```bash
uv run imp <command> --config-name <your local config>.local
```

This uses the default model setup (rescaling encoder, small UNet, rescaling decoder), which is sufficient for quick tests but not for larger training runs.

### Overriding model parameters

To switch to a different named model config or override specific parameters:

```yaml
defaults:
  - base
  - override /model: cnn_unet_cnn
  - _self_

model:
  processor:
    start_out_channels: 32

base_path: /local/path/to/my/data
```

You can also override individual options at the command line without a config file:

```bash
uv run imp <command> ++base_path=/local/path/to/my/data
```

!!! warning
    `baseline/00_persistence.yaml` overrides the options in `base.yaml` needed to run the `persistence` model. `baseline/00_climatology.yaml` does the same for the `climatology` baseline model, which predicts the calendar-day-mean (climatology) field for each forecast step.

## HPC systems

For shared HPC systems (Baskerville, DAWN, Isambard-AI, or JASMIN), add the matching `platform` override, which sets the pre-downloaded data path and the right GPU accelerator:

```bash
uv run imp <command> --config-name <your local config>.local platform=isambardai data=full_north  # or platform=baskerville, platform=dawn, or platform=jasmin
```

## Datasets

### Selecting a dataset

The default dataset group is controlled by the `data` key, which defaults to `sample` in `base.yaml` (i.e. `data/sample.yaml`).
To understand how dataset properties are encoded in dataset names, see `data/datasets/naming_convention.txt`.
To define a custom set of datasets, create `data/my_datasets.local.yaml`:

```yaml
defaults:
  - datasets:
    - samp_sicsouth_osisaf_25p0km_2017_2019_24h_v2
    - samp_weathersouth_era5_0p5_2017_2019_24h_v2
  - split: sample_dataset
  - _self_
```

Then reference it from your main config:

```yaml
defaults:
  - <the base config file you are using>
  - override /data: my_datasets.local
  - _self_
```

And run with:

```bash
uv run imp train --config-name my_local_config
```

### Generating Argo float missing dates

Some dates have no Argo float data. When specifying a new Argo float dataset for the first time it is necessary to generate a list of missing dates for a dataset. This can be done as follows:

1. Add `ignore_missing_dates: true` to the relevant dataset file.
2. Delete any previously downloaded version of the dataset.
3. Run:

```bash
uv run imp datasets create --config-name <config that requires this dataset>
```

This downloads the full dataset, skipping exceptions from missing dates, and prints the missing dates at the end of each data group.
