# Configuration

## Your local config file

Create a file in `cryocast/config` named `<chosen-name>.local.yaml`.
Local config files should inherit from `base.yaml` and override only what you need:

```yaml
defaults:
  - base
  - _self_

base_path: /local/path/to/my/data
```

Run any command with your config using:

```bash
uv run cryocast <command> --config-name <your local config>.local
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
uv run cryocast <command> ++base_path=/local/path/to/my/data
```

!!! warning
    `baseline/00_persistence.yaml` overrides the options in `base.yaml` needed to run the `persistence` model. `baseline/00_climatology.yaml` does the same for the `climatology` baseline model, which predicts the calendar-day-mean (climatology) field for each forecast step.

## HPC systems

For shared HPC systems (Baskerville, DAWN, Isambard-AI, or JASMIN), add the matching `platform` override, which sets the pre-downloaded data path and the right GPU accelerator:

```bash
uv run cryocast <command> --config-name <your local config>.local platform=isambardai data=full_north  # or platform=baskerville, platform=dawn, or platform=jasmin
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
uv run cryocast train --config-name my_local_config
```

### Selecting input and target variables

The set of variables used as input and target are controlled by the `variables` config key.
For each input dataset group (as defined by the `group_as` key in `datasets`) you should define the set of variables from that dataset that you want to use.
The same logic applies to the target variables, as shown in this example

```yaml
variables:
  input:
    era5:
      - 2t
      - msl
    sic-osisaf:
      - ice_conc
  target:
    sic-osisaf:
      - ice_conc
```

!!! note
    We currently enforce that the target variables must also be input variables.
    This requirement may be relaxed in future.

An empty list of target variables (e.g. `sic-osisaf: []`) selects every input variable in that dataset group.

### CARRA2

CryoCast includes a `carra2` ingestion source backed by the Copernicus Climate Data Store. The source submits one `reanalysis-pan-carra` request for each exact date/time requested by Anemoi, avoiding accidental Cartesian combinations of days and times.

A sample native-grid recipe is available as `samp_weathernorth_carra2_2p5km_2020_2024_24h_v1`. It requests a small set of surface weather variables at 12:00 UTC for the sample date ranges. Custom recipes can select other CARRA2 variables, level types, level locations, or geographic subsets. The source currently uses GRIB output so downloaded fields can be materialised safely before temporary files are removed:

```yaml
input:
  carra2:
    level_type: pressure_levels
    level_location: ["850"]
    variables: [temperature]
    product_type: analysis
    data_format: grib
```

Before downloading CARRA2, configure CDS API credentials and accept the dataset terms in the Climate Data Store. The recipe keeps CARRA2 on its native grid; normal CryoCast encoders can then map it into the model's shared latent space, or a `reproject` filter can be added to a custom recipe.

### Generating Argo float missing dates

Some dates have no Argo float data. When specifying a new Argo float dataset for the first time it is necessary to generate a list of missing dates for a dataset. This can be done as follows:

1. Add `ignore_missing_dates: true` to the relevant dataset file.
2. Delete any previously downloaded version of the dataset.
3. Run:

```bash
uv run cryocast datasets create --config-name <config that requires this dataset>
```

This downloads the full dataset, skipping exceptions from missing dates, and prints the missing dates at the end of each data group.
