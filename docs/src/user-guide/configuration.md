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

### UNet processor normalisation

The production `cnn_unet_cnn` configuration keeps **BatchNorm** in the
`UNetProcessor`. GroupNorm remains available as an experiment:

```bash
uv run cryocast train --config-name baseline/cnn_unet_cnn \
  model.processor.norm_type=groupnorm
```

Issue #575 tested whether GroupNorm should replace BatchNorm because autoregressive
BatchNorm uses per-batch statistics during training but stored running statistics during
evaluation. The comparison used the same CNN-UNet-CNN setup, 14-day autoregressive
rollout, batch size 1, two seeds (575 and 576), 20 optimiser steps per run and five
held-out test batches. Both arms used ERA5 and OSI SAF SIC; Argo was omitted because
the available local legacy Argo Zarr could not be loaded by the current Anemoi version.
The two normalisation arms had the same 44,650,383 parameters.

This was deliberately a bounded architecture comparison rather than converged
production training. Neither learned arm beat persistence, so the absolute skill values
must not be interpreted as production forecast skill. They are useful for the matched
BatchNorm-versus-GroupNorm decision because every setting other than normalisation was
held fixed.

| Method | Mean RMSE | Leads 10-14 RMSE | Mean absolute SIE error (km²) | Leads 10-14 absolute SIE error (km²) |
| --- | ---: | ---: | ---: | ---: |
| BatchNorm | 0.1593 | 0.1460 | 14,567,147 | 14,830,550 |
| GroupNorm | 0.1688 | 0.1670 | 26,530,134 | 27,215,374 |
| Persistence | 0.0564 | 0.0760 | 1,115,134 | 1,800,375 |

Across the two seeds, BatchNorm's mean RMSE was 5.6% lower than GroupNorm's and its
late-lead RMSE was 12.5% lower. Mean and late-lead absolute SIE error were about 45%
lower with BatchNorm. GroupNorm was slightly better on RMSE at the earliest leads, but
that advantage disappeared by lead 6 and did not reduce late-rollout drift.

The per-lead cross-seed means were:

| Lead | BatchNorm RMSE | GroupNorm RMSE | Persistence RMSE | BatchNorm abs. SIE (km²) | GroupNorm abs. SIE (km²) | Persistence abs. SIE (km²) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1803 | 0.1717 | 0.0213 | 16,917,126 | 25,561,124 | 146,125 |
| 2 | 0.1745 | 0.1710 | 0.0304 | 13,890,064 | 25,693,874 | 278,875 |
| 3 | 0.1797 | 0.1705 | 0.0370 | 14,186,000 | 25,830,750 | 415,750 |
| 4 | 0.1795 | 0.1704 | 0.0425 | 13,913,938 | 25,981,750 | 566,750 |
| 5 | 0.1766 | 0.1699 | 0.0470 | 13,539,626 | 26,126,500 | 711,500 |
| 6 | 0.1595 | 0.1692 | 0.0516 | 14,077,187 | 26,290,250 | 875,250 |
| 7 | 0.1511 | 0.1687 | 0.0559 | 14,931,250 | 26,456,250 | 1,041,250 |
| 8 | 0.1499 | 0.1684 | 0.0598 | 14,401,062 | 26,618,250 | 1,203,250 |
| 9 | 0.1490 | 0.1681 | 0.0640 | 13,931,062 | 26,786,250 | 1,371,250 |
| 10 | 0.1479 | 0.1677 | 0.0686 | 13,576,688 | 26,951,500 | 1,536,500 |
| 11 | 0.1470 | 0.1674 | 0.0725 | 14,503,187 | 27,090,374 | 1,675,375 |
| 12 | 0.1461 | 0.1670 | 0.0761 | 15,142,249 | 27,220,998 | 1,806,000 |
| 13 | 0.1452 | 0.1667 | 0.0797 | 15,428,438 | 27,347,252 | 1,932,250 |
| 14 | 0.1440 | 0.1662 | 0.0831 | 15,502,186 | 27,466,748 | 2,051,750 |

**Decision:** keep BatchNorm as the default. The bounded comparison gives no evidence
that GroupNorm reduces autoregressive drift, while changing the default would also make
existing BatchNorm checkpoints incompatible with the default architecture. This does
not establish that BatchNorm is intrinsically superior under converged production
training; the decision should be revisited if a longer, multi-seed production training
study produces contrary evidence.

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

### Generating Argo float missing dates

Some dates have no Argo float data. When specifying a new Argo float dataset for the first time it is necessary to generate a list of missing dates for a dataset. This can be done as follows:

1. Add `ignore_missing_dates: true` to the relevant dataset file.
2. Delete any previously downloaded version of the dataset.
3. Run:

```bash
uv run cryocast datasets create --config-name <config that requires this dataset>
```

This downloads the full dataset, skipping exceptions from missing dates, and prints the missing dates at the end of each data group.
