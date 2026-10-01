# Notebooks

This directory contains exploratory, diagnostic, case-study, and model-reference
notebooks used alongside IceNet-MP. The notebooks are not part of the automated test
suite, so the supported command-line workflows and installation instructions in the
main documentation should be preferred when reproducing normal training or evaluation
runs.

## Notebook inventory

| Notebook | Purpose |
| --- | --- |
| `0_notebook_tf.ipynb` | TensorFlow IceNet reference workflow. |
| `1_icenet_forecast_unet.ipynb` | PyTorch UNet IceNet forecasting reference. |
| `2_icenet_forecast_cgan.ipynb` | PyTorch CGAN IceNet forecasting reference. |
| `ARGO_data.ipynb` | Exploration and visualisation of Argo data. |
| `case_study_whale_corridors.ipynb` | Whale-corridor downstream case study. |
| `degrid_and_visualise.ipynb` | De-gridding and visualisation exploration. |
| `demo_pipeline.ipynb` | End-to-end pipeline demonstration. |
| `extract_anomalies.ipynb` | Anomaly extraction workflow. |
| `layer_diagnostics.ipynb` | Model-layer and activation diagnostics. |
| `persistence.ipynb` | Persistence-baseline exploration. |

The first three notebooks are based on material combined from the following IceNet
repositories:

- [eds-book-gallery](https://github.com/eds-book-gallery/67a1e320-7c47-4ea9-8df8-e868326bc90b/tree/main)
- [icenet-notebooks](https://github.com/icenet-ai/icenet-notebooks)

They were assembled and adapted to run on Baskerville as reference implementations of
TensorFlow UNet, PyTorch UNet, and PyTorch CGAN workflows.

## Environments

The directory also contains notebook-specific environment files:

- `environment.yml`
- `environment_full.yml`
- `nongriddedenv.yaml`
- `seaice_env_min.yml`

For the current IceNet-MP package and CLI, use the project environment defined in
`pyproject.toml` and the installation instructions in
`docs/src/user-guide/installation.md`. Notebook-specific environments should only be
used when a notebook explicitly requires them.

## Maintenance

When adding, removing, or renaming a notebook, update this inventory so users can tell
what each file is for without opening large notebook files. If a notebook is no longer
useful, remove it in a focused PR rather than leaving an undocumented copy in this
directory.
