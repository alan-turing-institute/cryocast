# Notebooks

The [`notebooks`](https://github.com/alan-turing-institute/cryocast/tree/main/notebooks)
directory contains seven supplementary examples and research artifacts for CryoCast.
They are not required to use the package. The scientific code, explanations,
illustrations referenced by Markdown, and teaching order are retained.

## Setup

Install the notebook dependencies from the repository root:

```bash
uv sync --group notebooks
cd notebooks
uv run --group notebooks jupyter lab
```

A number of notebooks need externally downloaded datasets, trained model
checkpoints, or credentials. The notebook-specific requirements below are
important: **not all notebooks are account-free**.

## Notebook inventory

| Notebook | Purpose | Prerequisites |
| --- | --- | --- |
| [`demo_pipeline.ipynb`](https://github.com/alan-turing-institute/cryocast/blob/main/notebooks/demo_pipeline.ipynb) | Full training/evaluation walkthrough, model flexibility, and configuration explanations | The current `demo_notebook` example accesses ERA5/OSISAF/Argo data and logs to W&B. Set up CDS and W&B credentials before running all cells. |
| [`metrics_examples.ipynb`](https://github.com/alan-turing-institute/cryocast/blob/main/notebooks/metrics_examples.ipynb) | Synthetic metric cases, ice-edge scores and masking comparisons | Package and notebook dependencies only. This is the account-free notebook executed in CI. |
| [`layer_diagnostics.ipynb`](https://github.com/alan-turing-institute/cryocast/blob/main/notebooks/layer_diagnostics.ipynb) | Investigate activation capture and model layers | A compatible local checkpoint and its training datasets. |
| [`argo_data.ipynb`](https://github.com/alan-turing-institute/cryocast/blob/main/notebooks/argo_data.ipynb) | Argo float downloading, inspection, gridding and plotting | Argo data access, network connectivity and geospatial dependencies. |
| [`case_study_whale_corridors.ipynb`](https://github.com/alan-turing-institute/cryocast/blob/main/notebooks/case_study_whale_corridors.ipynb) | Whale-corridor and shipping case-study visualisations | Case-study datasets, configuration and external geospatial resources. |
| [`extract_anomalies.ipynb`](https://github.com/alan-turing-institute/cryocast/blob/main/notebooks/extract_anomalies.ipynb) | Prepare gridded ERA5 pressure anomalies for a non-gridded experiment | CDS access and the optional `nongriddedenv.yaml` environment. |
| [`degrid_and_visualise.ipynb`](https://github.com/alan-turing-institute/cryocast/blob/main/notebooks/degrid_and_visualise.ipynb) | Sample generated anomalies into synthetic stations/buoys and visualise them | Outputs from `extract_anomalies.ipynb` and `nongriddedenv.yaml`. |

An account-free **CLI** alternative to the full `demo_pipeline.ipynb` is the
existing [`synthetic` configuration](https://github.com/alan-turing-institute/cryocast/blob/main/cryocast/config/synthetic.yaml),
which generates data locally and uses local-file logging. This is a separate
workflow; it does not make the CDS/W&B-dependent notebook account-free.

## CI and notebook maintenance

GitHub Actions validates the Jupyter format and saved-state cleanliness of **all
seven notebooks**. The standalone `metrics_examples.ipynb` is additionally
executed end-to-end with a temporary output file. The inline plotting backend
retains the example's figures, and a read-only check requires a PNG output for
every explicit `plt.show()` cell. Missing or out-of-order cell executions, error
outputs, and changed cell content fail the job. Other notebooks cannot be
fully executed in unattended public CI without their particular datasets,
checkpoints or credentials. The CI job does not silently skip their errors or
pretend to validate those external workflows.

After executing notebooks locally, run from the repository root:

```bash
python scripts/notebook_hygiene.py --clean
python scripts/notebook_hygiene.py
```

The cleanup removes only saved outputs, execution counts and execution
timestamps to avoid committing private data or machine-specific results.
The verifier flags empty cells, but **never deletes or modifies any notebook
cell content**. Already-clean notebook files are not rewritten, even when their
JSON formatting differs from the cleaner's output style. Existing plots remain in the scientific code and regenerate
when run. Keep the full explanatory walkthrough in `demo_pipeline.ipynb`; do
not replace it with a shortened series of CLI commands to satisfy CI.

When adding or renaming a notebook, update the inventory and assess whether it
can join the account-free execution job. Never introduce real credentials into
notebook sources, test data, CI variables or committed outputs.
