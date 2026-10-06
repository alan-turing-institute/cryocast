# CryoCast - TRL 2 - 2026-05-07

## Versioning and Reference Information

### Card Authors
Sophie Arana, Isabel Fenton, Maria Novitasari, Erin Quan, James Robinson, Shaerdan Shataer, Louisa van Zeeland

### Model/System Name
CryoCast

### Version
Release 2026.07

### TRL
TRL 2

### Model/System Release Date
Pending project-team confirmation.

### TRL Date
2026-05-07 (draft)

### TRL Card Date
2026-05-07 (draft)

### Lead
Louisa van Zeeland

### Contact Information
Louisa van Zeeland (lvanzeeland@turing.ac.uk) or seaice@turing.ac.uk (this sends an email to all team members)

### Access to Products
Code for the forecasting pipeline is open source and available on [GitHub](https://github.com/alan-turing-institute/cryocast).

There are no live forecast products being published alongside the code at this stage.

### Licence
MIT Licensed codebase

### References
Supporting technical documentation is available in the repository:

- [Data documentation](https://github.com/alan-turing-institute/cryocast/blob/main/docs/src/user-guide/data.md)
- [Metrics documentation](https://github.com/alan-turing-institute/cryocast/blob/main/docs/src/user-guide/metrics.md)

### Citation
The Alan Turing Institute. (2026). CryoCast [Source code]. GitHub. https://github.com/alan-turing-institute/cryocast

## Model/System Details

### Description
CryoCast is a multimodal forecasting system for Arctic and Antarctic sea ice, developed at the Alan Turing Institute. CryoCast integrates satellite observations, reanalysis data, and point-based in-situ sensor data to generate short-term sea-ice concentration forecasts. It is built around an encode-process-decode architecture, where dataset-specific encoders project each input into a shared latent space, a central processor operates on the combined representation, and output-specific decoders map back to the target resolution. This design makes it easy to add new input sources or prediction targets without restructuring the whole model. The codebase supports several model configurations, including a lightweight default setup suitable for quick tests, a CNN-based encoder-decoder variant wrapping a UNet processor, a vision transformer, a diffusion model variant and a persistence baseline for benchmarking.

### Intended Uses
This model is intended for research and exploratory inference using historical or real-time climate and observational inputs to generate sea-ice concentration forecasts. Model performance is evaluated across multiple forecast lead times, with training focused on short-range horizons. Training, fine-tuning, and evaluation are all supported through the provided pipeline. Typical direct uses include benchmarking against persistence and dynamical model baselines, sensitivity experiments such as varying input variables or data sources, and case-study analysis of notable sea-ice events.

### Out-of-Scope Uses
This stage of the codebase is not intended for operational forecasting, safety-critical decision making or issuing public warnings.

### Time Lag
- Short-term / near-term forecasting
- With particular focus on critical zones such as the **sea ice edge** and **marginal ice zone**

### Spatial Domain
Arctic and Antarctic

### Temporal / Seasonal Domain
Current full-resolution configurations use OSI SAF sea-ice concentration and ERA5 reanalysis data spanning 1979–2025, with Argo float observations available from 1999 onward.

### Approaches to Uncertainty & Variability
Currently no default uncertainty quantification. The DDPM architecture supports epistemic uncertainty estimation through multiple sampling runs, producing a spread of plausible SIC forecasts rather than a single deterministic prediction.

## Training, Testing, Validation Datasets and Procedures

### Input Information
- **Data sources:** OSI SAF sea-ice concentration, ERA5 atmospheric reanalysis, and Argo float observations, processed and normalised using Anemoi dataset tooling
- **Data leakage:** Strict temporal separation enforced, with test years held out exclusively for benchmarking and never exposed during development
- **Input uncertainty:** Missing data identified and handled via Anemoi's built-in inspection tooling

Detailed input data table: pending project-team completion.

### Output Sea Ice Variables
Sea-ice concentration (SIC) as percentage coverage per grid cell, with 0% representing open water and 100% representing full ice cover.

### Output Information
Predictions are used during evaluation and can optionally be exported as denormalised NetCDF using `cryocast evaluate --save-predictions`. NetCDF export currently requires single-process evaluation.

## Evaluation Details

### Evaluation Statement
- Evaluation metrics: `sieerror (mean)`, `rmse (mean)`, `mae (mean)`, with a 15% sea-ice threshold used for ice-edge classification
- Model comparison: simple UNet, persistence
- Key findings: pending project-team completion

### Rationale for Current TRL
The CryoCast system has made substantial progress through TRL 2, with optimised code, unit tests, initial curated datasets for in-situ data, and a basic software architecture established. Notably, several TRL 3 coding checkpoints have already been completed, including modular/reusable code structures and integration tests for module robustness. To complete TRL 2, preliminary benchmarking and evaluation results will be written up for publication.

### On-going Progress Towards Next TRL
To complete TRL 2, there are two outstanding requirements: formal documentation of baseline model performance metrics and fully validated focused experiments confirming model behaviours and goals. For progress towards TRL 3, the sea-ice team is actively looking for partners to narrow initial use cases for the CryoCast pipeline and develop tests that are grounded in real-world use.

### Ethical Considerations
CryoCast is a research product and currently operates on publicly accessible data so there are no known ethical considerations.
