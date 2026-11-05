# CryoCast - TRL2 - 2026-05-07

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
Regular releases, most recently July 2026.

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
CryoCast is a multimodal forecasting system for Arctic and Antarctic sea ice, developed at the Alan Turing Institute. Available inputs include satellite observations, atmospheric reanalysis data and *in situ* ocean measurements for short-term sea ice concentration (SIC) forecasting. The selected data and model configuration determines which inputs are used. In the modular encode-process-decode models, dataset-specific encoders map inputs to a common latent grid, a processor forecasts from the combined representation, and a decoder maps the result to the target grid. Standalone models and baselines are also available. Configurations include a lightweight setup for quick tests, a CNN encoder-decoder with a UNet processor, a vision transformer, diffusion models and a persistence baseline.

### Intended Uses
This model is intended for research and exploratory inference using configured climate and observational inputs to generate SIC forecasts. The pipeline supports training, fine-tuning and evaluation at multiple forecast lead times. Typical uses include comparisons with persistence, sensitivity experiments that vary inputs or model settings, and case-study analysis. The presence of a model configuration or an illustrative forecast does not establish its accuracy for a particular region, season or lead time.

### Out-of-Scope Uses
This stage of the codebase is not intended for operational forecasting, safety-critical decision making or issuing public warnings.

### Time Lag
- Short-term / near-term forecasting
- With particular focus on critical zones such as the **sea ice edge** and **marginal ice zone**

### Spatial Domain
Arctic and Antarctic

### Temporal / Seasonal Domain
Current full-data configurations use OSI SAF L3/L4 reprocessed SIC at 25 km resolution and ERA5 reanalysis, with configured records spanning 1979–2025. The wider satellite archive extends to 1978, but these combined datasets start in 1979. Argo configurations cover records from 1999 onward and are being used in ongoing model training. Usable coverage varies with missing dates and sensor availability; these archive ranges do not establish forecasting skill across every season and region.

### Approaches to Uncertainty & Variability
Repeated DDPM sampling with fixed trained weights can generate different SIC fields through stochastic sampling. The resulting spread does not by itself quantify uncertainty in the learned model parameters (epistemic uncertainty) or establish calibrated forecast probabilities. No uncertainty-calibration assessment is reported in this card.

## Training, Testing, Validation Datasets and Procedures

### Input Information
- **Data sources:** OSI SAF L3/L4-derived SIC products, ERA5 atmospheric reanalysis, and Argo float observations, processed and normalised using Anemoi dataset tooling. Argo inputs are being used in ongoing training, and their contribution is not yet fully evaluated.
- **Data separation:** The shipped [full-data split](https://github.com/alan-turing-institute/cryocast/blob/main/cryocast/config/data/split/full_dataset.yaml) assigns records through 2021 to training, 2022–2023 to validation, 2024 to testing and 2025 to a designated holdout. [Sample configurations](https://github.com/alan-turing-institute/cryocast/blob/main/cryocast/config/data/split/sample_dataset.yaml) use narrower periods. Each reported experiment must document its actual splits, preprocessing and model-selection history before claiming an untouched evaluation set.
- **Missing data:** Dataset recipes declare missing dates and apply dataset-specific filling or masking. Anemoi inspection checks dataset readiness; it does not establish observational accuracy or forecast uncertainty.

Detailed input data table: pending project-team completion.

### Output Sea Ice Variables
SIC is the fraction of each grid cell covered by ice. Current OSI SAF recipes convert source percentages to fractions, and the exported NetCDF `ice_conc` variable uses dimensionless units (`1`): 0 represents open water and 1 represents full ice cover, equivalent to 0% and 100%. Export reverses model normalisation to the stored dataset values.

### Output Information
Predictions can be saved in NetCDF format during evaluation using `cryocast evaluate --save-predictions`. See the [evaluation guide](https://github.com/alan-turing-institute/cryocast/blob/main/docs/src/how-to/evaluate.md) for the required checkpoint, configuration and single-process export settings.

## Evaluation Details

### Evaluation Statement
- Configured metrics include `sieerror (mean)`, `rmse (mean)` and `mae (mean)`, with per-forecast-day results also available. Ice presence uses SIC greater than 0.15 (15%).
- `sieerror` is signed predicted-minus-observed extent in km²; positive and negative errors can cancel. Interpret it alongside concentration and spatial-error metrics, using matched dates, masks, grid resolution and forecast lead times.
- Model comparison: UNet and persistence configurations are available; reviewed comparative results remain to be reported.
- Key findings: No comparative forecast-skill conclusions are reported in this card yet. The project team will add results once the benchmark evidence has been reviewed.

### Rationale for Current TRL
The CryoCast system has made substantial progress through TRL 2, with optimised code, unit tests, initial curated datasets for *in situ* data, and a basic software architecture established. Notably, several TRL 3 coding checkpoints have already been completed, including modular/reusable code structures and integration tests for module robustness. To complete TRL 2, preliminary benchmarking and evaluation results will be written up for publication.

### On-going Progress Towards Next TRL
To complete TRL 2, there are two outstanding requirements: formal documentation of baseline model performance metrics and fully validated focused experiments confirming model behaviours and goals. For progress towards TRL 3, the sea ice team is actively looking for partners to narrow initial use cases for the CryoCast pipeline and develop tests that are grounded in real-world use.

### Ethical Considerations
Publicly accessible inputs do not remove the need to assess downstream impacts. Users should respect dataset terms, communicate evaluation limitations and avoid relying on these research outputs for the operational or safety-critical uses excluded above.
