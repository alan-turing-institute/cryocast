# Submit forecasts to SIPN

This guide records the requirements for a first manual CryoCast submission to the Sea Ice Prediction Network (SIPN) or SIPN South. Submission protocols change between seasons, so check the current call before generating files.

## Requirements at a glance

| Programme | Forecast window | Main CryoCast-compatible outputs | Grid and metadata |
| --- | --- | --- | --- |
| 2026 Sea Ice Outlook | September 2026 | September mean extent; optionally daily sea-ice concentration or derived SIP/IFD/IAD fields | Extent follows the NSIDC >15% definition. The public full-field instructions do not prescribe one common grid or complete NetCDF schema. |
| SIPN South 2025-2026 | 1 December 2025 to 28 February 2026, 90 daily steps | Total area, regional area, and sea-ice concentration | No common grid is specified; concentration files include `longitude`, `latitude`, `sftof`, and `areacello`. |

The remaining 2026 SIO schedule lists the September deadline as **14 September 2026** and states that the September Outlook is based on May-August data. Do not use September observations in the September submission. The latest published SIPN South protocol is still the 2025-2026 call; update its dates and filenames when the 2026-2027 call is published.

## Check the forecast horizon

The tracked forecast-window configs in `cryocast/config/window/` currently provide 3-, 7-, 14-, and 21-step horizons; for daily SIC data these correspond to 3, 7, 14, and 21 days. A complete September forecast requires the whole month, and SIPN South requires 90 daily steps.

Before preparing a submission, agree how the selected checkpoint will cover the required window. Use a checkpoint/configuration trained and validated for that horizon, or document and validate an agreed rollout strategy. Do not silently reinitialise with observations after the call's information cutoff, because that changes the forecast being submitted.

## Choose the forecast run

Use the best current model agreed by the project and record enough information to reproduce the submitted forecast:

- checkpoint path or identifier;
- CryoCast git commit;
- inference configuration;
- forecast initialisation date and input-data cutoff;
- forecast horizon and any rollout/reinitialisation strategy;
- ensemble method and member count, if applicable;
- native grid, coordinate reference information, and land/ocean mask;
- units and missing-value convention for each field.

Keep the unmodified model output together with the final submission files.

## Arctic Sea Ice Outlook

The [2026 Sea Ice Outlook solicitation](https://www.seaiceprediction.org/sea-ice-outlook/august-2026-call-for-contributions) requests September monthly mean sea-ice extent and optionally full spatial forecast fields. It also solicits optional pan-Arctic extent-anomaly forecasts; those are outside this first manual workflow.

### Core Outlook submission

A first manual SIO contribution does not require a full-field upload. Use the submission form linked from the relevant monthly call for the core Outlook package. For the pan-Arctic contribution, prepare:

- the September monthly mean sea-ice extent forecast;
- the forecasting method/model description;
- uncertainty or probability information, where scientifically justified;
- a plain-language executive summary describing the Outlook, contributing factors, and methodology.

Pan-Antarctic and Alaska regional extent forecasts are optional additions. Full spatial fields use the separate SIPN data-server route described below. The solicitation states that a new Outlook must be submitted for each month; earlier monthly Outlooks are not carried forward automatically.

### Sea-ice extent

To match the NSIDC Sea Ice Index definition used by the call:

1. For each day, sum the areas of grid cells whose sea-ice concentration exceeds 15%.
2. Average the daily extent values over September to obtain the September monthly mean extent.

The call accepts pan-Arctic and pan-Antarctic extent forecasts and an Alaska regional forecast covering the Bering, Chukchi, and Beaufort seas. The Alaska request references the NSIDC 25 km polar-stereographic regional mask. If a different regional definition is used, document it in the submission.

### Full spatial fields

The call requests:

- Sea Ice Probability (SIP), the fraction of ensemble members with September sea-ice concentration above 15%;
- Ice-Free Date (IFD), the first date concentration falls below 15%, with an optional 80% threshold;
- Ice Advance Date (IAD) for August and September, reported as day of year, when freeze-up concentration first rises above 15%, with an optional 80% threshold;
- spatial initial-condition fields, particularly sea-ice concentration and sea-ice thickness, with the initialisation date recorded.

Daily sea-ice concentration fields can be submitted instead of pre-computed SIP, IFD, and IAD fields; the organisers state that they can derive those metrics from daily concentration. For a single deterministic forecast, retaining and submitting daily concentration also avoids presenting a one-member binary field as an ensemble probability.

The public 2026 call does not prescribe a common grid or complete NetCDF schema for full-field uploads. Preserve the model's geospatial metadata and confirm the current file-format and horizon expectations with the SIO team before upload. Existing contributors use the SIPN data server; new contributors can request access from the contact named in the current call.

## SIPN South

The latest published [SIPN South call](https://fmassonn.github.io/sipn-south.github.io/call-contributions/) is for the 2025-2026 season. It requests **90 daily timesteps from 1 December 2025 through 28 February 2026**.

SIPN South requests sea-ice **area**, while SIO requests sea-ice **extent**. Keep those calculations separate. Area is concentration-weighted ocean surface; extent counts qualifying grid cells at their full area. The published SIPN South call does not specify the SIO 15% extent threshold for its area diagnostic, so do not reuse that threshold automatically. Record the concentration treatment, ocean mask, and cell-area convention used to derive submitted area values.

The SIPN South call also accepts grid-cell thickness (`sivol`) and long forecasts as low-priority diagnostics. They are outside this first SIC-focused workflow: the tracked SIC configs do not produce sea-ice thickness, and their current horizons are much shorter than the six-month minimum requested for the long-forecast diagnostic.

### Circumpolar sea-ice area

Submit one text file containing one row of 90 comma-separated daily Antarctic sea-ice area values.

- Units are 10^6 km².
- Values use four decimal places, including trailing zeroes.
- The 2025-2026 filename is `<group-name>_<forecast-id>_20251201-20260228_total-area.txt`.
- Ensemble forecasts use one file per member and increment the three-digit forecast ID.

### Regional sea-ice area

Submit one text file containing 36 rows, each with 90 comma-separated daily values using the same units and precision as the circumpolar file.

Rows represent successive 10° longitude bins from `0° <= longitude < 10°` through `350° <= longitude < 360°`.

The 2025-2026 filename is `<group-name>_<forecast-id>_20251201-20260228_regional-area.txt`.

### Sea-ice concentration

Submit one NetCDF file with 90 daily timesteps. The protocol requires CMIP6-style conventions and these variables:

| Variable | Meaning | Units |
| --- | --- | --- |
| `siconc` | Sea-ice concentration | % |
| `longitude` | Grid-cell longitude | degrees east |
| `latitude` | Grid-cell latitude | degrees north |
| `sftof` | Ocean fraction of each grid cell | % |
| `areacello` | Grid-cell area | m² |

The 2025-2026 filename is `<group-name>_<forecast-id>_20251201-20260228_concentration.nc`.

The protocol does not specify a mandatory common spatial grid for concentration submissions. Keep the forecast on its documented contributor grid and include longitude, latitude, ocean fraction, and cell area. For ensembles, use one file per member and increment the forecast ID.

## Validate the files

Before submission, check that:

- the intended checkpoint, commit, and configuration are recorded;
- the first and last forecast dates match the current call;
- the forecast contains every required daily timestep without gaps or duplicate dates;
- the forecast does not use observations beyond the permitted information cutoff; for the September 2026 SIO, this means no September observations;
- concentration units match the protocol, especially `%` for SIPN South `siconc`;
- longitude, latitude, ocean fraction, and cell area align with the forecast grid;
- land and missing values are handled consistently;
- area or extent diagnostics reproduce from the archived spatial fields and recorded conventions;
- every ensemble member has a unique forecast identifier;
- filenames use the current season's required dates and naming convention.

Keep a submission record containing the checkpoint identifier, git commit, configuration, initialisation date, input-data cutoff, forecast horizon, generated filenames, and submission date.

## Submit manually

For the first implementation:

1. Generate the required forecast from the selected current model.
2. Convert the output into the current call's scalar and/or spatial formats.
3. Run the validation checklist above.
4. Archive the exact files being submitted in shared project storage.
5. Submit through the route specified by the current call.
6. Record the submission date and forecast identifiers.

For Arctic SIO full fields, use the SIPN data-server process described in the current call. For SIPN South, the latest protocol asks contributors to place the files in a URL-accessible archive, provide that URL through the submission form, and notify the organiser after submission.
