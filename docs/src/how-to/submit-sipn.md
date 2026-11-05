# Submit forecasts to SIPN

This guide supports a first **manual** CryoCast contribution to the Sea Ice Outlook (SIO) or SIPN South. It records the published formats and the work still needed between a CryoCast evaluation export and a valid submission. It does not select a checkpoint, establish seasonal forecast skill, or implement conversion or upload.

## Protocol versions and dates

**Checked on 23 October 2026.** Both dated calls below concern past submission rounds. Recheck the [SIO website](https://www.seaiceprediction.org/) and [SIPN South call page](https://fmassonn.github.io/sipn-south.github.io/call-contributions/) before preparing a new contribution.

| Published call | Target period | Deadline and status at this check |
| --- | --- | --- |
| [September 2026 SIO](https://www.seaiceprediction.org/sea-ice-outlook/september-2026-call-for-contributions) | September 2026 monthly mean extent | 14 September 2026, 18:00 AKDT, in the heading and schedule. Its linked submission form is closed. |
| [SIPN South 2025-2026](https://fmassonn.github.io/sipn-south.github.io/call-contributions/) | 1 December 2025 to 28 February 2026 inclusive, 90 daily values | 8 December 2025. The public call page still describes this season; a 2026-2027 protocol was not found there. |

The SIO page repeats an inconsistent August date in one paragraph; the September heading and schedule agree on 14 September. Do not treat either past deadline as an open invitation or infer a new deadline by changing the year. South filenames below are **historical examples**, to be replaced only from a confirmed new call.

The September SIO call specifies May-August information. September observations, later reanalysis revisions or later initialisations must not enter that forecast. A retrospective experiment must be labelled as such and reproduce the declared information cutoff; downloading historical inputs now does not demonstrate that they were available then.

## Check the forecast horizon

The tracked `cryocast/config/window/` presets contain 3, 7, 14 and 21 forecast steps. These are days only for daily input frequency. A complete September requires 30 target days, plus any lead-in from an earlier initialisation; the historical South call requires 90. Ice-free and ice-advance dates may require additional melt/freeze-season coverage.

Agree a checkpoint trained and evaluated for the required horizon, or a validated free-running rollout with an explicit conditioning strategy. Loading a checkpoint restores its saved model/window settings; simply overriding `window.n_forecast_steps` is not evidence of a valid longer forecast. Do not concatenate short forecasts reinitialised with later observations to simulate a single seasonal trajectory. This applies to every conditioning input, including ERA5 and Argo.

Record the checkpoint and configuration, code commit, initialisation time, information cutoff, ensemble method/seeds, output frequency, grid and mask provenance. Keep the original output alongside the converted files. Repeated diffusion samples do not by themselves establish calibrated probabilities.

## What CryoCast currently exports

The [evaluation guide](evaluate.md#saving-predictions-as-netcdf) describes the optional NetCDF writer already present in the code. For a retrospective evaluation with an existing local configuration and trusted checkpoint:

```bash
uv run cryocast evaluate \
  --config-name my.local \
  --checkpoint /path/to/run/checkpoints/model.ckpt \
  --save-predictions \
  train.trainer.devices=1
```

Replace the paths and configuration with the intended run and retain its `files/model_config.yaml`. The standard evaluation configuration inherits `train.trainer`, so this override keeps the complete trainer configuration and selects one device. For a custom evaluation trainer, verify its own single-process settings. The evaluation configuration must include the `prediction_writer` callback; the standard evaluation configuration does. See [configuration](../user-guide/configuration.md) and [commands](../user-guide/commands.md#evaluate) for setup.

The writer saves `files/predictions.nc` in the run directory. It uses `data.split.test` and writes matching observed targets, so this is an **evaluation export**, not a standalone future-forecast or SIPN submission command. Missing future observations cannot be replaced with invented truth to make this route work.

| Exported feature | Required submission check or conversion |
| --- | --- |
| `ice_conc` with dimensions `(forecast_reference_time, lead_time, y, x)` | Select a declared forecast origin/member and construct a continuous target-time series from `valid_time`. Flattening both time axes introduces duplicates and mixes initialisations. |
| `ice_conc` in dimensionless units `1`, range 0-1 | South requires `siconc` in percent. Convert finite values once with `100 * ice_conc`, rename the variable and update units/range metadata. |
| `ice_conc_observed` | Verification truth is not a forecast. Exclude it from the submission payload while retaining it separately for evaluation. |
| `latitude`, `longitude` and optional binary `land_mask`/`active_mask` | Coordinate arrays must align with the submitted grid. An active-ice mask is not an ocean-fraction field; binary masks alone do not establish coastal ocean fractions. |
| No `sftof` or `areacello` | Obtain and validate ocean fraction and cell area for the actual grid. Do not invent them from the presence of latitude/longitude or assume all grids have 25 km square cells. |
| One value at each dataset valid time | A daily timestamp does not establish a daily mean. The OSI SAF recipes sample at 12:00 UTC; confirm aggregation/acceptance for the South daily-mean request. |

The writer declares CF-1.10 metadata, not SIPN South CMIP6-format compliance. It can also clip out-of-range concentration and records clipping counts. Inspect these counts, missing values and masks before conversion. Land represented as zero in a model output must not silently become observed open water.

A submission-ready long trajectory, conversion of its time axis and units, grid-area/ocean-fraction metadata, and protocol validation remain separate work. No current short-horizon export is declared ready to submit by this guide.

## Arctic Sea Ice Outlook

### Core Outlook package

The [September 2026 call](https://www.seaiceprediction.org/sea-ice-outlook/september-2026-call-for-contributions) distinguishes the core Outlook form from the optional full-field upload. The core pan-Arctic contribution includes the September monthly mean extent, method description, plain-language summary and any justified uncertainty information. Pan-Antarctic and Alaska regional outlooks are additional contributions. Follow the selected form's units and fields; a new Outlook is required for each monthly round rather than carrying an older entry forward.

For the call's extent calculation, threshold concentration **for each day first**, sum qualifying cell areas, then average the daily extents across September. With concentration fraction `c`, valid ocean cells and cell areas `A` in km²:

```text
extent_day = sum(A[cell] for cells where c[day, cell] > 0.15)
september_mean_extent = mean(extent_day over all September days)
```

Thresholding a monthly mean concentration field is a different calculation. Likewise, the model's signed `sieerror` metric is an error against observations, not predicted extent. Exclude land and islands, document the treatment of partial-ocean cells, and reject unaccounted missing coverage rather than silently treating it as zero. Convert km² to million km² only when that is the requested reporting unit.

For Alaska, the call requests the combined Bering, Chukchi and Beaufort seas, preferably using its linked NSIDC 25 km polar-stereographic regional mask. Report the region definition **and maximum possible ice extent over its ocean cells**. CryoCast's EASE2 grid is not that polar-stereographic grid; do not copy a mask by array index merely because shapes or nominal resolution match.

### Full spatial fields

The call requests SIP (ensemble fraction above 15% September concentration), IFD (first melt-season date below 15%) and, for August/September submissions, IAD (first post-minimum freeze-up date above 15%). IFD80/IAD80 are additional 80% diagnostics. Supply dates as day of year and record the season searched, treatment of never-crossing cells and ensemble conventions. Initial-condition fields and their dates are also requested where available; SIC models cannot supply unmodelled thickness.

Daily SIC fields can be provided for the organisers to derive these metrics. This avoids inventing unconfirmed aggregation rules or presenting a one-member binary field as a calibrated ensemble probability. Confirm the required temporal coverage, including crossings outside September.

The call links the [spatial-metrics instructions](https://www.atmos.washington.edu/~ed/sio_metrics/), which still label themselves 2025. Those instructions request latitude, longitude, grid area and land identified with the NetCDF variable's `_FillValue`. Include those explicit requirements; neither linked page specifies a mandatory common grid or a complete current upload schema. Confirm remaining schema, missing-value and horizon details with the SIO team before upload.

Optional pan-Arctic extent anomalies and supplemental material remain separate from this first SIC workflow. The call gives its own baseline-trend and anomaly-sign instructions; do not substitute a generic sign convention. Its supplemental-PDF limit is less than 20 MB.

## SIPN South

The [published 2025-2026 call](https://fmassonn.github.io/sipn-south.github.io/call-contributions/) accepts one to five diagnostics in priority order. All three SIC-related formats below are not mandatory for a first contribution.

### Area diagnostics

The high-priority total-area file has one row of 90 comma-separated values. The medium-priority regional-area file has 36 rows of 90 values. Both use **million km²**, four decimal places and trailing zeroes. Historical filenames are:

```text
<group-name>_<forecast-id>_20251201-20260228_total-area.txt
<group-name>_<forecast-id>_20251201-20260228_regional-area.txt
```

Use `001` for a single forecast; an ensemble has one file per member with matching three-digit IDs across diagnostics. Regional rows cover successive half-open longitude intervals `[0, 10)`, `[10, 20)`, through `[350, 360)`. Normalize longitudes into `[0, 360)` and ensure each valid cell is counted once. Regional sums must reproduce the circumpolar total before rounding.

South requests **sea ice area**, not SIO extent. Apply concentration weighting rather than the 15% extent threshold. Record whether concentration is defined over the entire cell or only its ocean portion: for whole-cell concentration use `sum(c * cell_area)`; for ocean-relative concentration use `sum(c * ocean_fraction * cell_area)`. Do not apply the ocean fraction twice. With area inputs in m², divide the resulting sum by `1e12` to obtain million km². Confirm the convention with the source metadata and organiser when it is ambiguous.

### Concentration diagnostic

The lower-priority spatial diagnostic requests a 90-timestep NetCDF using CMIP6 conventions:

| Variable | Meaning | Units |
| --- | --- | --- |
| `siconc` | Sea ice concentration | % |
| `longitude`, `latitude` | Grid-cell coordinates | degrees east, degrees north |
| `sftof` | Ocean fraction of each grid cell | % |
| `areacello` | Grid-cell area | m² |

The historical filename is `<group-name>_<forecast-id>_20251201-20260228_concentration.nc`, with one file per ensemble member. No mandatory common spatial grid is stated. Keep aligned coordinates, cell areas, ocean fractions and documented missing-value conventions on the contributor grid.

The South page's regional/concentration headings say "February", but their explicit date ranges, counts and filenames span December-February. Use all 90 stated days for this historical protocol and confirm any ambiguous future call before generating files. Do not infer a new season's timestep count without checking its calendar.

### Other requested diagnostics

Lower-priority `sivol` is grid-cell mean ice thickness (volume per cell area), in metres; SIC alone cannot produce it. The published thickness filename ends in `20251228` despite a 90-day specification, so obtain clarification rather than silently choosing an end date. The optional long-forecast diagnostic requests monthly area from forecasts initialised in 2025 and extending at least six months into 2026. Neither is covered by the current SIC-only short-horizon presets.

## Validate before submission

Check the exact current call and record its URL, retrieval date, target period and submission cutoff. Then verify:

1. **Provenance and horizon.** Approved checkpoint/configuration, code commit, origin, conditioning-data availability, ensemble seeds and complete required horizon. No post-cutoff observations or reused retrospective truth.
2. **Calendar and frequency.** Correct first/last valid times, calendar, daily-mean convention and no missing/duplicate target dates. Each trajectory retains its own initialisation and member identity.
3. **Units and geography.** Fractions versus percentages, area conversions, coordinate ordering, CRS, masks, coastal ocean fractions and cell areas agree. Missing cells and clipping are accounted for.
4. **Numerical consistency.** Recompute extent/area from the archived fields. Check regional-total agreement before four-decimal rounding and confirm the text-file shape, delimiters and precision.
5. **Payload and naming.** Only requested forecast products, unique matching member IDs, current season filenames and protocol-compliant time/variable metadata. Do not upload verification observations or credentials.

Archive the unmodified output, exact final files, hashes and validation record before any submission. Successful unit tests or file checks establish implementation behaviour, not forecast accuracy or organiser acceptance.

## Submit manually

Use the current round's form for core SIO contributions and the separately documented SIPN data server for full fields. New full-field contributors should contact the person named in the current call for access and schema clarification. The September 2026 form is closed; retrospective contributions require organiser agreement.

The historical South workflow asks for a URL-accessible archive, its URL and method/contact metadata in the linked form, then an email to the organiser confirming submission. The call states that forecast and verification data will be made public. Confirm the applicable season, data-sharing permission and current route before making files accessible. This documentation change does not upload forecasts, submit forms or send email.
