# Data

CryoCast downloads data using [Anemoi](https://anemoi.readthedocs.io/projects/datasets/en/latest/).
The northern and southern hemispheres are treated as separate datasets.

- OSI SAF sea ice concentration (serving as both a model input and a forecast target)
- ERA5 atmospheric reanalysis fields (e.g. temperature, wind, radiation)
- Argo ocean float observations (e.g. surface and sub-surface ocean state)
- CS2SMOS sea ice thickness (northern hemisphere, freeze season only)

Each dataset has a "full" (multi-decadal) and/or a "sample" (2-3 year) variant (sea ice thickness currently has a sample variant only), with the resolution and date range encoded into the filename, following [Anemoi conventions](https://anemoi.readthedocs.io/projects/registry/en/stable/naming-conventions.html).

## OSI SAF data

To obtain continuous sea-ice concentration records from 1979 through to the present day, a series of OSI SAF datasets with 25 km spatial resolution are combined.
Each of these is a L3/L4 reprocessed product, drawing primarily on satellite data.

| Period | Product | Sensor family |
| --- | --- | --- |
| 1979-01-01 to 2002-12-31 | OSI-450-a1 | SMMR, SSM/I, SSMIS |
| 2003-01-01 to 2011-10-04 | OSI-458 | AMSR-E |
| 2011-10-05 to 2012-07-23 | OSI-450-a1 | SSMIS |
| 2012-07-24 to 2020-12-31 | OSI-458 | AMSR2 |
| 2021-01-01 to 2025-12-31 | OSI-438 | AMSR2 |

The full-resolution datasets are detailed in:

- `cryocast/config/data/datasets/full_sicnorth_osisaf_25p0km_1979_2025_24h_v1.yaml`
- `cryocast/config/data/datasets/full_sicsouth_osisaf_25p0km_1979_2025_24h_v1.yaml`

and contain the following subset of [available variables](https://osisaf-hl.met.no/sites/osisaf-hl/files/user_manuals/osisaf_pum_sea-ice-conc-climate-data-record_3.3.pdf)

- `ice_conc`: sea ice concentration
- `status_flag`: categorisation of the predominant surface
- `total_standard_uncertainty`: uncertainty on the sea ice concentration

### Other OSI SAF products

Higher resolution OSI SAF datasets exist (for example with 10 km spatial resolution), but these are not currently used by CryoCast.

## ERA 5 data

Atmospheric reanalysis data from 1979 through to the present day is retrieved from [ERA5](https://cds.climate.copernicus.eu/) via the CDS MARS API.
Data is download at 0.25 degree resolution before being reprojected onto the same 25 km EASE2 grid as the sea-ice concentration records.

The full-resolution datasets are detailed in:

- `cryocast/config/data/datasets/full_weathernorth_era5_25p0km_1979_2025_24h_v3.yaml`
- `cryocast/config/data/datasets/full_weathersouth_era5_25p0km_1979_2025_24h_v3.yaml`

and contain the following subset of [available variables](https://codes.ecmwf.int/grib/param-db?encoding=grib2&ordering=id&limit=20&page=1):

Surface fields (`levtype: sfc`):

- `2t`: 2 m air temperature
- `sp`: surface pressure
- `10u`, `10v`: 10 m wind components
- `msl`: mean sea level pressure

Pressure-level fields (`levtype: pl`, at 10, 250, 500 and 1000 hPa):

- `z`: geopotential height
- `t`: air temperature
- `q`: specific humidity
- `u`, `v`: wind components

Time-varying forcings, computed rather than retrieved:

- `cos_julian_day`, `sin_julian_day`: encoded day of year
- `insolation`: top-of-atmosphere solar radiation

## Argo float data

In-situ ocean observations are drawn from the [Argo](https://argo.ucsd.edu/) float programme, gridded onto the same 25 km EASE2 grid as the other datasets.
Coverage begins on 1999-07-26, once enough floats were deployed to provide usable spatial coverage, and is sparser in the early years, with many missing dates before continuous daily coverage becomes available.

The full-resolution datasets are detailed in:

- `cryocast/config/data/datasets/full_floatnorth_argo_25p0km_1999_2025_24h_v2.yaml`
- `cryocast/config/data/datasets/full_floatsouth_argo_25p0km_1999_2025_24h_v2.yaml`

and contain the following variables:

- `TEMP`: ocean temperature
- `PSAL`: practical salinity

Missing values (where no float profile is available at a grid cell) are filled with a sentinel value of `99` rather than left as `NaN`.

## CS2SMOS sea ice thickness

Arctic sea ice thickness is taken from the [AWI CS2SMOS](https://spaces.awi.de/confluence/x/DwVmEQ) product (v2.06), which merges CryoSat-2 radar altimetry with SMOS L-band radiometry.
It is distributed on the same 25 km EASE2 grid as the sea ice concentration records, so no reprojection is needed.
Files are downloaded anonymously from `ftp://ftp.awi.de/sea_ice/product/cryosat2_smos/v206/nh/`.

Each daily file is a 7-day running mean centred on the file date, and the dataset indexes files by that centre date.
Satellite thickness retrievals are only possible during the freeze season (mid-October to mid-April), so summers are listed as missing dates.
Because training windows are only drawn from dates available in every input dataset, models using thickness are trained and evaluated on freeze-season windows only.

The sample dataset (November-March, winters 2019/20 to 2022/23) is detailed in:

- `cryocast/config/data/datasets/samp_sitnorth_cs2smos_25p0km_2019_2023_24h_v1.yaml`

and contains all variables in the product, of which these are used as model inputs:

- `analysis_sea_ice_thickness`: merged sea ice thickness (m)
- `analysis_sea_ice_thickness_unc`: uncertainty on the merged sea ice thickness (m)

Open water and land have no thickness, so missing thickness values are filled with `0` and missing uncertainties with a sentinel value of `99`.

It is paired with a matching freeze-season SSMIS sea ice concentration sample covering the same dates:

- `cryocast/config/data/datasets/samp_sicnorth_ssmis_25p0km_2019_2023_24h_v1.yaml`

Both datasets keep their normalisation statistics within the training winters.
Create them (this also generates the land and active-cell masks) before training:

```bash
cryocast datasets create --config-name sample_north_sit
```

To train on sea ice concentration plus thickness, use the `sample_north_sit` config.
This uses the freeze-season `sample_sit_winters` split, where each winter is listed as its own period, so that a SIC-only baseline trained by dropping the thickness inputs uses the same split windows:

```bash
cryocast train --config-name sample_north_sit
cryocast train --config-name sample_north_sit '~variables.input.sit-cs2smos'
```

Any encode-process-decode `model=` choice can be used, with thickness encoded by the same encoder type as sea ice concentration.
Because the training periods contain no April-October dates, the calendar-day climatology cannot be built, so climatology-based metrics and `model=climatology` are unavailable.
Models without per-group encoders (`persistence`, `climatology`, `ddpm`) are also not supported.
