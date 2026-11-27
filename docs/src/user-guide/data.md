# Data

CryoCast downloads data using [Anemoi](https://anemoi.readthedocs.io/projects/datasets/en/latest/).
The northern and southern hemispheres are treated as separate datasets.

- OSI SAF sea ice concentration (serving as both a model input and a forecast target)
- ERA5 atmospheric reanalysis fields (e.g. temperature, wind, radiation)
- Argo ocean float observations (e.g. surface and sub-surface ocean state)
- CS2SMOS sea ice thickness (northern hemisphere, freeze season only)

Each dataset has a "full" (multi-decadal) and a "sample" (2-5 year) variant, with the resolution and date range encoded into the filename, following [Anemoi conventions](https://anemoi.readthedocs.io/projects/registry/en/stable/naming-conventions.html).

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
The value for a given date therefore includes observations from up to three days later; when thickness is used as a model input, the most recent history steps overlap the start of the forecast period.
Satellite thickness retrievals are only possible during the freeze season (mid-October to mid-April), so summers are listed as missing dates.

The datasets are detailed in:

- `cryocast/config/data/datasets/full_sitnorth_cs2smos_25p0km_2010_2025_24h_v1.yaml`
- `cryocast/config/data/datasets/samp_sitnorth_cs2smos_25p0km_2020_2024_24h_v1.yaml`

and contain all variables in the product, of which these are used as model inputs:

- `analysis_sea_ice_thickness`: merged sea ice thickness (m)
- `analysis_sea_ice_thickness_unc`: uncertainty on the merged sea ice thickness (m)

Open water and land have no thickness, so missing values are filled with `0` for thickness and the other non-uncertainty variables (including `sea_ice_concentration` and `sea_ice_type`), and with a sentinel value of `99` for uncertainties.

Both are part of the `full_north` and `sample_north` data groups, so `cryocast datasets create` with those groups also downloads them.
They are only opened and used as inputs when selected by the variables config, so runs that do not select thickness do not need the thickness store.
To train with thickness, select the thickness variables and allow the thickness group to be missing, so that summer days are filled with a constant (in normalised units) rather than dropping every window that touches a summer:

```bash
cryocast train data=sample_north variables=sic-osisaf-sit-cs2smos data/allow_missing=sit-cs2smos
```
