"""Attach ERA5 / ERA5-Land climate data to every box and save it as CSV.

Reads the CSVs written by run_box_analysis() (box_coordinates.csv,
mass_balance_summary.csv, annual_mass_balance.csv), downloads monthly-mean
climate from the Copernicus Climate Data Store (CDS), averages it inside each
box, and writes new CSVs next to the old ones:

    era5_monthly.csv          box x month, converted to sensible units
    era5_static.csv           box -> mean grid elevation (m) and cell count
    era5_annual_features.csv  box x hydrological year, model-ready features
    box_features.csv          annual mass balance + climate features + box info
                              (the file to load when building models)
    era5_config.csv           settings used (drives the cache check)

Requires:
    pip install cdsapi xarray netcdf4 pandas numpy python-dotenv

    A free CDS account (separate from the Sentinel / Data Space account) and
    the licence accepted once on the web page of each dataset. Credentials go
    in the .env file of your project, next to your Sentinel ones:

        CDSAPI_URL=https://cds.climate.copernicus.eu/api
        CDSAPI_KEY=<personal access token from your CDS profile page>

    (A ~/.cdsapirc file also works if there is no .env.)

Usage:
    from era5_boxes import add_era5_to_boxes, load_box_features

    add_era5_to_boxes(dataset="era5")          # 0.25 deg, matches the boxes
    add_era5_to_boxes(dataset="era5_land")     # 0.1 deg, averaged inside box
    df = load_box_features()
"""

import hashlib
import time
import warnings
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_DATA_FOLDER = "results/data"
DEFAULT_RAW_FOLDER = "results/era5_raw"

DATASETS = {
    "era5": {
        "monthly": "reanalysis-era5-single-levels-monthly-means",
        "hourly": "reanalysis-era5-single-levels",
        "hourly_product_type": "reanalysis",
        "resolution": 0.25,
        "first_year": 1940,
    },
    "era5_land": {
        "monthly": "reanalysis-era5-land-monthly-means",
        "hourly": "reanalysis-era5-land",
        "hourly_product_type": None,
        "resolution": 0.1,
        "first_year": 1950,
    },
}

_COMMON_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "skin_temperature",
    "total_precipitation",
    "snowfall",
    "snowmelt",
    "surface_net_solar_radiation",
    "surface_solar_radiation_downwards",
    "surface_net_thermal_radiation",
    "surface_thermal_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]

DEFAULT_VARIABLES = {
    "era5": _COMMON_VARIABLES + ["snow_depth"],
    "era5_land": _COMMON_VARIABLES + ["snow_depth_water_equivalent"],
}

# Hydrological year start month and season months by hemisphere. The year label
# is the calendar year in which the hydrological year ENDS. Change these if
# your WGMS balance years are defined differently.
HYDRO_START_MONTH = {"north": 10, "south": 4}

SEASONS = {
    "north": {"ablation": [5, 6, 7, 8, 9]},
    "south": {"ablation": [11, 12, 1, 2, 3]},
}


# Years per CDS request. Lower this (e.g. 5) if you still hit "cost limits
# exceeded".
CHUNK_YEARS = 10

# Regions downloaded first, in this order. Everything else follows.
# Requests are submitted strictly one at a time: ECMWF asks for sequential
# submission and disables accounts that run requests in parallel.
PRIORITY_REGIONS = ["Karakoram", "Pamir", "Kunlun"]


@dataclass
class Era5Result:
    monthly: pd.DataFrame
    static: pd.DataFrame
    annual_features: pd.DataFrame
    features: pd.DataFrame


# --------------------------------------------------------------------------
# CDS download
# --------------------------------------------------------------------------

def _make_client():
    """
    Create the CDS client, reading CDSAPI_URL and CDSAPI_KEY from the .env
    file in the working directory (or any parent) if one exists. Falls back to
    ~/.cdsapirc when the variables are not set.
    """

    import os

    import cdsapi

    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    url = os.environ.get("CDSAPI_URL")
    key = os.environ.get("CDSAPI_KEY")

    if url and key:
        return cdsapi.Client(url=url, key=key)

    return cdsapi.Client()


def _retrieve(client, name, request, target):
    """Download to a .part file and rename when complete, so an
    interrupted download never leaves a corrupt cached file."""

    target = Path(target)
    tmp = target.with_name(target.name + ".part")

    client.retrieve(name, request, str(tmp))
    tmp.replace(target)


def _area_for(bounds, pad):
    """CDS area = [north, west, south, east], padded by one grid cell."""

    lon_min = min(b[0] for b in bounds.values()) - pad
    lon_max = max(b[1] for b in bounds.values()) + pad
    lat_min = min(b[2] for b in bounds.values()) - pad
    lat_max = max(b[3] for b in bounds.values()) + pad

    return [
        round(lat_max, 3),
        round(lon_min, 3),
        round(lat_min, 3),
        round(lon_max, 3),
    ]


def _slug(text):
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def _download_monthly(client, dataset, variables, y0, y1, area, target):

    spec = DATASETS[dataset]

    _retrieve(
        client,
        spec["monthly"],
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variables,
            "year": [str(y) for y in range(y0, y1 + 1)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area,
        },
        target,
    )


def _download_static(client, dataset, area, target):
    """Surface geopotential (orography) for one time step."""

    spec = DATASETS[dataset]

    request = {
        "variable": ["geopotential"],
        "year": "2000",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }

    if spec["hourly_product_type"]:
        request["product_type"] = spec["hourly_product_type"]

    _retrieve(client, spec["hourly"], request, target)


# --------------------------------------------------------------------------
# NetCDF reading and per-box averaging
# --------------------------------------------------------------------------

def _open(path):
    """
    Open a CDS download. The new CDS may return a zip holding several .nc
    files (instantaneous and accumulated variables are separate streams).
    """

    import xarray as xr

    path = Path(path)

    if zipfile.is_zipfile(path):

        out = path.with_name(path.stem + "_unzipped")
        out.mkdir(exist_ok=True)

        with zipfile.ZipFile(path) as z:
            z.extractall(out)

        parts = [
            xr.open_dataset(f).drop_vars(["number", "expver"], errors="ignore")
            for f in sorted(out.glob("*.nc"))
        ]

        ds = xr.merge(parts, compat="override")

    else:
        ds = xr.open_dataset(path)

    rename = {}

    if "valid_time" in ds.dims:
        rename["valid_time"] = "time"
    if "lat" in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims:
        rename["lon"] = "longitude"

    ds = ds.rename(rename)

    # Recent months can arrive as ERA5T (expver=5); merge with final data.
    if "expver" in ds.dims:
        ds = ds.sel(expver=1).drop_vars("expver", errors="ignore").combine_first(
            ds.sel(expver=5).drop_vars("expver", errors="ignore")
        )

    return ds.load()


def _extract(ds, bounds):
    """
    Average every grid cell whose centre lies inside each box. For ERA5
    (0.25 deg, aligned with the boxes) that is exactly one cell; for
    ERA5-Land (0.1 deg) it is a few cells. Returns a long DataFrame.
    """

    core = ("time", "latitude", "longitude")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lon = np.where(lon > 180, lon - 360, lon)

    times = pd.to_datetime(ds["time"].values)

    arrays = {}

    for name in ds.data_vars:

        da = ds[name]

        if not set(core) <= set(da.dims):
            continue

        extra = [d for d in da.dims if d not in core]
        da = da.isel({d: 0 for d in extra}).transpose(*core)
        arrays[name] = da.values

    tol = 1e-6
    frames = []

    for bid, (lon_min, lon_max, lat_min, lat_max) in bounds.items():

        li = np.where((lat >= lat_min - tol) & (lat <= lat_max + tol))[0]
        lj = np.where((lon >= lon_min - tol) & (lon <= lon_max + tol))[0]

        if li.size == 0:
            li = np.array([np.abs(lat - (lat_min + lat_max) / 2).argmin()])
        if lj.size == 0:
            lj = np.array([np.abs(lon - (lon_min + lon_max) / 2).argmin()])

        df = pd.DataFrame({"box": bid, "date": times})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            for name, arr in arrays.items():
                df[name] = np.nanmean(arr[:, li][:, :, lj], axis=(1, 2))

        df["n_cells"] = int(li.size * lj.size)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Unit conversion
# --------------------------------------------------------------------------

def _convert(raw):
    """
    Convert CDS monthly-mean units.

    Accumulated variables (tp, sf, smlt) come as mean DAILY totals in metres,
    so they are scaled by days in the month to give mm per month. Radiation
    comes as J m-2 per day, so it is divided by 86400 to give W m-2.
    """

    days = raw["date"].dt.days_in_month
    out = raw[["box", "date"]].copy()

    for c in raw.columns:

        if c in ("box", "date", "n_cells"):
            continue

        if c in ("t2m", "d2m", "skt"):
            out[f"{c}_c"] = raw[c] - 273.15
        elif c in ("tp", "sf", "smlt"):
            out[f"{c}_mm"] = raw[c] * 1000.0 * days
        elif c in ("ssr", "ssrd", "str", "strd"):
            out[f"{c}_wm2"] = raw[c] / 86400.0
        elif c == "sp":
            out["sp_hpa"] = raw[c] / 100.0
        elif c == "sd":
            out["snow_mwe"] = raw[c]
        else:
            out[c] = raw[c]

    if {"u10", "v10"} <= set(raw.columns):
        out["wind_speed_ms"] = np.hypot(raw["u10"], raw["v10"])

    return out


# --------------------------------------------------------------------------
# Annual (hydrological-year) features
# --------------------------------------------------------------------------

def _annual_features(monthly, lat_of):
    """
    Collapse monthly data to one row per box and hydrological year. Years
    without all 12 months are dropped. pdd_monthly_proxy is a rough positive
    degree-day sum from monthly means (it underestimates true PDD).
    """

    rows = []

    for bid, g in monthly.groupby("box"):

        hemi = "south" if lat_of[bid] < 0 else "north"
        start_m = HYDRO_START_MONTH[hemi]

        month = g["date"].dt.month
        year = g["date"].dt.year

        g = g.assign(
            hyear=np.where(month >= start_m, year + 1, year),
            abl=month.isin(SEASONS[hemi]["ablation"]),
        )

        for yr, y in g.groupby("hyear"):

            if len(y) < 12:
                continue

            a = y[y["abl"]]
            c = y[~y["abl"]]

            row = {"box": bid, "year": int(yr)}

            if "t2m_c" in y:
                row["temp_annual_c"] = y["t2m_c"].mean()
                row["temp_ablation_c"] = a["t2m_c"].mean()
                row["temp_accum_c"] = c["t2m_c"].mean()
                row["pdd_monthly_proxy"] = float(
                    (
                        y["t2m_c"].clip(lower=0)
                        * y["date"].dt.days_in_month
                    ).sum()
                )

            for key, col in (
                ("precip", "tp_mm"),
                ("snowfall", "sf_mm"),
                ("snowmelt", "smlt_mm"),
            ):
                if col in y:
                    row[f"{key}_annual_mm"] = y[col].sum()
                    row[f"{key}_accum_mm"] = c[col].sum()
                    row[f"{key}_ablation_mm"] = a[col].sum()

            if "precip_accum_mm" in row and "snowfall_accum_mm" in row:
                p = row["precip_accum_mm"]
                row["snowfall_frac_accum"] = (
                    row["snowfall_accum_mm"] / p if p > 0 else np.nan
                )

            for col in ("ssrd_wm2", "ssr_wm2", "str_wm2", "strd_wm2"):
                if col in y:
                    row[col.replace("_wm2", "") + "_ablation_wm2"] = (
                        a[col].mean()
                    )

            if "ssr_wm2" in y and "str_wm2" in y:
                row["net_radiation_ablation_wm2"] = (
                    a["ssr_wm2"] + a["str_wm2"]
                ).mean()

            if "wind_speed_ms" in y:
                row["wind_speed_annual_ms"] = y["wind_speed_ms"].mean()

            if "sp_hpa" in y:
                row["pressure_annual_hpa"] = y["sp_hpa"].mean()

            rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cache check
# --------------------------------------------------------------------------

OUTPUT_CSVS = [
    "era5_monthly.csv",
    "era5_static.csv",
    "era5_annual_features.csv",
    "box_features.csv",
    "era5_config.csv",
]


def _box_hash(bounds):
    text = ";".join(
        f"{b}:{','.join(f'{v:.4f}' for v in bounds[b])}"
        for b in sorted(bounds)
    )
    return hashlib.md5(text.encode()).hexdigest()


def _cache_ok(folder, settings):

    if any(not (folder / n).exists() for n in OUTPUT_CSVS):
        return False

    cfg = (
        pd.read_csv(folder / "era5_config.csv", dtype=str)
        .set_index("key")["value"]
        .to_dict()
    )

    return all(str(cfg.get(k)) == str(v) for k, v in settings.items())


def _load_saved(folder):

    monthly = pd.read_csv(folder / "era5_monthly.csv", parse_dates=["date"])
    static = pd.read_csv(folder / "era5_static.csv")
    annual = pd.read_csv(folder / "era5_annual_features.csv")
    features = pd.read_csv(folder / "box_features.csv")

    return Era5Result(monthly, static, annual, features)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def add_era5_to_boxes(
    data_folder=DEFAULT_DATA_FOLDER,
    dataset="era5",
    variables=None,
    start=None,
    end=None,
    raw_folder=DEFAULT_RAW_FOLDER,
    only_analysed=True,
    force=False,
    verbose=True,
    regions=None,
    elevation=True,
):
    """
    Download ERA5 or ERA5-Land monthly means and save them per box.

    dataset        "era5" (0.25 deg, one cell per box) or "era5_land"
                   (0.1 deg, cells inside the box are averaged).
    variables      CDS variable names; defaults to DEFAULT_VARIABLES[dataset].
    start, end     balance years to cover. Default: the full range in
                   annual_mass_balance.csv (end capped at last calendar year).
                   One extra year before start is fetched for the
                   hydrological-year grouping.
    only_analysed  True: boxes in mass_balance_summary.csv only.
                   False: every grid box in box_coordinates.csv.
    force          Redownload / recompute even if the saved CSVs match.
    elevation      False skips the elevation request (elevation_m is then
                   empty). Useful if the hourly-dataset licence is missing.
    regions        Only these regions, e.g. ["Karakoram", "Pamir", "Kunlun"].
                   Downloaded files are cached, so a later full run reuses
                   them. Note the saved CSVs are overwritten by each run.

    Raw CDS files are kept in raw_folder and reused, so re-running after a
    crash does not download again.
    """

    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")

    spec = DATASETS[dataset]
    folder = Path(data_folder)

    for name in (
        "box_coordinates.csv",
        "mass_balance_summary.csv",
        "annual_mass_balance.csv",
    ):
        if not (folder / name).exists():
            raise FileNotFoundError(
                f"{folder / name} not found. Run run_box_analysis() first."
            )

    variables = list(variables or DEFAULT_VARIABLES[dataset])

    # 1. Box geometry and selection (CSV is the source of truth).
    stats = pd.read_csv(folder / "mass_balance_summary.csv", index_col="box")
    stats.index = stats.index.astype(int)

    geometry = pd.read_csv(folder / "box_coordinates.csv")

    ids = (
        list(stats.index)
        if only_analysed
        else sorted(geometry["box"].unique())
    )

    if regions is not None:

        wanted = {regions} if isinstance(regions, str) else set(regions)
        by_box = geometry.drop_duplicates("box").set_index("box")["region"]
        ids = [b for b in ids if by_box.get(b) in wanted]

        if not ids:
            raise ValueError(f"No boxes found for regions {sorted(wanted)}.")

    geometry = geometry[geometry["box"].isin(ids)]

    bounds = {}
    region_of = {}
    lat_of = {}

    for bid, g in geometry.groupby("box"):
        bid = int(bid)
        bounds[bid] = (
            g["longitude"].min(),
            g["longitude"].max(),
            g["latitude"].min(),
            g["latitude"].max(),
        )
        region_of[bid] = g["region"].iloc[0]
        lat_of[bid] = (g["latitude"].min() + g["latitude"].max()) / 2

    # 2. Years.
    annual = pd.read_csv(folder / "annual_mass_balance.csv")

    first = int(annual["year"].min()) if start is None else int(start)
    last = int(annual["year"].max()) if end is None else int(end)

    if end is None:
        last = min(last, date.today().year - 1)

    y0 = max(first - 1, spec["first_year"])
    y1 = last

    settings = {
        "dataset": dataset,
        "year_from": y0,
        "year_to": y1,
        "variables": ";".join(variables),
        "boxes": _box_hash(bounds),
    }

    if not force and _cache_ok(folder, settings):
        if verbose:
            print(
                f"Found saved ERA5 CSVs in {folder.resolve()} for these "
                "settings; loading them (force=True to recompute)."
            )
        return _load_saved(folder)

    # 3. Download (strictly sequential, priority regions first), then extract.
    raw_dir = Path(raw_folder)
    raw_dir.mkdir(parents=True, exist_ok=True)

    var_tag = hashlib.md5(";".join(sorted(variables)).encode()).hexdigest()[:8]

    # Priority regions first, then the rest in the order they were defined.
    present = list(dict.fromkeys(region_of[b] for b in sorted(bounds)))
    order = [r for r in PRIORITY_REGIONS if r in present] + [
        r for r in present if r not in PRIORITY_REGIONS
    ]

    plan = {}
    jobs = []

    for region in order:

        rb = {b: bounds[b] for b in bounds if region_of[b] == region}
        area = _area_for(rb, spec["resolution"])
        tag = f"{dataset}_{_slug(region)}_" + "_".join(map(str, area))

        static_nc = raw_dir / f"{tag}_static.nc"
        files = []

        # Small requests stay under the CDS cost limit; each is cached.
        for a in range(y0, y1 + 1, CHUNK_YEARS):

            b = min(a + CHUNK_YEARS - 1, y1)
            f = raw_dir / f"{tag}_{a}-{b}_{var_tag}.nc"
            files.append(f)

            if not f.exists():
                jobs.append(
                    {
                        "kind": "monthly",
                        "region": region,
                        "label": f"{a}-{b}",
                        "a": a,
                        "b": b,
                        "area": area,
                        "target": f,
                    }
                )

        if elevation and not static_nc.exists():
            jobs.append(
                {
                    "kind": "static",
                    "region": region,
                    "label": "elevation",
                    "area": area,
                    "target": static_nc,
                }
            )

        plan[region] = {"rb": rb, "files": files, "static": static_nc}

    if jobs:

        if verbose:
            print(
                f"{len(jobs)} CDS requests to run, strictly one at a time, "
                f"in this region order: {', '.join(order)}"
            )

        client = _make_client()

        for n, job in enumerate(jobs, 1):

            if verbose:
                print(
                    f"({n}/{len(jobs)}) [{job['region']}] "
                    f"requesting {job['label']}"
                )

            t0 = time.time()

            try:
                if job["kind"] == "monthly":
                    _download_monthly(
                        client, dataset, variables, job["a"], job["b"],
                        job["area"], job["target"],
                    )
                else:
                    _download_static(
                        client, dataset, job["area"], job["target"]
                    )

            except Exception as err:
                # Stop at the first failure: the CDS flags accounts that keep
                # submitting invalid requests. Finished files stay cached.
                raise RuntimeError(
                    f"CDS request failed for [{job['region']}] "
                    f"{job['label']}: {err}\n\n"
                    "Stopped without submitting any further requests. Check "
                    "https://cds.climate.copernicus.eu/requests for the "
                    "reason, fix it, then re-run; finished downloads are "
                    "cached."
                ) from err

            if verbose:
                print(
                    f"    done in {(time.time() - t0) / 60:.1f} min"
                )

    monthly_parts = []
    static_parts = []

    for region in order:

        p = plan[region]

        if verbose:
            print(f"[{region}] averaging {len(p['rb'])} boxes")

        monthly_parts.extend(_extract(_open(f), p["rb"]) for f in p["files"])

        if p["static"].exists():
            s = _extract(_open(p["static"]), p["rb"])
            s = s.groupby("box").first().reset_index()
            s["elevation_m"] = s["z"] / 9.80665
            static_parts.append(s[["box", "elevation_m", "n_cells"]])

    monthly = _convert(pd.concat(monthly_parts, ignore_index=True))
    monthly = monthly.sort_values(["box", "date"]).reset_index(drop=True)

    if static_parts:
        static = pd.concat(static_parts, ignore_index=True)
    else:
        static = pd.DataFrame(
            {"box": list(bounds), "elevation_m": np.nan, "n_cells": np.nan}
        )

    # 4. Annual features and the merged modelling table.
    annual_clim = _annual_features(monthly, lat_of)

    info = pd.DataFrame(
        {
            "box": list(bounds),
            "region": [region_of[b] for b in bounds],
            "lon": [(bounds[b][0] + bounds[b][1]) / 2 for b in bounds],
            "lat": [lat_of[b] for b in bounds],
        }
    )

    features = (
        annual.merge(annual_clim, on=["box", "year"], how="left")
        .merge(info, on="box", how="left")
        .merge(static, on="box", how="left")
    )

    # 5. Save.
    monthly.to_csv(folder / "era5_monthly.csv", index=False)
    static.to_csv(folder / "era5_static.csv", index=False)
    annual_clim.to_csv(folder / "era5_annual_features.csv", index=False)
    features.to_csv(folder / "box_features.csv", index=False)

    pd.DataFrame(
        [{"key": k, "value": v} for k, v in settings.items()]
    ).to_csv(folder / "era5_config.csv", index=False)

    if verbose:
        print(f"\nERA5 CSVs saved to: {folder.resolve()}")

        check = [
            c
            for c in ("temp_annual_c", "precip_annual_mm", "snowfall_annual_mm")
            if c in annual_clim
        ]

        if check:
            print("\nSanity check, mean annual values by region:")
            tmp = annual_clim.assign(region=annual_clim["box"].map(region_of))
            print(tmp.groupby("region")[check].mean().round(1).to_string())

    return Era5Result(monthly, static, annual_clim, features)


def load_box_features(data_folder=DEFAULT_DATA_FOLDER):
    """Load the modelling table: mass balance + ERA5 features, one row per box-year."""

    path = Path(data_folder) / "box_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run add_era5_to_boxes() first."
        )

    return pd.read_csv(path)