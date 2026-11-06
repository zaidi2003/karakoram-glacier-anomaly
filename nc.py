"""Build the per-box ERA5 CSVs and the merged WGMS + climate table from
NetCDF files you already have on disk. No CDS requests are made.

Reads (from data_folder, written by run_box_analysis()):
    box_coordinates.csv, mass_balance_summary.csv, annual_mass_balance.csv

Reads (from nc_folder):
    every NetCDF file matching `pattern`, e.g. the cached downloads in
    results/era5_raw. Monthly-mean files and the geopotential (elevation)
    files are told apart automatically. Zip files holding several .nc streams
    (the new CDS format) are handled too.

Writes (to data_folder), the same files as era5_boxes.add_era5_to_boxes():
    era5_monthly.csv          box x month, converted to sensible units
    era5_static.csv           box -> mean grid elevation (m) and cell count
    era5_annual_features.csv  box x hydrological year, model-ready features
    box_features.csv          annual mass balance + climate features + box info
    era5_source_files.csv     which NetCDF file(s) each box was read from

Usage (notebook):
    from era5_from_nc import build_from_nc

    res = build_from_nc()                           # results/era5_raw
    res = build_from_nc(nc_folder="my_nc_files")    # any other folder
    res.features.head()

Needs the helper functions from the module that downloads the data. This file
looks for it as era5_data.py first, then era5_boxes.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from era5_data import (
        Era5Result,
        _annual_features,
        _convert,
        _extract,
        _open,
    )
except ImportError:
    from era5_boxes import (
        Era5Result,
        _annual_features,
        _convert,
        _extract,
        _open,
    )


DEFAULT_DATA_FOLDER = "results/data"
DEFAULT_NC_FOLDER = "results/era5_raw"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _is_static(ds):
    """A geopotential-only file (orography), not a monthly climate file."""
    return "z" in ds.data_vars and len(ds.data_vars) == 1


def _boxes_covered(ds, bounds):
    """
    Boxes lying inside the file's grid extent (within half a cell). This
    stops a box outside the file from being filled with the nearest edge cell.
    """

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lon = np.where(lon > 180, lon - 360, lon)

    def half_step(values):
        values = np.unique(values)
        return float(np.diff(values).mean()) / 2 if len(values) > 1 else 0.125

    hy, hx = half_step(lat), half_step(lon)
    tol = 1e-6

    return {
        bid: b
        for bid, b in bounds.items()
        if b[0] >= lon.min() - hx - tol
        and b[1] <= lon.max() + hx + tol
        and b[2] >= lat.min() - hy - tol
        and b[3] <= lat.max() + hy + tol
    }


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def build_from_nc(
    nc_folder=DEFAULT_NC_FOLDER,
    data_folder=DEFAULT_DATA_FOLDER,
    pattern="*.nc",
    only_analysed=True,
    regions=None,
    verbose=True,
):
    """
    Average the stored NetCDF data inside every box, convert the units, build
    the annual features and merge them with the WGMS annual mass balance.

    nc_folder      Folder holding the NetCDF files (not searched recursively).
    pattern        File name pattern. Use "era5_land_*.nc" or "era5_Karakoram*"
                   to pick a subset. Do not mix ERA5 and ERA5-Land files.
    only_analysed  True: boxes in mass_balance_summary.csv only.
                   False: every grid box in box_coordinates.csv.
    regions        Only these regions, e.g. ["Karakoram"].
    """

    folder = Path(data_folder)
    nc_dir = Path(nc_folder)

    for name in (
        "box_coordinates.csv",
        "mass_balance_summary.csv",
        "annual_mass_balance.csv",
    ):
        if not (folder / name).exists():
            raise FileNotFoundError(
                f"{folder / name} not found. Run run_box_analysis() first."
            )

    files = sorted(nc_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {nc_dir.resolve()}."
        )

    names = [f.name for f in files]
    has_land = any(n.startswith("era5_land") for n in names)
    has_era5 = any(
        n.startswith("era5_") and not n.startswith("era5_land") for n in names
    )

    if has_land and has_era5:
        raise ValueError(
            "The folder mixes ERA5 and ERA5-Land files. Use pattern="
            "'era5_land_*.nc' or move one set to another folder."
        )

    # 1. Boxes (CSV is the source of truth).
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

    # 2. Read every file and average inside the boxes it covers.
    monthly_parts = []
    static_parts = []
    source_rows = []

    for f in files:

        try:
            ds = _open(f)
        except Exception as err:
            print(f"Skipping {f.name}: cannot open ({err})")
            continue

        covered = _boxes_covered(ds, bounds)

        if not covered:
            if verbose:
                print(f"{f.name}: covers none of the selected boxes, skipped")
            continue

        part = _extract(ds, covered)
        kind = "static" if _is_static(ds) else "monthly"

        (static_parts if kind == "static" else monthly_parts).append(part)

        source_rows.extend(
            {"box": bid, "file": f.name, "kind": kind} for bid in covered
        )

        if verbose:
            extra = ""
            if kind == "monthly":
                extra = (
                    f", {part['date'].min():%Y-%m} to "
                    f"{part['date'].max():%Y-%m}"
                )
            print(f"{f.name}: {kind}, {len(covered)} boxes{extra}")

    if not monthly_parts:
        raise ValueError(
            "None of the NetCDF files cover the selected boxes. Check that "
            "nc_folder, pattern and regions are right."
        )

    # Overlapping files (padded areas, shared years) give duplicate rows;
    # keep the first non-missing value for each box and month.
    raw = (
        pd.concat(monthly_parts, ignore_index=True)
        .groupby(["box", "date"], as_index=False)
        .first()
    )

    monthly = _convert(raw).sort_values(["box", "date"]).reset_index(drop=True)

    if static_parts:
        s = (
            pd.concat(static_parts, ignore_index=True)
            .groupby("box", as_index=False)
            .first()
        )
        s["elevation_m"] = s["z"] / 9.80665
        static = s[["box", "elevation_m", "n_cells"]]
    else:
        static = pd.DataFrame(
            {"box": sorted(set(raw["box"])), "elevation_m": np.nan,
             "n_cells": np.nan}
        )
        print(
            "No geopotential file found, so elevation_m is empty. Download "
            "the elevation file or ignore this."
        )

    # 3. Annual features and the merged modelling table.
    annual_clim = _annual_features(monthly, lat_of)

    annual = pd.read_csv(folder / "annual_mass_balance.csv")

    info = pd.DataFrame(
        {
            "box": list(bounds),
            "region": [region_of[b] for b in bounds],
            "lon": [(bounds[b][0] + bounds[b][1]) / 2 for b in bounds],
            "lat": [lat_of[b] for b in bounds],
        }
    )

    features = (
        annual[annual["box"].isin(bounds)]
        .merge(annual_clim, on=["box", "year"], how="left")
        .merge(info, on="box", how="left")
        .merge(static, on="box", how="left")
    )

    # 4. Save.
    monthly.to_csv(folder / "era5_monthly.csv", index=False)
    static.to_csv(folder / "era5_static.csv", index=False)
    annual_clim.to_csv(folder / "era5_annual_features.csv", index=False)
    features.to_csv(folder / "box_features.csv", index=False)
    pd.DataFrame(source_rows).to_csv(
        folder / "era5_source_files.csv", index=False
    )

    # 5. Report coverage.
    if verbose:

        print(f"\nCSVs saved to: {folder.resolve()}")

        covered_boxes = set(monthly["box"])
        missing = [b for b in bounds if b not in covered_boxes]

        if missing:
            by_region = pd.Series([region_of[b] for b in missing]).value_counts()
            print(
                f"\n{len(missing)} selected boxes have no climate data "
                "(no file covers them):"
            )
            print(by_region.to_string())

        if "temp_annual_c" in features:
            have = features.dropna(subset=["temp_annual_c"])
            print(
                f"\n{len(have)} of {len(features)} box-years have both "
                "mass balance and climate:"
            )
            print(
                have.groupby("region")
                .agg(
                    box_years=("year", "size"),
                    first_year=("year", "min"),
                    last_year=("year", "max"),
                )
                .to_string()
            )

            check = [
                c
                for c in ("temp_annual_c", "precip_annual_mm")
                if c in annual_clim
            ]
            tmp = annual_clim.assign(region=annual_clim["box"].map(region_of))
            print("\nSanity check, mean annual values by region:")
            print(tmp.groupby("region")[check].mean().round(1).to_string())

    return Era5Result(monthly, static, annual_clim, features)


if __name__ == "__main__":
    build_from_nc()