"""Regional glacier mass balance from WGMS data inside a rough boundary polygon."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon

WGMS_FOLDER = Path("wgms-amce-2026-02-10/glacier")

# Rough boundaries as (longitude, latitude). Approximate research boundaries,
# not official range boundaries.
KARAKORAM = [
    (72.5, 36.5), (73.0, 37.2), (74.2, 37.8), (76.0, 38.0), (78.2, 37.4),
    (79.0, 36.5), (78.0, 35.0), (76.5, 34.2), (74.5, 34.4), (73.2, 35.0),
    (72.5, 36.5),
]
SWISS_ALPS = [
    (5.9, 46.1), (6.2, 46.8), (6.8, 47.0), (7.5, 47.6), (8.5, 47.8),
    (9.5, 47.5), (10.5, 47.0), (10.5, 46.3), (10.0, 45.8), (9.2, 45.7),
    (8.2, 45.8), (7.2, 45.8), (6.3, 45.9), (5.9, 46.1),
]


def _show(df):
    """Use display() in notebooks, fall back to print elsewhere."""
    try:
        display(df)  # noqa: F821 (only defined in IPython/Jupyter)
    except NameError:
        print(df)


def load_wgms_metadata(folder=WGMS_FOLDER):
    """Read every regional *_metadata.csv into one clean DataFrame.

    Call this once and pass the result to regional_mass_balance(metadata=...)
    if you analyse several regions, so the files are only read once.
    """
    folder = Path(folder)
    parts = []
    for file in folder.glob("*_metadata.csv"):
        df = pd.read_csv(file)
        required = {"outline_id", "latitude", "longitude", "area_km2"}
        if not required.issubset(df.columns):
            continue
        for col in ["latitude", "longitude", "area_km2"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["region"] = file.stem.replace("_metadata", "")
        parts.append(df)

    if not parts:
        raise FileNotFoundError(f"No *_metadata.csv files found in {folder}")

    metadata = pd.concat(parts, ignore_index=True)
    return metadata.dropna(subset=["latitude", "longitude", "area_km2"])


def select_glaciers(metadata, polygon):
    """Keep glaciers whose centroid lies inside the polygon."""
    inside = metadata.apply(
        lambda r: polygon.covers(Point(r["longitude"], r["latitude"])), axis=1
    )
    return (
        metadata.loc[inside]
        .drop_duplicates(subset=["region", "outline_id"])
        .copy()
    )


def load_mass_series(glaciers, folder=WGMS_FOLDER):
    """Long-format annual mass change (m w.e.) for the selected glaciers."""
    folder = Path(folder)
    parts = []

    for region, group in glaciers.groupby("region"):
        mass_file = folder / f"{region}_mwe.csv"
        if not mass_file.exists():
            print(f"Missing mass file: {mass_file.name}")
            continue

        mass = pd.read_csv(mass_file)
        year_cols = [c for c in mass.columns if str(c).isdigit()]
        if not year_cols:
            continue

        selected = group[["outline_id", "area_km2"]].merge(
            mass[["outline_id"] + year_cols], on="outline_id", how="inner"
        )
        if selected.empty:
            continue

        long = selected.melt(
            id_vars=["outline_id", "area_km2"],
            value_vars=year_cols,
            var_name="year",
            value_name="mass_change_mwe",
        )
        long["year"] = pd.to_numeric(long["year"])
        long["mass_change_mwe"] = pd.to_numeric(
            long["mass_change_mwe"], errors="coerce"
        )
        parts.append(long)

    if not parts:
        raise ValueError(
            "No matching annual mass-change records found. "
            "Check the folder path and regional CSV files."
        )

    data = pd.concat(parts, ignore_index=True)
    data = data.dropna(subset=["mass_change_mwe", "area_km2"])
    return data[data["area_km2"] > 0]


def annual_balance(data):
    """Area-weighted and simple mean annual mass balance per year."""
    data = data.assign(weighted_change=data["mass_change_mwe"] * data["area_km2"])
    annual = data.groupby("year").agg(
        weighted_sum=("weighted_change", "sum"),
        area_with_data=("area_km2", "sum"),
        glaciers_with_data=("outline_id", "nunique"),
        mean_mwe=("mass_change_mwe", "mean"),
    )
    annual["area_weighted_mwe"] = annual["weighted_sum"] / annual["area_with_data"]
    annual.index = annual.index.astype(int)
    annual.index.name = "year"
    return annual


def regional_mass_balance(
    coords,
    region_name="Region",
    folder=WGMS_FOLDER,
    metadata=None,
    plot=True,
    verbose=True,
):
    """Run the full workflow for one rough region.

    Parameters
    ----------
    coords : list of (lon, lat) tuples outlining the region (closing point optional).
    region_name : label used in titles, e.g. "Karakoram".
    folder : path to the WGMS glacier folder.
    metadata : optional pre-loaded output of load_wgms_metadata().
    plot : draw the map and annual bar chart.
    verbose : print and display summary tables.

    Returns
    -------
    glaciers : DataFrame of glaciers inside the polygon.
    annual : DataFrame of annual mass balance (area_weighted_mwe, mean_mwe, ...).
    """
    polygon = Polygon(coords)
    if metadata is None:
        metadata = load_wgms_metadata(folder)

    glaciers = select_glaciers(metadata, polygon)
    if glaciers.empty:
        raise ValueError(f"No glaciers found inside the {region_name} polygon.")

    if verbose:
        print(f"Glaciers inside {region_name} boundary:", len(glaciers))
        print(
            "Total inventory glacier area (km²):",
            round(glaciers["area_km2"].sum(), 2),
        )
        _show(
            glaciers.sort_values("area_km2", ascending=False)[
                ["region", "outline_id", "latitude", "longitude", "area_km2"]
            ]
        )

    if plot:
        fig, ax = plt.subplots(figsize=(9, 8))
        x, y = polygon.exterior.xy
        ax.plot(x, y, linewidth=2, label=f"Approximate {region_name} boundary")
        ax.scatter(
            glaciers["longitude"], glaciers["latitude"],
            s=10, alpha=0.6, label="Selected glacier centroids",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"WGMS glacier centroids in {region_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        plt.show()

    data = load_mass_series(glaciers, folder)
    annual = annual_balance(data)

    if verbose:
        _show(
            annual[
                ["area_weighted_mwe", "mean_mwe",
                 "glaciers_with_data", "area_with_data"]
            ]
        )

    if plot:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(
            annual.index, annual["area_weighted_mwe"],
            label="Area-weighted mass balance",
        )
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xlabel("Hydrological year")
        ax.set_ylabel("Mass balance (m w.e.)")
        ax.set_title(f"Annual Mass Balance of Glaciers in {region_name}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return glaciers, annual


if __name__ == "__main__":
    meta = load_wgms_metadata()
    _, karakoram_annual = regional_mass_balance(KARAKORAM, "Karakoram", metadata=meta)
    _, swiss_alps_annual = regional_mass_balance(SWISS_ALPS, "Swiss Alps", metadata=meta)