"""Grid mountain regions into ERA5-aligned boxes, run WGMS mass-balance
analysis, save reproducible CSV datasets, and build an interactive Folium map.

The CSV files are the source of truth.

Functions
---------
run_box_analysis()
    Runs the WGMS analysis and saves all datasets. If saved CSVs already exist
    for the same settings it loads them instead of rerunning (force=True to
    override).

build_map_from_csv()
    Reconstructs the interactive map using saved CSV files only.

Requires:
    pip install folium pandas numpy

    wgms_exploration.py in the same project, providing:
    WGMS_FOLDER, load_wgms_metadata, regional_mass_balance
"""

from dataclasses import dataclass
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from branca.element import Element, MacroElement, Template
from folium.plugins import FastMarkerCluster

from wgms_exploration import (
    WGMS_FOLDER,
    load_wgms_metadata,
    regional_mass_balance,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_REGIONS = {
    "Karakoram":   (72.5, 79.0, 34.2, 38.0),
    "Pamir":       (71.0, 75.5, 37.0, 39.5),
    "Kunlun":      (79.0, 95.0, 34.0, 37.0),
    "Himalaya":    (75.0, 97.0, 27.0, 34.0),
    "Tien Shan":   (69.0, 82.0, 40.0, 44.0),

    "Scandinavia": (5.0, 9.0, 59.5, 70.0),
    "Svalbard":    (10.0, 35.0, 76.5, 80.5),
    "Caucasus":    (37.0, 49.0, 41.0, 44.0),
    "Alaska":      (-153.0, -130.0, 58.0, 64.0),
    "Patagonia":   (-75.0, -72.0, -52.0, -46.0),
    "New Zealand": (167.0, 172.0, -46.0, -43.0),
    "Swiss Alps":  (5.9, 10.5, 45.7, 47.8),
}

DEFAULT_CLASSES = [
    (-0.50, "Strong loss (< -0.5)", "#b2182b"),
    (-0.25, "Moderate loss (-0.5 to -0.25)", "#ef8a62"),
    (-0.10, "Slight loss (-0.25 to -0.1)", "#fddbc7"),
    (0.10, "Near-balanced (-0.1 to +0.1)", "#ffffff"),
    (np.inf, "Gain (> +0.1)", "#2166ac"),
]

DEFAULT_DATA_FOLDER = "results/data"
DEFAULT_HTML = "results/figures/high_mountain_asia.html"


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def classify(value, classes=DEFAULT_CLASSES):
    """Return classification label and colour for mean annual balance."""

    for upper, label, colour in classes:
        if value < upper:
            return label, colour

    return classes[-1][1], classes[-1][2]


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class BoxAnalysis:
    stats: pd.DataFrame
    annual: dict
    boxes: dict
    region_of: dict
    map: folium.Map


# --------------------------------------------------------------------------
# Grid generation
# --------------------------------------------------------------------------

def build_grid(regions, step=0.25):
    """
    Generate ERA5-aligned rectangular boxes.

    Returns:
        boxes: {box_id: [(lon, lat), ...]}
        region_of: {box_id: region_name}
    """

    half = step / 2

    boxes = {}
    region_of = {}
    seen = set()

    i = 1

    for name, (lon_min, lon_max, lat_min, lat_max) in regions.items():

        lons = np.round(
            np.arange(
                np.ceil(lon_min / step) * step,
                lon_max,
                step,
            ),
            4,
        )

        lats = np.round(
            np.arange(
                np.ceil(lat_min / step) * step,
                lat_max,
                step,
            ),
            4,
        )

        for lat in lats:
            for lon in lons:

                key = (lon, lat)

                if key in seen:
                    continue

                seen.add(key)

                boxes[i] = [
                    (lon - half, lat - half),
                    (lon + half, lat - half),
                    (lon + half, lat + half),
                    (lon - half, lat + half),
                    (lon - half, lat - half),
                ]

                region_of[i] = name
                i += 1

    return boxes, region_of


# --------------------------------------------------------------------------
# CSV export
# --------------------------------------------------------------------------

def save_analysis_data(
    stats,
    annual,
    boxes,
    region_of,
    metadata,
    regions,
    start,
    end,
    step,
    classes,
    show_empty,
    min_years,
    min_glaciers,
    wgms_folder,
    data_folder=DEFAULT_DATA_FOLDER,
):
    """
    Save every dataset needed to reconstruct the map.

    No geometry or analysis information is dependent on the HTML.
    """

    folder = Path(data_folder)
    folder.mkdir(parents=True, exist_ok=True)

    # 1. Mass-balance summary: one row per analysed box.
    stats.to_csv(folder / "mass_balance_summary.csv")

    # 2. Annual mass-balance data: box + year MultiIndex.
    annual_data = pd.concat(
        annual,
        names=["box", "year"],
    )

    annual_data.to_csv(
        folder / "annual_mass_balance.csv"
    )

    # 3. Exact polygon coordinates for every box.
    geometry = []

    for bid, coords in boxes.items():

        for vertex, (lon, lat) in enumerate(coords):

            geometry.append({
                "box": bid,
                "region": region_of[bid],
                "vertex": vertex,
                "longitude": lon,
                "latitude": lat,
            })

    pd.DataFrame(geometry).to_csv(
        folder / "box_coordinates.csv",
        index=False,
    )

    # 4. Original region boundaries.
    region_data = []

    for name, bounds in regions.items():

        lon_min, lon_max, lat_min, lat_max = bounds

        region_data.append({
            "region": name,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        })

    pd.DataFrame(region_data).to_csv(
        folder / "region_bounds.csv",
        index=False,
    )

    # 5. WGMS glacier locations.
    glacier_locations = metadata[
        ["longitude", "latitude"]
    ].copy()

    glacier_locations.to_csv(
        folder / "glacier_locations.csv",
        index=False,
    )

    # 6. Classification thresholds and colours.
    pd.DataFrame(
        classes,
        columns=["upper_bound", "label", "colour"],
    ).to_csv(
        folder / "classification.csv",
        index=False,
    )

    # 7. Original analysis settings.
    config = pd.DataFrame([
        {"key": "start", "value": start},
        {"key": "end", "value": end},
        {"key": "step", "value": step},
        {"key": "show_empty", "value": show_empty},
        {"key": "min_years", "value": min_years},
        {"key": "min_glaciers", "value": min_glaciers},
        {"key": "wgms_folder", "value": str(wgms_folder)},
    ])

    config.to_csv(
        folder / "analysis_config.csv",
        index=False,
    )

    print(f"\nCSV datasets saved to: {folder.resolve()}")


# --------------------------------------------------------------------------
# Cache check
# --------------------------------------------------------------------------

REQUIRED_CSVS = [
    "mass_balance_summary.csv",
    "annual_mass_balance.csv",
    "box_coordinates.csv",
    "region_bounds.csv",
    "glacier_locations.csv",
    "classification.csv",
    "analysis_config.csv",
]


def _cache_matches(
    data_folder,
    regions,
    step,
    start,
    end,
    min_years,
    min_glaciers,
    wgms_folder,
):
    """
    Decide whether the saved CSVs can be reused for these settings.

    Returns (True, "") or (False, reason). Settings that change which boxes
    are analysed or over which years (regions, step, start, end, min_years,
    min_glaciers, WGMS folder) must all match. Classes do not matter because
    they are applied when the map is drawn.
    """

    folder = Path(data_folder)

    if any(not (folder / name).exists() for name in REQUIRED_CSVS):
        return False, "no complete set of saved CSVs found"

    cfg = (
        pd.read_csv(folder / "analysis_config.csv", dtype=str)
        .set_index("key")["value"]
        .to_dict()
    )

    try:
        saved = {
            "start": int(cfg["start"]),
            "end": int(cfg["end"]),
            "step": float(cfg["step"]),
            "min_years": int(cfg["min_years"]),
            "min_glaciers": int(cfg["min_glaciers"]),
            "wgms_folder": cfg["wgms_folder"],
        }
    except KeyError as missing:
        return False, f"saved config is from an older version (no {missing})"

    wanted = {
        "start": int(start),
        "end": int(end),
        "step": float(step),
        "min_years": int(min_years),
        "min_glaciers": int(min_glaciers),
        "wgms_folder": str(wgms_folder),
    }

    for key, value in wanted.items():
        same = (
            abs(saved[key] - value) < 1e-9
            if key == "step"
            else saved[key] == value
        )
        if not same:
            return False, f"{key} changed ({saved[key]} -> {value})"

    region_df = pd.read_csv(folder / "region_bounds.csv").set_index("region")

    if set(region_df.index) != set(regions):
        return False, "regions changed"

    columns = ["lon_min", "lon_max", "lat_min", "lat_max"]

    for name, bounds in regions.items():
        if not np.allclose(region_df.loc[name, columns].to_numpy(float), bounds):
            return False, f"bounds of {name} changed"

    return True, ""


# --------------------------------------------------------------------------
# Main WGMS analysis
# --------------------------------------------------------------------------

def run_box_analysis(
    regions=None,
    step=0.25,
    start=2000,
    end=2025,
    min_years=10,
    min_glaciers=1,
    classes=None,
    show_empty=False,
    save_html=DEFAULT_HTML,
    save_data=True,
    data_folder=DEFAULT_DATA_FOLDER,
    folder=WGMS_FOLDER,
    metadata=None,
    verbose=True,
    force=False,
    glacier_points="none",
    label_min_zoom=8,
):
    """
    Run WGMS mass-balance analysis for every eligible ERA5 box.

    CSV datasets are saved independently of the HTML map.

    If complete CSVs from the same settings already exist in data_folder, the
    analysis is skipped and the results are loaded from them. Pass force=True
    to rerun anyway. The cache is not used when metadata is passed in.

    label_min_zoom is the map zoom level at which the box numbers appear
    (8 shows them when a box is about 45 px wide; raise it to show them later).

    glacier_points controls the WGMS glacier layer in the HTML, which is by far
    its biggest part: "none" (default, no layer), "clustered" (compact), or
    "all" (one marker per glacier, tens of MB for high mountain Asia).
    """

    classes = DEFAULT_CLASSES if classes is None else classes

    if regions is None:
        regions = DEFAULT_REGIONS.copy()

    elif (
        isinstance(regions, (tuple, list))
        and len(regions) == 4
        and np.isscalar(regions[0])
    ):
        regions = {"Region": tuple(regions)}

    # Reuse saved results instead of rerunning the whole analysis.
    if save_data and not force and metadata is None:

        reusable, reason = _cache_matches(
            data_folder,
            regions,
            step,
            start,
            end,
            min_years,
            min_glaciers,
            folder,
        )

        if reusable:

            if verbose:
                print(
                    f"Found saved results in {Path(data_folder).resolve()}; "
                    "loading them instead of rerunning "
                    "(pass force=True to recompute)."
                )

            return build_map_from_csv(
                data_folder=data_folder,
                classes=classes,
                show_empty=show_empty,
                output=save_html,
                glacier_points=glacier_points,
                label_min_zoom=label_min_zoom,
            )

        if verbose:
            print(f"Running analysis: {reason}.")

    boxes, region_of = build_grid(regions, step)

    if verbose:
        print(
            f"{len(boxes)} boxes of {step} degrees "
            f"across {len(regions)} regions"
        )

    # Load glacier metadata once.
    meta = (
        load_wgms_metadata(folder)
        if metadata is None
        else metadata
    )

    required = {"longitude", "latitude"}

    if not required.issubset(meta.columns):
        raise ValueError(
            "WGMS metadata must contain longitude and latitude columns."
        )

    # Keep glaciers inside the requested regions.
    in_any = np.zeros(len(meta), dtype=bool)

    for lon_min, lon_max, lat_min, lat_max in regions.values():

        in_any |= (
            meta["longitude"].between(lon_min, lon_max)
            & meta["latitude"].between(lat_min, lat_max)
        ).to_numpy()

    meta = meta.loc[in_any].copy()

    glon = meta["longitude"].to_numpy()
    glat = meta["latitude"].to_numpy()

    # Run mass-balance analysis.
    box_annual = {}
    rows = []

    for bid, coords in boxes.items():

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]

        mask = (
            (glon >= min(xs))
            & (glon <= max(xs))
            & (glat >= min(ys))
            & (glat <= max(ys))
        )

        if not mask.any():
            continue

        try:

            _, annual = regional_mass_balance(
                coords,
                f"Box {bid}",
                folder=folder,
                metadata=meta.loc[mask],
                plot=False,
                verbose=False,
            )

        except ValueError:
            continue

        window = annual.loc[start:end]

        s = window["area_weighted_mwe"].dropna()

        n_glac = (
            int(window["glaciers_with_data"].max())
            if len(window)
            else 0
        )

        if len(s) < min_years or n_glac < min_glaciers:
            continue

        box_annual[bid] = annual

        mean = s.mean()

        sem = (
            s.std(ddof=1) / np.sqrt(len(s))
            if len(s) > 1
            else np.nan
        )

        label, _ = classify(mean, classes)

        rows.append({
            "box": bid,
            "region": region_of[bid],
            "lon": float(np.mean(xs[:-1])),
            "lat": float(np.mean(ys[:-1])),
            "mean_mwe_yr": mean,
            "ci95_low": mean - 1.96 * sem,
            "ci95_high": mean + 1.96 * sem,
            "class": label,
            "cumulative_mwe": s.sum(),
            "years": len(s),
            "glaciers": n_glac,
        })

    if not rows:
        raise ValueError(
            "No boxes have enough data. Try a lower min_years, "
            "a wider start-end window, a larger step, or check "
            "the regions and WGMS folder."
        )

    stats = pd.DataFrame(rows).set_index("box")

    if verbose:
        print(f"\n{len(stats)} boxes with usable data:")
        print(
            stats.groupby("region")
            .size()
            .rename("boxes")
            .to_string()
        )

    # Save all source datasets.
    if save_data:

        save_analysis_data(
            stats=stats,
            annual=box_annual,
            boxes=boxes,
            region_of=region_of,
            metadata=meta,
            regions=regions,
            start=start,
            end=end,
            step=step,
            classes=classes,
            show_empty=show_empty,
            min_years=min_years,
            min_glaciers=min_glaciers,
            wgms_folder=folder,
            data_folder=data_folder,
        )

    # Construct the map.
    m = _draw_map(
        stats,
        boxes,
        regions,
        meta,
        start,
        end,
        classes,
        show_empty,
        glacier_points,
        label_min_zoom,
    )

    if save_html:

        output = Path(save_html)
        output.parent.mkdir(parents=True, exist_ok=True)

        m.save(str(output))

        if verbose:
            print(f"\nMap saved to: {output.resolve()}")

    return BoxAnalysis(
        stats=stats,
        annual=box_annual,
        boxes=boxes,
        region_of=region_of,
        map=m,
    )


# --------------------------------------------------------------------------
# Rebuild map using CSV files ONLY
# --------------------------------------------------------------------------

def _window_stats(annual, base_stats, start, end, min_years, min_glaciers):
    """
    Recompute the per-box statistics for another start-end window from the
    saved annual series. Only boxes that were saved can appear.
    """

    rows = []

    for bid, a in annual.items():

        if bid not in base_stats.index:
            continue

        window = a.loc[start:end]
        s = window["area_weighted_mwe"].dropna()

        n_glac = (
            int(window["glaciers_with_data"].max())
            if len(window)
            else 0
        )

        if len(s) < min_years or n_glac < min_glaciers:
            continue

        mean = s.mean()

        sem = (
            s.std(ddof=1) / np.sqrt(len(s))
            if len(s) > 1
            else np.nan
        )

        rows.append({
            "box": bid,
            "region": base_stats.at[bid, "region"],
            "lon": base_stats.at[bid, "lon"],
            "lat": base_stats.at[bid, "lat"],
            "mean_mwe_yr": mean,
            "ci95_low": mean - 1.96 * sem,
            "ci95_high": mean + 1.96 * sem,
            "cumulative_mwe": s.sum(),
            "years": len(s),
            "glaciers": n_glac,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("box")


def build_map_from_csv(
    data_folder=DEFAULT_DATA_FOLDER,
    start=None,
    end=None,
    classes=None,
    show_empty=None,
    output=DEFAULT_HTML,
    glacier_points="none",
    label_min_zoom=8,
):
    """
    Reconstruct the Folium map using previously saved CSV files.

    Does NOT load WGMS data.
    Does NOT call regional_mass_balance().
    Does NOT rerun the analysis.

    start/end default to the saved window. If you give a different window,
    the statistics are recomputed from the saved annual series (so the colours
    and numbers match the label). Only boxes saved by the original run can
    appear, so a window can narrow the result but not recover boxes that the
    original run filtered out.

    Classes are always applied fresh, so you can change them freely.
    """

    folder = Path(data_folder)

    missing = [
        filename
        for filename in REQUIRED_CSVS
        if not (folder / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing CSV datasets:\n"
            + "\n".join(missing)
            + "\nRun run_box_analysis(save_data=True) first."
        )

    # 1. Load mass-balance summary.
    stats = pd.read_csv(
        folder / "mass_balance_summary.csv",
        index_col="box",
    )

    stats.index = stats.index.astype(int)

    # 2. Load exact polygon coordinates.
    geometry = pd.read_csv(folder / "box_coordinates.csv")

    boxes = {}
    region_of = {}

    for bid, group in geometry.groupby("box", sort=False):

        bid = int(bid)

        group = group.sort_values("vertex")

        boxes[bid] = list(
            zip(
                group["longitude"],
                group["latitude"],
            )
        )

        region_of[bid] = group["region"].iloc[0]

    # 3. Load original region boundaries.
    region_df = pd.read_csv(folder / "region_bounds.csv")

    regions = {
        row["region"]: (
            row["lon_min"],
            row["lon_max"],
            row["lat_min"],
            row["lat_max"],
        )
        for _, row in region_df.iterrows()
    }

    # 4. Load glacier locations.
    meta = pd.read_csv(folder / "glacier_locations.csv")

    # 5. Load original analysis settings.
    config = (
        pd.read_csv(folder / "analysis_config.csv", dtype=str)
        .set_index("key")["value"]
        .to_dict()
    )

    saved_start = int(config["start"])
    saved_end = int(config["end"])

    start = saved_start if start is None else int(start)
    end = saved_end if end is None else int(end)

    if show_empty is None:
        show_empty = str(config["show_empty"]).lower() == "true"

    # 6. Load classification settings.
    if classes is None:

        class_df = pd.read_csv(folder / "classification.csv")

        classes = [
            (
                float(row.upper_bound),
                row.label,
                row.colour,
            )
            for row in class_df.itertuples(index=False)
        ]

    # 7. Load annual data (also available for later statistics).
    annual_data = pd.read_csv(
        folder / "annual_mass_balance.csv",
        index_col=["box", "year"],
    )

    annual = {}

    for bid, group in annual_data.groupby(level="box"):

        annual[int(bid)] = group.droplevel("box")

    # 8. A different window means different statistics.
    if (start, end) != (saved_start, saved_end):

        stats = _window_stats(
            annual,
            stats,
            start,
            end,
            int(config.get("min_years", 1)),
            int(config.get("min_glaciers", 1)),
        )

        if stats.empty:
            raise ValueError(
                f"No saved boxes have enough data in {start}-{end}."
            )

    # 9. Classes are applied fresh so edited classes take effect.
    stats["class"] = [
        classify(value, classes)[0]
        for value in stats["mean_mwe_yr"]
    ]

    # 10. Reconstruct the map.
    m = _draw_map(
        stats,
        boxes,
        regions,
        meta,
        start,
        end,
        classes,
        show_empty,
        glacier_points,
        label_min_zoom,
    )

    # 11. Save HTML.
    if output:

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        m.save(str(output))

        print(
            f"Map generated from CSV data: "
            f"{output.resolve()}"
        )

    return BoxAnalysis(
        stats=stats,
        annual=annual,
        boxes=boxes,
        region_of=region_of,
        map=m,
    )


# --------------------------------------------------------------------------
# Zoom-dependent box numbers
# --------------------------------------------------------------------------

class _ZoomLabels(MacroElement):
    """
    Hide the box numbers while zoomed out so they do not pile up into a black
    mass. Adds a CSS class to the map container when the zoom level is below
    min_zoom; the CSS then hides every .box-label element.
    """

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var zoomMap = {{ this._parent.get_name() }};
            var minZoom = {{ this.min_zoom }};
            function updateBoxLabels() {
                zoomMap.getContainer().classList.toggle(
                    "hide-box-labels", zoomMap.getZoom() < minZoom
                );
            }
            zoomMap.on("zoomend", updateBoxLabels);
            updateBoxLabels();
        })();
        {% endmacro %}
    """)

    def __init__(self, min_zoom=8):
        super().__init__()
        self._name = "ZoomLabels"
        self.min_zoom = min_zoom


# --------------------------------------------------------------------------
# Folium map
# --------------------------------------------------------------------------

def _draw_map(
    stats,
    boxes,
    regions,
    meta,
    start,
    end,
    classes,
    show_empty,
    glacier_points="none",
    label_min_zoom=8,
):

    if glacier_points not in ("none", "clustered", "all"):
        raise ValueError(
            "glacier_points must be 'none', 'clustered' or 'all'."
        )

    # Open on the first region in the dict (keep Karakoram first). Centring
    # on the middle of all regions lands in Africa once distant regions such
    # as Alaska or New Zealand are included.
    first_lon_min, first_lon_max, first_lat_min, first_lat_max = next(
        iter(regions.values())
    )

    m = folium.Map(
        location=[
            (first_lat_min + first_lat_max) / 2,
            (first_lon_min + first_lon_max) / 2,
        ],
        zoom_start=5,
        tiles="OpenStreetMap",
    )

    m.fit_bounds(
        [
            [first_lat_min, first_lon_min],
            [first_lat_max, first_lon_max],
        ]
    )

    # Satellite imagery.
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/"
            "services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
    ).add_to(m)

    # Region boundaries.
    outline = folium.FeatureGroup(
        name="Region outlines"
    )

    for name, (
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    ) in regions.items():

        folium.Rectangle(
            bounds=[
                [lat_min, lon_min],
                [lat_max, lon_max],
            ],
            color="black",
            weight=1.5,
            dash_array="6",
            fill=False,
            tooltip=name,
        ).add_to(outline)

    outline.add_to(m)

    # Toggleable region layers.
    groups = {
        name: folium.FeatureGroup(name=name).add_to(m)
        for name in regions
    }

    # Optional empty boxes.
    if show_empty:

        for bid, coords in boxes.items():

            if bid in stats.index:
                continue

            folium.Polygon(
                locations=[
                    (lat, lon)
                    for lon, lat in coords
                ],
                color="grey",
                weight=0.4,
                fill=False,
            ).add_to(m)

    # Analysed boxes.
    for bid, row in stats.iterrows():

        bid = int(bid)

        coords = boxes[bid]
        group = groups[row["region"]]

        _, colour = classify(
            row["mean_mwe_yr"],
            classes,
        )

        tip = (
            f"<b>Box {bid}</b> ({row['region']})<br>"
            f"Centroid (lat, lon): "
            f"{row['lat']:.3f}, {row['lon']:.3f}<br>"
            f"<b>{row['class']}</b><br>"
            f"Mean {start}-{end}: "
            f"{row['mean_mwe_yr']:.3f} m w.e./yr<br>"
            f"95% CI: "
            f"{row['ci95_low']:.2f} to "
            f"{row['ci95_high']:.2f}<br>"
            f"Cumulative: "
            f"{row['cumulative_mwe']:.2f} m w.e.<br>"
            f"Years with data: {row['years']}<br>"
            f"Glaciers: {row['glaciers']}"
        )

        folium.Polygon(
            locations=[
                (lat, lon)
                for lon, lat in coords
            ],
            color="black",
            weight=1,
            fill=True,
            fill_color=colour,
            fill_opacity=0.8,
            tooltip=tip,
        ).add_to(group)

        # Box number.
        folium.Marker(
            location=[
                row["lat"],
                row["lon"],
            ],
            icon=folium.DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                class_name="box-label",
                html=(
                    '<div style="font-size:12px;'
                    'font-weight:bold;color:black;'
                    'text-align:center;'
                    'text-shadow:0 0 3px #fff,0 0 3px #fff;">'
                    f'{bid}</div>'
                ),
            ),
        ).add_to(group)

    # WGMS glacier locations. One marker per glacier makes the HTML tens of MB,
    # so this layer is off by default (the locations stay in the CSV).
    if glacier_points == "clustered":

        FastMarkerCluster(
            meta[["latitude", "longitude"]].to_numpy().tolist(),
            name="WGMS glaciers",
            show=False,
        ).add_to(m)

    elif glacier_points == "all":

        dots = folium.FeatureGroup(
            name="WGMS glaciers",
            show=False,
        )

        for lon, lat in zip(
            meta["longitude"],
            meta["latitude"],
        ):

            folium.CircleMarker(
                [lat, lon],
                radius=2,
                color="black",
                fill=True,
                fill_opacity=1,
                weight=0,
            ).add_to(dots)

        dots.add_to(m)

    # Legend.
    items = "".join(
        (
            '<div>'
            '<span style="display:inline-block;'
            'width:14px;height:14px;'
            f'background:{colour};'
            'border:1px solid #444;'
            'margin-right:6px;'
            'vertical-align:middle;">'
            '</span>'
            f'{label}'
            '</div>'
        )
        for _, label, colour in reversed(classes)
    )

    legend = (
        '<div style="position:fixed;'
        'bottom:30px;left:30px;z-index:9999;'
        'background:white;padding:8px 10px;'
        'border:1px solid #888;border-radius:4px;'
        'font-size:12px;line-height:1.5;">'
        f'<b>Mean annual balance {start}-{end}'
        '<br>(m w.e./yr)</b>'
        f'{items}'
        '</div>'
    )

    m.get_root().html.add_child(
        Element(legend)
    )

    # Box numbers only appear once zoomed in far enough.
    m.get_root().header.add_child(
        Element(
            "<style>.hide-box-labels .box-label "
            "{display:none;}</style>"
        )
    )

    m.add_child(_ZoomLabels(min_zoom=label_min_zoom))

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m