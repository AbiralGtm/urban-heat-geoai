"""Attach one row of raster features to every hexagon.

Each raster is grouped by hexagon with a rasterised copy of the grid, so a pixel is
counted for exactly one cell. Continuous layers give a mean and a standard deviation,
land cover gives the share of each class, and population is summed at its own 100 m
resolution. Hexagons whose surface temperature is mostly gap are dropped.

The rasters are read from the directory in config.yaml; a layer that has not been
exported yet is skipped with a warning instead of failing the run.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parents[1]

# Continuous layers: mean and spread per hexagon.
CONTINUOUS = ["lst", "ndvi", "dem", "slope"]

# ESA WorldCover v200 class codes, kept as readable names.
WORLDCOVER_CLASSES = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "cropland",
    50: "builtup",
    60: "bare",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def raster_path(raw_dir: Path, layer: str) -> Path | None:
    """Earth Engine writes kathmandu_<layer>.tif; accept <layer>.tif as well."""
    for name in (f"kathmandu_{layer}.tif", f"{layer}.tif"):
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    return None


def zone_index(grid: gpd.GeoDataFrame, src: rasterio.DatasetReader) -> np.ndarray:
    """Burn the hexagons onto the raster grid; 0 means no hexagon covers the pixel."""
    shapes = ((geom, i + 1) for i, geom in enumerate(grid.to_crs(src.crs).geometry))
    return rasterize(
        shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="int32",
    )


def read_valid(src: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    """Return the band as float plus a mask of the pixels that carry a value.

    The Earth Engine exports mark gaps with NaN and carry no nodata tag, so both are
    treated as missing.
    """
    band = src.read(1).astype("float64")
    valid = np.isfinite(band)
    if src.nodata is not None and np.isfinite(src.nodata):
        valid &= band != src.nodata
    return band, valid


def continuous_stats(grid: gpd.GeoDataFrame, path: Path, layer: str) -> pd.DataFrame:
    with rasterio.open(path) as src:
        zones = zone_index(grid, src)
        band, valid = read_valid(src)

    n_zones = len(grid)
    keep = (zones > 0) & valid
    index = zones[keep] - 1
    values = band[keep]

    counts = np.bincount(index, minlength=n_zones)
    total = np.bincount(zones[zones > 0] - 1, minlength=n_zones)
    sums = np.bincount(index, weights=values, minlength=n_zones)
    sums_sq = np.bincount(index, weights=values**2, minlength=n_zones)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
        variance = np.where(counts > 0, sums_sq / counts - mean**2, np.nan)

    frame = pd.DataFrame(
        {
            f"{layer}_mean": mean,
            f"{layer}_std": np.sqrt(np.clip(variance, 0, None)),
            f"{layer}_valid_share": np.where(total > 0, counts / np.maximum(total, 1), np.nan),
        },
        index=grid["h3_id"],
    )
    return frame


def worldcover_shares(grid: gpd.GeoDataFrame, path: Path) -> pd.DataFrame:
    with rasterio.open(path) as src:
        zones = zone_index(grid, src)
        band, valid = read_valid(src)

    n_zones = len(grid)
    keep = (zones > 0) & valid
    index = zones[keep] - 1
    codes = band[keep].astype("int32")

    totals = np.bincount(index, minlength=n_zones).astype("float64")
    shares = {}
    for code, name in WORLDCOVER_CLASSES.items():
        counts = np.bincount(index[codes == code], minlength=n_zones)
        with np.errstate(invalid="ignore", divide="ignore"):
            shares[f"cover_{name}"] = np.where(totals > 0, counts / totals, np.nan)

    frame = pd.DataFrame(shares, index=grid["h3_id"])
    # Classes that never appear in this valley carry no information for the model.
    return frame.loc[:, frame.fillna(0).sum() > 0]


def population_sum(grid: gpd.GeoDataFrame, path: Path) -> pd.DataFrame:
    """WorldPop is a count per 100 m cell, so the pixels are summed, not averaged."""
    with rasterio.open(path) as src:
        zones = zone_index(grid, src)
        band, valid = read_valid(src)

    keep = (zones > 0) & valid
    sums = np.bincount(zones[keep] - 1, weights=band[keep], minlength=len(grid))
    return pd.DataFrame({"population": sums}, index=grid["h3_id"])


def main() -> None:
    config = load_config()
    raw_dir = ROOT / config["paths"]["raw_raster_dir"]

    grid = gpd.read_file(ROOT / config["paths"]["grid"], layer="grid")
    print(f"grid cells: {len(grid)}")

    frames = []
    for layer in CONTINUOUS:
        path = raster_path(raw_dir, layer)
        if path is None:
            print(f"warning: {layer} raster not found in {raw_dir}, skipping")
            continue
        frames.append(continuous_stats(grid, path, layer))
        print(f"{layer}: done")

    cover_path = raster_path(raw_dir, "worldcover")
    if cover_path is None:
        print("warning: worldcover raster not found, skipping land-cover shares")
    else:
        frames.append(worldcover_shares(grid, cover_path))
        print("worldcover: done")

    pop_path = raster_path(raw_dir, "pop")
    if pop_path is None:
        print("warning: population raster not found, skipping population")
    else:
        frames.append(population_sum(grid, pop_path))
        print("pop: done")

    if not frames:
        raise SystemExit("no rasters found; run gee/heat_kathmandu.js and download the exports")

    features = pd.concat(frames, axis=1).reset_index()
    features = grid.drop(columns="geometry").merge(features, on="h3_id", how="left")
    features["pop_density"] = features["population"] / features["area_km2"]

    out_path = ROOT / config["paths"]["features"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    target = config["model"]["target"]
    print(f"rows: {len(features)}, columns: {features.shape[1]}")
    if target in features:
        print(f"{target}: min {features[target].min():.2f}, mean {features[target].mean():.2f}, max {features[target].max():.2f}")
    print(f"written: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
