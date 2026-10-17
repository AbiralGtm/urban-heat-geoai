"""Add road density and distance to the nearest waterway to the feature table.

OpenStreetMap is downloaded once for the study box and cached as GeoPackages, so a
re-run needs no network. Road length is clipped to each hexagon and divided by its area;
the waterway distance is measured from the hexagon centroid. Both are computed in the
projected CRS, so lengths are metres.

Run after 03_raster_features.py:
    python src/04_osm_features.py
    python src/04_osm_features.py --skip   # leave the table unchanged
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "raw" / "osm"

# Motorways through residential streets: the classes that carry most of the sealed
# surface. Tracks and footpaths are left out.
ROAD_FILTER = (
    '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]'
)
WATER_TAGS = {"waterway": ["river", "stream", "canal"]}


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def cached(name: str) -> gpd.GeoDataFrame | None:
    cache = CACHE_DIR / f"{name}.gpkg"
    if cache.exists():
        print(f"{name}: read from cache")
        return gpd.read_file(cache)
    return None


def store(edges: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    edges = edges[["geometry"]].reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    edges.to_file(CACHE_DIR / f"{name}.gpkg", driver="GPKG")
    print(f"{name}: {len(edges)} segments cached")
    return edges


def with_mirrors(name: str, fetch) -> gpd.GeoDataFrame:
    """Call fetch() against each Overpass mirror until one answers.

    The public endpoint refuses queries when its workers are busy, so the mirrors are
    tried in turn with a widening pause between attempts.
    """
    import osmnx as ox

    # The public Overpass endpoint refuses queries when its workers are busy, so the
    # mirrors are tried in turn with a widening pause between attempts.
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.osm.ch/api/interpreter",
    ]
    for attempt, url in enumerate(endpoints):
        ox.settings.overpass_url = url
        try:
            return fetch(ox)
        except Exception as error:  # network or server side, not a bug in the query
            print(f"{name}: {url} failed ({error.__class__.__name__}), retrying")
            time.sleep(5 * (attempt + 1))

    raise SystemExit(
        f"could not download {name} from any Overpass mirror; "
        "try again later or run src/04_osm_features.py --skip"
    )


def download_roads(config: dict) -> gpd.GeoDataFrame:
    """Road centrelines, taken as a routable network so that the geometry is clean."""
    edges = cached("roads")
    if edges is not None:
        return edges

    west, south, east, north = config["bbox"]
    print("roads: downloading from OpenStreetMap")

    def fetch(ox):
        graph = ox.graph_from_bbox(
            (west, south, east, north),
            custom_filter=ROAD_FILTER,
            retain_all=True,
            simplify=True,
        )
        return ox.graph_to_gdfs(graph, nodes=False, edges=True)

    return store(with_mirrors("roads", fetch), "roads")


def download_waterways(config: dict) -> gpd.GeoDataFrame:
    """Rivers, streams and canals.

    These are plain geometries rather than a routable network, so they are queried as
    features; asking for a graph returns nothing because they carry no highway tag.
    """
    edges = cached("waterways")
    if edges is not None:
        return edges

    west, south, east, north = config["bbox"]
    print("waterways: downloading from OpenStreetMap")

    def fetch(ox):
        features = ox.features_from_bbox((west, south, east, north), tags=WATER_TAGS)
        return features[features.geometry.type.isin(["LineString", "MultiLineString"])]

    return store(with_mirrors("waterways", fetch), "waterways")


def road_density(grid: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    """Metres of road per square kilometre of hexagon."""
    pieces = gpd.overlay(
        grid[["h3_id", "geometry"]],
        roads[["geometry"]].assign(_road=1),
        how="intersection",
        keep_geom_type=False,
    )
    pieces["length_m"] = pieces.length
    lengths = pieces.groupby("h3_id")["length_m"].sum()
    return (lengths / grid.set_index("h3_id")["area_km2"]).reindex(grid["h3_id"]).fillna(0.0)


def water_distance(grid: gpd.GeoDataFrame, water: gpd.GeoDataFrame) -> pd.Series:
    """Metres from the hexagon centroid to the nearest river, stream or canal."""
    centroids = grid[["h3_id"]].copy()
    centroids = gpd.GeoDataFrame(
        centroids, geometry=grid.geometry.centroid, crs=grid.crs
    )
    joined = gpd.sjoin_nearest(
        centroids, water[["geometry"]], how="left", distance_col="water_distance_m"
    )
    # A centroid equally near two segments is returned twice; keep the first match.
    joined = joined.drop_duplicates(subset="h3_id").set_index("h3_id")
    return joined["water_distance_m"].reindex(grid["h3_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", action="store_true", help="leave the feature table unchanged")
    args = parser.parse_args()

    config = load_config()
    features_path = ROOT / config["paths"]["features"]
    features = pd.read_csv(features_path)

    if args.skip:
        print("skipped; the model will run without the OpenStreetMap features")
        return

    grid = gpd.read_file(ROOT / config["paths"]["grid"], layer="grid")
    grid = grid[grid["h3_id"].isin(features["h3_id"])].reset_index(drop=True)
    grid = grid.to_crs(config["crs"]["projected"])

    roads = download_roads(config).to_crs(grid.crs)
    water = download_waterways(config).to_crs(grid.crs)

    osm = pd.DataFrame(
        {
            "h3_id": grid["h3_id"],
            "road_density_m_km2": road_density(grid, roads).to_numpy(),
            "water_distance_m": water_distance(grid, water).to_numpy(),
        }
    )

    features = features.drop(columns=osm.columns.drop("h3_id"), errors="ignore")
    features = features.merge(osm, on="h3_id", how="left")
    features.to_csv(features_path, index=False)

    print(f"road density: median {osm['road_density_m_km2'].median():.0f} m/km2, "
          f"max {osm['road_density_m_km2'].max():.0f} m/km2")
    print(f"waterway distance: median {osm['water_distance_m'].median():.0f} m, "
          f"max {osm['water_distance_m'].max():.0f} m")
    print(f"written: {features_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
