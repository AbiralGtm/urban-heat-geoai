"""Build the H3 hexagon grid that every later stage joins onto.

Reads the study box from config.yaml, fills it with H3 cells at the configured
resolution, and writes one GeoPackage with the cell id, its coarser parent (used as the
block for spatial cross-validation) and the cell area in square kilometres.
"""

from pathlib import Path

import geopandas as gpd
import h3
import yaml
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def cells_in_bbox(bbox: list[float], resolution: int) -> list[str]:
    """Return every H3 cell whose centre falls inside the study box."""
    west, south, east, north = bbox
    ring = [
        (south, west),
        (south, east),
        (north, east),
        (north, west),
        (south, west),
    ]
    poly = h3.LatLngPoly(ring)
    return list(h3.h3shape_to_cells(poly, resolution))


def cell_polygon(cell: str) -> Polygon:
    """H3 returns (lat, lng) pairs; shapely wants (lng, lat)."""
    boundary = h3.cell_to_boundary(cell)
    return Polygon([(lng, lat) for lat, lng in boundary])


def build_grid(config: dict) -> gpd.GeoDataFrame:
    resolution = config["grid"]["h3_resolution"]
    parent_resolution = config["grid"]["parent_resolution"]

    cells = cells_in_bbox(config["bbox"], resolution)
    grid = gpd.GeoDataFrame(
        {
            "h3_id": cells,
            "h3_parent": [h3.cell_to_parent(c, parent_resolution) for c in cells],
            "centroid_lat": [h3.cell_to_latlng(c)[0] for c in cells],
            "centroid_lon": [h3.cell_to_latlng(c)[1] for c in cells],
        },
        geometry=[cell_polygon(c) for c in cells],
        crs=config["crs"]["geographic"],
    )

    projected = grid.to_crs(config["crs"]["projected"])
    grid["area_km2"] = projected.area / 1e6
    return grid.sort_values("h3_id").reset_index(drop=True)


def main() -> None:
    config = load_config()
    grid = build_grid(config)

    out_path = ROOT / config["paths"]["grid"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_file(out_path, layer="grid", driver="GPKG")

    print(f"cells: {len(grid)}")
    print(f"blocks (resolution {config['grid']['parent_resolution']}): {grid['h3_parent'].nunique()}")
    print(f"mean cell area: {grid['area_km2'].mean():.3f} km2")
    print(f"written: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
