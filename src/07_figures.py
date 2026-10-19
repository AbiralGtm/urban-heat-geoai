"""Render the figures the README shows.

Two choropleths of the same hexagons: measured surface temperature, and exposure. Both
use a single hue running light to dark, classed on quantiles so the whole ramp is used,
with the class edges printed on the colour bar. The SHAP plot written by 05_model.py is
copied here as well, so every image in the README lives under docs/images/.
"""

import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "docs" / "images"

# Single-hue ramps, light to dark, matching the two layers on the web map.
HEAT_RAMP = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
EXPOSURE_RAMP = ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"]

INK = "#1b1f23"
MUTED = "#5b6570"


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def quantile_edges(values: np.ndarray, classes: int) -> np.ndarray:
    """Class edges at equal counts, nudged apart if two quantiles coincide."""
    edges = np.quantile(values, np.linspace(0, 1, classes + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def panel(ax, cells: gpd.GeoDataFrame, field: str, ramp: list[str], title: str, unit: str) -> None:
    values = cells[field].to_numpy()
    edges = quantile_edges(values, len(ramp))
    cmap = LinearSegmentedColormap.from_list(field, ramp, N=len(ramp))
    norm = BoundaryNorm(edges, cmap.N)

    # A hairline in the surface colour separates neighbours without drawing a grid.
    cells.plot(column=field, cmap=cmap, norm=norm, ax=ax, linewidth=0.15, edgecolor="white")

    ax.set_title(title, color=INK, fontsize=12, pad=10, loc="left")
    ax.set_axis_off()

    bar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.02,
        ticks=edges,
    )
    bar.ax.set_xticklabels([f"{edge:.{1 if unit else 2}f}" for edge in edges], fontsize=8, color=MUTED)
    bar.set_label(unit or "index, heat x people", color=MUTED, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)


def main() -> None:
    config = load_config()
    cells = gpd.read_file(ROOT / config["paths"]["web_geojson"])
    cells = cells.to_crs(config["crs"]["projected"])

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    figure.patch.set_facecolor("white")

    panel(axes[0], cells, "lst_mean", HEAT_RAMP,
          "Mean surface temperature, pre-monsoon 2024", "°C")
    panel(axes[1], cells, "vulnerability", EXPOSURE_RAMP,
          "Exposure: heat x population density", "")

    figure.text(
        0.5, 0.025,
        f"Kathmandu valley, {len(cells)} hexagons of about 0.74 km\u00b2. "
        "Landsat 8/9 surface temperature, four scenes 15 March to 31 May 2024. "
        "Classes are quantiles. Exposure measures how many people live where it is hot, not harm.",
        color=MUTED, fontsize=8.5, ha="center",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.95))

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGE_DIR / "heat_and_exposure.png"
    figure.savefig(out_path, dpi=160, facecolor="white")
    plt.close(figure)
    print(f"written: {out_path.relative_to(ROOT)}")

    shap_source = ROOT / config["paths"]["outputs_dir"] / "shap_summary.png"
    if shap_source.exists():
        shutil.copyfile(shap_source, IMAGE_DIR / "shap_summary.png")
        print(f"written: {(IMAGE_DIR / 'shap_summary.png').relative_to(ROOT)}")
    else:
        print("note: run src/05_model.py first to produce the SHAP plot")


if __name__ == "__main__":
    main()
