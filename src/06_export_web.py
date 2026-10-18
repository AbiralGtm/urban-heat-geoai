"""Combine heat and people into a vulnerability index and write the file the map reads.

The index is the measured surface temperature, scaled 0 to 1 across the valley, times
the normalised log of population density. It measures exposure: how hot a neighbourhood
runs and how many people live there. It is not a measure of harm, and it carries no
information about who can cope with the heat.

Writes docs/data/kathmandu.geojson.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Columns the map needs; everything else is left out to keep the file small.
WEB_COLUMNS = [
    "h3_id",
    "lst_mean",
    "predicted_full",
    "ndvi_mean",
    "cover_builtup",
    "population",
    "pop_density",
    "heat_index",
    "people_index",
    "vulnerability",
    "top_decile",
]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def scale_01(values: pd.Series) -> pd.Series:
    """Min-max scaling across the valley, so 0 is the coolest hexagon and 1 the hottest."""
    low, high = values.min(), values.max()
    if high == low:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def vulnerability_index(features: pd.DataFrame) -> pd.DataFrame:
    """Heat times people, both on a 0 to 1 scale.

    Population density is logged first because it spans four orders of magnitude across
    the valley; without that, a handful of core hexagons would set the whole scale.
    """
    heat = scale_01(features["lst_mean"])
    people = scale_01(np.log1p(features["pop_density"]))

    index = pd.DataFrame(
        {
            "heat_index": heat,
            "people_index": people,
            "vulnerability": heat * people,
        },
        index=features.index,
    )
    index["top_decile"] = index["vulnerability"] >= index["vulnerability"].quantile(0.9)
    return index


def main() -> None:
    config = load_config()

    features = pd.read_csv(ROOT / config["paths"]["features"])
    predictions = pd.read_csv(ROOT / config["paths"]["outputs_dir"] / "predictions.csv")
    features = features.merge(predictions[["h3_id", "predicted_full"]], on="h3_id", how="left")

    features = pd.concat([features, vulnerability_index(features)], axis=1)

    grid = gpd.read_file(ROOT / config["paths"]["grid"], layer="grid")
    web = grid[["h3_id", "geometry"]].merge(features[WEB_COLUMNS], on="h3_id", how="inner")
    web = web.to_crs(config["crs"]["geographic"])

    # Six decimals is about 0.1 m at this latitude; more only inflates the file.
    web.geometry = web.geometry.set_precision(1e-6)
    for column in web.columns.drop(["h3_id", "geometry", "top_decile"]):
        web[column] = web[column].round(4)

    out_path = ROOT / config["paths"]["web_geojson"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(web.to_json(drop_id=True))

    hottest = web.nlargest(5, "vulnerability")[["h3_id", "lst_mean", "pop_density", "vulnerability"]]
    print(f"cells written: {len(web)}")
    print(f"top decile: {int(web['top_decile'].sum())} cells")
    print("most exposed cells:")
    print(hottest.to_string(index=False))
    print(f"size: {out_path.stat().st_size / 1e6:.2f} MB")
    print(f"written: {out_path.relative_to(ROOT)}")

    summary = {
        "cells": int(len(web)),
        "top_decile_cells": int(web["top_decile"].sum()),
        "lst_mean_min": float(web["lst_mean"].min()),
        "lst_mean_mean": float(web["lst_mean"].mean()),
        "lst_mean_max": float(web["lst_mean"].max()),
        "population_total": float(web["population"].sum()),
        "population_in_top_decile": float(web.loc[web["top_decile"], "population"].sum()),
    }
    summary_path = ROOT / config["paths"]["outputs_dir"] / "exposure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"written: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
