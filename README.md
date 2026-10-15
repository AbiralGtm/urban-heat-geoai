# Kathmandu urban heat GeoAI

Which neighbourhoods of the Kathmandu valley run hottest in the pre-monsoon season,
what land-surface features go with that heat, and where do the hot areas overlap with
dense population?

The valley is split into H3 resolution 8 hexagons of roughly 0.74 km2. Each hexagon
gets a mean land surface temperature from Landsat 8/9, plus greenness, elevation,
slope, land-cover shares and population. A Random Forest learns which features go with
hot hexagons, tested with spatial cross-validation so that whole blocks stay unseen.
The hot and densely populated hexagons are then shown on an interactive map.

This is a descriptive study of one season in one valley. It does not forecast
temperature, and it does not show that any feature causes heat.

## Pipeline

| Stage | Script | Output |
| --- | --- | --- |
| 1. Grid | `src/01_make_grid.py` | `data/processed/kathmandu_grid.gpkg` |
| 2. Satellite data | `gee/heat_kathmandu.js` | 6 rasters in `data/raw/kathmandu/` |
| 3. Features | `src/03_raster_features.py` | `data/processed/features.csv` |
| 4. OSM (optional) | `src/04_osm_features.py` | road density, waterway distance |
| 5. Model | `src/05_model.py` | `outputs/metrics.json`, SHAP plot |
| 6. Vulnerability and map | `src/06_export_web.py` | `docs/data/kathmandu.geojson` |

Every stage reads `config.yaml` and the output of the stage before it, so any stage can
be re-run on its own.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Results

To be filled in with the numbers the code produces. Nothing is reported here before it
has been measured.

## Limits

To be written once the model has been run.
