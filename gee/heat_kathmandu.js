/**
 * Kathmandu urban heat - Earth Engine export script.
 *
 * Builds a pre-monsoon composite of Landsat 8/9 surface temperature and greenness over
 * the Kathmandu valley, and exports it together with terrain, land cover and population
 * as six GeoTIFFs. Run it in the Earth Engine Code Editor, start the six tasks, then
 * download the files from Drive into data/raw/kathmandu/.
 *
 * The settings below must match config.yaml.
 */

// ---------------------------------------------------------------- settings
var BBOX = [85.22, 27.60, 85.52, 27.83];   // west, south, east, north
var START = '2024-03-15';
var END = '2024-05-31';
var MAX_CLOUD = 30;
var CRS = 'EPSG:32645';                    // UTM 45N
var SCALE = 30;                            // metres, for every export
var FOLDER = 'kathmandu_heat';

var region = ee.Geometry.Rectangle(BBOX, null, false);

// ---------------------------------------------------------------- Landsat
// Collection 2 Level 2 is already atmospherically corrected. Bits 1-4 of QA_PIXEL mark
// dilated cloud, cirrus, cloud and cloud shadow; a pixel is kept only if all are clear.
function maskLandsat(image) {
  var qa = image.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(parseInt('11110', 2)).eq(0);
  var saturated = image.select('QA_RADSAT').eq(0);
  return image.updateMask(clear).updateMask(saturated);
}

// Scale factors from the Collection 2 Level 2 documentation.
function addBands(image) {
  var lst = image.select('ST_B10')
      .multiply(0.00341802).add(149.0).subtract(273.15)
      .rename('lst');
  var sr = image.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2);
  var ndvi = sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('ndvi');
  return image.addBands([lst, ndvi]);
}

var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
    .filterBounds(region)
    .filterDate(START, END)
    .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD))
    .map(maskLandsat)
    .map(addBands);

print('scenes used:', landsat.size());
print('scene ids:', landsat.aggregate_array('LANDSAT_PRODUCT_ID'));

// Median over the scenes: a single cloudy pixel in one scene cannot pull the composite.
var composite = landsat.select(['lst', 'ndvi']).median().clip(region);

// Share of the window for which each pixel had a valid observation. Stage 3 uses this
// to drop hexagons that are mostly gap.
var validShare = landsat.select('lst').count()
    .divide(landsat.size())
    .rename('valid_share')
    .clip(region);

print('composite statistics:', composite.addBands(validShare).reduceRegion({
  reducer: ee.Reducer.minMax().combine(ee.Reducer.mean(), '', true),
  geometry: region,
  scale: SCALE,
  maxPixels: 1e9
}));

// ---------------------------------------------------------------- terrain, cover, people
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()
    .rename('dem').clip(region);
var slope = ee.Terrain.slope(dem.setDefaultProjection('EPSG:4326', null, 30))
    .rename('slope').clip(region);
var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
    .select('Map').rename('worldcover').clip(region);
var population = ee.ImageCollection('WorldPop/GP/100m/pop')
    .filter(ee.Filter.eq('country', 'NPL'))
    .filter(ee.Filter.eq('year', 2020))
    .first()
    .rename('population').clip(region);

// ---------------------------------------------------------------- exports
// Land cover is categorical, so it keeps Earth Engine's default nearest-neighbour
// resampling; population is a count per 100 m cell and is exported at its own
// resolution so that the totals stay right.
function export_(image, name, scale, crs) {
  Export.image.toDrive({
    image: image.toFloat(),
    description: 'kathmandu_' + name,
    folder: FOLDER,
    fileNamePrefix: name,
    region: region,
    scale: scale || SCALE,
    crs: crs || CRS,
    maxPixels: 1e9
  });
}

export_(composite.select('lst'), 'lst');
export_(composite.select('ndvi'), 'ndvi');
export_(dem, 'dem');
export_(slope, 'slope');
export_(worldcover, 'worldcover');
export_(population, 'pop', 100);
export_(validShare, 'valid_share');
