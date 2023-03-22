# Image Preprocessing in ArcGIS Pro

## Prerequisites
### 1. ArcGIS Pro
ArcGIS Pro with one of the following licenses:
- 3D Analyst
- Spatial Analyst

### 2. Relief Visualization Toolbox (optional)

  The [Relief Visualization Toolbox](https://rvt-py.readthedocs.io/en/latest/index.html) includes ready to use raster functions, specifically for elevation datasets, and simplifies the process of of calculating `Multi-Scale Relief Model` and `Simple Local Relief Model`. See [RVT Installation Guide](https://rvt-py.readthedocs.io/en/latest/install_arcgis.html#install-arcgis) for more details about the tool.

## Workflow
### Prepare Ground Truth Database
1. Download the cultural heritage database from [GeoNorge](https://geonorge.no). The dataset are called `Enkeltminner` and can be found here: [https://kartkatalog.geonorge.no/metadata/kulturminner-lokaliteter-enkeltminner-og-sikringssoner/c72906a0-2bc2-41d7-bea2-c92d368e3c49](https://kartkatalog.geonorge.no/metadata/kulturminner-lokaliteter-enkeltminner-og-sikringssoner/c72906a0-2bc2-41d7-bea2-c92d368e3c49).
2. Import the downloaded file to ArgGIS Pro. Go to `Catalog` pane, right click on `Databases` directory and then `Add Database`. 

3. Open a new map.
4. The imported database includes a feature class called `enkeltminne`. Drag this into the map. 
5. Apply a filter to the feature class to only show burial mounds. Go to `Contents` pane. Right click on the feature class to display `properties` window. Go to the `Definition Query` tab, and apply the following SQL filter:
```
enkeltminneart IN (1702, 1703) AND vernetype NOT IN ('FJE')
```

### Prepare LiDAR data

1. Download LiDAR data from [Norwegian Mapping Authority](https://hoydedata.no).
2. Import the raw data into a chosen folder.
3. Merge parts into one raster using `Moasic to New Raster` tool in ArcGIS Pro. 
4. Calculate selected relief models.


### Annotation
1. Use the `Export Training Data fro Deep Learning` tool for each relief model. 