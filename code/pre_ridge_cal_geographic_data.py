# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from osgeo import gdal, ogr, gdalnumeric, osr
from shapely.geometry import MultiLineString, Point


# this file is for calculating geographic data of each provinces in China
# including dem range, sea distance range, lat range, lon range, area 


# def a function to combine outpath and outname
def combinename(outpath, outname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)
    return filepath


# def a function read csv data
# return, labels, label name, list, length=33, except Macau
# datals, combined data, list, length=n, according to the user and csvfile
def readcsv(path, datanum, istranspose):
    allfile = os.listdir(path)
    allfile.sort()
    filelen = len(allfile)
    labels = list(range(filelen))
    datals = [[] for i in range(datanum)]
    i = 0
    for ff in allfile:
        key = ff.lower().split('_')
        if key[0] in ['hong', 'inner']:
            key = key[0:2]
            key = [word.capitalize() for word in key]
            key = ' '.join(key)
        elif key[0] == 'north':
            key = 'NH'
        elif key[0] == 'south':
            key = 'SH'
        else:
            key = key[0].capitalize()
        labels[i] = key
        filepath = os.path.join(path, ff)
        # read data
        data = pd.read_csv(filepath, sep=',', header=0, index_col=0)
        data1 = data.values[:, 0:datanum]
        # save to list
        for j in range(datanum):
            datals[j].append(data1[:, j])
        i += 1

    # change to np.array
    for lsnum in range(datanum):
        arrnum = 0
        for ls in datals[lsnum]:
            if arrnum == 0:
                tempt = ls
                arrnum += 1
            else:
                # 37*datalegnth
                tempt = np.vstack((tempt, ls))
        # datalength*37
        if istranspose == 1:
            tempt = tempt.T
        datals[lsnum] = tempt

    return labels, datals


# set outpath
outpath = r'/data1/fyliu/a_temperature_range/process_data/correlation/'

# get DSTR data
path = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_mmmvalue/'
# 0-1 reperensts DSTR
datanum = 1
labels, datals = readcsv(path, datanum, 0)
dstr = datals[0]
df = pd.DataFrame(dstr, index=labels,
                  columns=['Max of maxDSTR', 'Mean of maxDSTR', 'Min of MaxDSTR', 'Std of maxDSTR', 'Max of meanDSTR',
                           'Mean of meanDSTR', 'Min of meanDSTR', 'Std of meanDSTR', 'Max of minDSTR',
                           'Mean of minDSTR', 'Min of minDSTR', 'Std of minDSTR'])
outname = 'dstr.csv'
filepath = combinename(outpath, outname)
df.to_csv(filepath, sep=',')
print(filepath)

# get dem/lat/lon range, area
proshp = r'/data1/fyliu/a_temperature_range/data/boundary/China_GS(2020)4619/China_provincial_polygon.shp'
conpath = r'/data1/fyliu/a_temperature_range/data/boundary/China_GS(2020)4619/China_province_code.xlsx'
# set f(x) table
mapping_df = pd.read_excel(conpath)
pro_maps = {}
for index, row in mapping_df.iterrows():
    pro_maps[str(row['DZM'])] = row['key']

# read gdf
progdf = gpd.read_file(proshp, encoding='utf-8')
progdf['pro'] = progdf['DZM'].map(pro_maps)

gdf = progdf

# cal dem, LUCC area, biodiversity value, and distance to sea
coastlineshp = r'/data1/fyliu/a_temperature_range/data/boundary/China_GS(2020)4619/China_coastline.shp'
coastline = gpd.read_file(coastlineshp)
coastline = coastline.to_crs(epsg=32650)
coastline_combined = coastline.geometry.unary_union

dem_path = r'/data1/fyliu/a_temperature_range/data/dem_1km/dem_1km/'
dem_ds = gdal.Open(dem_path)

max_dis_sea = []
min_dis_sea = []
dis_sea_range = []
max_dem_values = []
min_dem_values = []
dem_range_ls = []

# iterate by province
for index, row in gdf.iterrows():
    # get geometry
    province_geom = row['geometry']

    # create memory source
    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('')

    # create new layer
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)
    mem_layer = mem_ds.CreateLayer('province_geom', spatial_ref)

    # add feature to layer
    geometry = ogr.CreateGeometryFromWkb(province_geom.wkb)
    feature_defn = mem_layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(geometry)
    mem_layer.CreateFeature(feature)

    # create mask
    # dem mask, an empty grid with the same size, projection, and geotransform as dem_ds
    mask_ds = gdal.GetDriverByName('MEM').Create('', dem_ds.RasterXSize, dem_ds.RasterYSize, 1, gdal.GDT_Byte)
    mask_ds.SetProjection(dem_ds.GetProjection())
    mask_ds.SetGeoTransform(dem_ds.GetGeoTransform())
    # transfer geometry to grid
    gdal.RasterizeLayer(mask_ds, [1], mem_layer, burn_values=[1])
    # transfer to array
    mask_array = gdalnumeric.BandReadAsArray(mask_ds.GetRasterBand(1))
    # clip
    clipped_dem_array = dem_ds.ReadAsArray()[mask_array == 1]
    # cal max, min dem
    max_dem = clipped_dem_array.max()
    clipped_dem_array[clipped_dem_array == -32768] = max_dem
    min_dem = clipped_dem_array.min()
    dem_range = max_dem - min_dem
    # append to list
    max_dem_values.append(max_dem)
    min_dem_values.append(min_dem)
    dem_range_ls.append(dem_range)

# add new field
gdf['max Dem'] = max_dem_values
gdf['min Dem'] = min_dem_values
gdf['Dem range'] = dem_range_ls

# cal sea distance
gdf_utm = gdf.to_crs(epsg=32650)
for index, row in gdf_utm.iterrows():
    # get geometry
    province_geom = row['geometry']

    # Check if the geometry is a MultiPolygon or MultiLineString (multi-part geometry)
    if province_geom.boundary.is_empty:
        continue

    boundary_points = []

    # If it's a multi-part geometry (e.g., MultiPolygon or MultiLineString)
    if isinstance(province_geom.boundary, MultiLineString):
        # Use the `geoms` attribute to iterate through the sub-geometries
        for part in province_geom.boundary.geoms:
            boundary_points.extend(part.coords)  # Extract coordinates from each part
    else:
        # If it's a single geometry (Polygon or LineString), extract coords directly
        boundary_points = list(province_geom.boundary.coords)

    # Now boundary_points contains all the boundary coordinates
    distances = [Point(pt).distance(coastline_combined) for pt in boundary_points]

    # Find the minimum and maximum distances
    min_distance = min(distances) / 1000
    max_distance = max(distances) / 1000

    # Append results to lists
    min_dis_sea.append(min_distance)
    max_dis_sea.append(max_distance)
    dis_sea_range.append(max_distance - min_distance)

gdf['min sea distance'] = min_dis_sea
gdf['max sea distance'] = max_dis_sea
gdf['Sea distance range'] = dis_sea_range

# cal bounds
merged_geom = gdf.unary_union
china_bbox = merged_geom.bounds

bbox = gdf.geometry.bounds
gdf['min Lon'] = bbox['minx']
gdf['max Lon'] = bbox['maxx']
gdf['min Lat'] = bbox['miny']
gdf['max Lat'] = bbox['maxy']
gdf['Lon range'] = bbox['maxx'] - bbox['minx']
gdf['Lat range'] = bbox['maxy'] - bbox['miny']

# project
proj_crs = {'proj': 'aea',
            'lat_1': china_bbox[1] + (china_bbox[3] - china_bbox[1]) / 4,
            'lat_2': china_bbox[3] - (china_bbox[3] - china_bbox[1]) / 4,
            'lat_0': (china_bbox[1] + china_bbox[3]) / 2,
            'lon_0': (china_bbox[0] + china_bbox[2]) / 2}
gdf = gdf.to_crs(proj_crs)

# cal shp area and lat/lon range
gdf['total Area based on Shp'] = gdf.geometry.area / 10 ** 6
gdf = gdf.sort_values(by='pro')

# print
province_data = gdf[
    ['pro', 'min sea distance', 'max sea distance', 'Sea distance range', 'max Dem', 'min Dem', 'Dem range',
    'min Lon', 'max Lon', 'min Lat', 'max Lat', 'Lon range', 'Lat range']]

# save result
outname = 'province_dem_lon_lat_area.csv'
filepath = combinename(outpath, outname)
province_data.to_csv(filepath, sep=',', index=False)
print(filepath)



