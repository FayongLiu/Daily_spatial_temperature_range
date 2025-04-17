# -*- coding: utf-8 -*-
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import gc
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
import rasterio.features
from scipy import stats
from scipy.stats import linregress


# this file is for calculating global area at 0.1 deg soluiton


# def a function to combine outpath and outname
def combine_name(outpath, outname):

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)
    
    return filepath
    
    

# def a function to cal the area, km^2
def cal_area(ref_path):

    ref_ds = xr.open_dataset(ref_path)
    ref_ds = ref_ds['fal'].isel(valid_time=0)
    ref_data = ref_ds.values
    ref_lat = ref_ds['latitude'].values
    ref_lon = ref_ds['longitude'].values
    
    latsize = 111.3214
    
    for lat in range(len(ref_lat)):
        lat_cos = np.cos(np.pi*ref_lat[lat]/180)
        ref_data[lat, :] = latsize*latsize*lat_cos
    
    # create a new xarray
    area_ds = xr.Dataset(data_vars={
        'area': (['lat', 'lon'], ref_data)
    }, coords={
        'lat': ref_lat,
        'lon': ref_lon
    })
    
    area_ds['lon'] = xr.where(area_ds['lon'] > 180, area_ds['lon'] - 360, area_ds['lon'])
    area_ds = area_ds.sortby('lon').sortby('lat')
    
    return area_ds
    
    
    
# cal area for regional aggregate
# ===============================================================================
# ===============================================================================
ref_path = r'/data_backup/share/ERA5_land/radiation/monthly/merged_annual.nc'
area_ds = cal_area(ref_path)
#area_ds['area'].plot()
#plt.show()

# save result
outpath = r'/data1/fyliu/a_temperature_range/data/area_mask/'
os.makedirs(outpath, exist_ok = True)
outname = 'area_0.1deg.nc'
finalpath = combine_name(outpath, outname)
area_ds.to_netcdf(finalpath)
print(finalpath)
