# -*- coding: utf-8 -*-
import func_timeout
from func_timeout import func_set_timeout
import os
import netCDF4 as nc
import xarray as xr
import numpy as np
import ephem
import gc
import pickle
import matplotlib.pyplot as plt
import salem
from datetime import timedelta
import time
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import traceback
from pypinyin import pinyin, Style

# this file is for calculating DSTR in each provinces in china based on ERA5-Land

# Initialize logging
logging.basicConfig(filename=r'/data1/fyliu/a_temperature_range/code/logging/data_processing_province.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)


# Set over time, 10 sec
@func_set_timeout(10)


# Function to determine whether it's day or night
def is_daytime(lat, lon, utc_time):
    obs = ephem.Observer()
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.date = str(utc_time)
    
    try:
        prise = obs.previous_rising(ephem.Sun()).datetime()
        pset = obs.previous_setting(ephem.Sun()).datetime()
        # determine whether it is day(1) or night(0)
        dn = 1 if pset < prise else 0

        # cal the local time
        timezone = lon / 15
        localtime = utc_time + timedelta(hours=timezone)
        localhour = localtime.timetuple().tm_hour
        
    # if error, sun is below/above the horizon, night, day 
    except Exception as e:
        if 'below' in e.args[0]:
            dn = 0
            localhour = 25
        elif 'above' in e.args[0]:
            dn = 1
            localhour = 24
        else:
            logging.error(f"Error in is_daytime: {e}")
            dn = 2
            localhour = 26
    return dn, localhour


# Function to calculate the DSTR for a specific year
def cal_dstr(year):
    try:
        for index, row in coun.iterrows():
            if '61' in row['DZM']:
                prov = 'Shaanxi'
            elif '54' in row['DZM']:
                prov = 'Tibet'
            elif '15' in row['DZM']:
                prov = 'Inner_Mongolia'
            elif '500000' in row['DZM']:
                prov = 'Chongqing'
            elif '81' in row['DZM']:
                prov = 'Hong_Kong'
            elif '82' in row['DZM']:
                prov = 'Macau'
            else:
                prov = row['NAME']
                prov = ''.join([p[0] for p in pinyin(prov, style=Style.NORMAL)])
                prov = prov.capitalize()
                
            # new out path
            pro_name = 'ERA5_Land_t2m_pro_' + prov
            pro_outpath1 = os.path.join(pro_outpath, pro_name)
            os.makedirs(pro_outpath1, exist_ok=True)

            geom = row.geometry
            grid = np.full((num_lat, num_lon), np.nan)
            rasterize([(geom, 1)], out_shape=(num_lat, num_lon), transform=transform, fill=0, all_touched=True, out=grid)
            # create mask
            lon_values = np.linspace(min_lon, max_lon, num_lon)
            lat_values = np.linspace(max_lat, min_lat, num_lat)
            grid_ls = xr.DataArray(
                grid,
                coords={'longitude': lon_values, 'latitude': lat_values},
                dims=('latitude', 'longitude'),
                name = 'region'
            )
    
            # display filepath
            pathname = 'year_' + str(year)
            filepath = os.path.join(path, pathname)
            allfile1 = os.listdir(filepath)
            allfile1.sort()
            logging.info(f"Processing file: {filepath}")
            
            # create an empty array for each year hourly max T range
            # each column represents, year, month, day of month, hour, day of year, maxT, minT, max Trange
            # maxT, maxT lon, maxT lat, maxT day/night, localhour
            # minT, minT, minT lon, minT lat, minT day/night, localhour
            teststart = 0
            testend = len(allfile1)
            testleng = testend - teststart
            eyhmtr = np.zeros((testleng, 8))
            eyhmaxt = list(range(testleng))
            eyhmint = list(range(testleng))
            
            last_log_time = None
            
            for tm in range(teststart, testend):
            
                filename1 = allfile1[tm]
                filepath1 = os.path.join(filepath, filename1)
                # print(filepath1)
                
                # get time info use netCDF
                dataset = nc.Dataset(filepath1, 'r')
                datatime = dataset.variables['time']
                datatime = nc.num2date(datatime, datatime.units)
                dataset.close()
                
                # read data use xarray
                dataset = xr.open_dataset(filepath1)
                
                # get time info according to tm, time index
                realtime = datatime
                ind = tm - teststart
                tempt_time = realtime.timetuple()
                eyhmtr[ind, 0] = tempt_time.tm_year
                eyhmtr[ind, 1] = tempt_time.tm_mon
                eyhmtr[ind, 2] = tempt_time.tm_mday
                eyhmtr[ind, 3] = tempt_time.tm_hour
                eyhmtr[ind, 4] = tempt_time.tm_yday
                
                # print info
                if last_log_time is None or (realtime - last_log_time).days >= 1:
                    logging.info(f"Year {year}: Processing date {realtime}")
                    logging.info(f"Year {year}: Processing province {prov}")
                    last_log_time = realtime
                
                # get max and min temperature, location and day/night info
                da = dataset.t2m
                da['longitude'] = xr.where(da['longitude'] > 180, da['longitude'] - 360, da['longitude'])
                da = da.sortby('longitude')
             
                da = da.where(grid_ls.data == 1)
                da = da.dropna(dim = 'longitude', how = 'all').dropna(dim =  'latitude', how = 'all')
                
                # Max temperature
                t = da.where(da == da.max(), drop=True).squeeze()
                notnan = t.to_numpy()[~np.isnan(t.to_numpy())]
                eyhmtr[ind, 5] = notnan[0] - 273.15
                eyhmaxt[ind] = process_temperature(t)
    
                # Min temperature
                t = da.where(da == da.min(), drop=True).squeeze()
                notnan = t.to_numpy()[~np.isnan(t.to_numpy())]
                eyhmtr[ind, 6] = notnan[0] - 273.15
                eyhmint[ind] = process_temperature(t)
    
                # Max temperature range
                eyhmtr[ind, 7] = eyhmtr[ind, 5] - eyhmtr[ind, 6]
                
                # save clip data
                this_year_name = 'year_' + str(int(eyhmtr[ind, 0]))
                pro_outpath2 = os.path.join(pro_outpath1, this_year_name)
                os.makedirs(pro_outpath2, exist_ok=True)
                outname = prov + '_t2m_' + str(int(eyhmtr[ind, 0])) + str(int(eyhmtr[ind, 1])).zfill(2) + str(int(eyhmtr[ind, 2])).zfill(2) + str(int(eyhmtr[ind, 3])).zfill(2) + '.nc'
                pro_filepath = os.path.join(pro_outpath2, outname)
                da.to_netcdf(pro_filepath)
            
                del da
                gc.collect()
    
            save_results(eyhmtr, eyhmaxt, eyhmint, year, prov)
            del eyhmtr, eyhmaxt, eyhmint
            gc.collect()
            
            logging.info(f"Completed processing for year {year}")
    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(f"Error processing year {year}: {e}\nTraceback: {tb_str}")


def process_temperature(t):
    latlen = t.latitude.size 
    lonlen = t.longitude.size
    tnp = t.to_numpy()
    lonnp = t.longitude.to_numpy()
    latnp = t.latitude.to_numpy()
    
    local_notnan = tnp[~np.isnan(tnp)]
    tempt = np.zeros((len(local_notnan), 5))
    num = 0
    
    for lati in range(latlen):
        for loni in range(lonlen):
            if (lonlen == 1) & (latlen == 1):
                a = tnp
                b = lonnp
                c = latnp
            elif (lonlen == 1) & (latlen > 1):
                a = tnp[lati]
                b = lonnp
                c = latnp[lati]
            elif (lonlen > 1) & (latlen == 1):
                a = tnp[loni]
                b = lonnp[loni]
                c = latnp
            else:
                a = tnp[lati, loni]
                b = lonnp[loni]
                c = latnp[lati]
            
            if ~np.isnan(a):
                tempt[num, 0] = a - 273.15
                tempt[num, 1] = b
                tempt[num, 2] = c
                try:
                    tempt[num, 3], tempt[num, 4] = is_daytime(tempt[num, 2], tempt[num, 1], realtime)
                except:
                    tempt[num, 3] = 3
                    tempt[num, 4] = 26
                num += 1

    return tempt


# Function to save results
def save_results(eyhmtr, eyhmaxt, eyhmint, year, prov):
    maxdoy = int(eyhmtr[:, 4].max())
    mindoy = int(eyhmtr[:, 4].min())
    doy = maxdoy - mindoy + 1
    
    eydmtr = list(range(doy))
    eydmaxt = list(range(doy))
    eydmint = list(range(doy))
    for date in range(mindoy, maxdoy + 1):
        row_ind = np.where(eyhmtr[:, 4] == date)
        tempt = eyhmtr[row_ind[0], :]
        temptmax = [eyhmaxt[ind] for ind in row_ind[0]]
        temptmin = [eyhmint[ind] for ind in row_ind[0]]
        maxTrange = tempt[:, 7].max()
        row_ind = np.where(tempt[:, 7] == maxTrange)
        dateind = date - mindoy
        eydmtr[dateind] = tempt[row_ind[0], :]
        eydmaxt[dateind] = [temptmax[ind] for ind in row_ind[0]] 
        eydmint[dateind] = [temptmin[ind] for ind in row_ind[0]]
        del tempt, temptmax, temptmin
        gc.collect()
   
    each_year = [eydmtr, eydmaxt, eydmint]
    outpath1 = os.path.join(outpath, prov)
    outname1 = f'{prov}_eachyear_maxTrange_{year}'
    filepath1 = os.path.join(outpath1, outname1)
    with open(filepath1, 'wb') as f:
        pickle.dump(each_year, f)
    logging.info(f"Results saved for year {year}")
    
    return

        
# Multiprocessing function
def parallel_process(year_range):
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(cal_dstr, year_range), total=len(year_range)):
            pass


if __name__ == "__main__":
    time_start = time.time()

    path = r'/data_backup/fyliu/ERA5_Land/t2m/ERA5_Land_t2m_China/'
    outpath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_original_list/'
    os.makedirs(outpath, exist_ok=True)
    pro_outpath = r'/data_backup/fyliu/ERA5_Land/t2m/'
    coun_path = r'/data1/fyliu/a_temperature_range/data/boundary/China_GS(2020)4619/China_provincial_polygon.shp'
    
    # read country boundary data
    coun = gpd.read_file(coun_path)
    
    # set for transfer to .nc
    min_lat = 18.2
    min_lon = 73.5
    max_lon = 135
    max_lat = 53.6
    resolution = 0.1
    
    # ensure totally overlay
    num_lon = 616
    num_lat = 354
    transform = rasterio.transform.from_origin(min_lon, max_lat, resolution, resolution)
    
    years = np.arange(1950, 2024)
    
    parallel_process(years)
    
    time_end = time.time()
    time_sum = time_end - time_start
    logging.info(f"Total processing time: {time_sum} seconds")
    
    
    
    
