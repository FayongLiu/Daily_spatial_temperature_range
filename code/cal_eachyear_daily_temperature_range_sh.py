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

# this file is for calculating DSTR in south hemisphere based on ERA5-Land

# Initialize logging
logging.basicConfig(filename=r'/data1/fyliu/a_temperature_range/code/logging/data_processing_south_hemisphere.log', 
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
        filename = f'temperature_2m_{year}.nc'
        filepath = os.path.join(path, filename)
        logging.info(f"Processing file: {filepath}")
        
        # Get time info using netCDF
        dataset = nc.Dataset(filepath, 'r')
        datatime = dataset.variables['time']
        datatime = nc.num2date(datatime, datatime.units)
        dataset.close()
        
        # Read data using xarray
        dataset = xr.open_dataset(filepath)
        
        # create an empty array for each year hourly max T range
        # each column represents, year, month, day of month, hour, day of year, maxT, minT, max Trange
        # maxT, maxT lon, maxT lat, maxT day/night, localhour
        # minT, minT, minT lon, minT lat, minT day/night, localhour
        teststart = 0
        testend = len(datatime)
        testleng = testend - teststart
        eyhmtr = np.zeros((testleng, 8))
        eyhmaxt = list(range(testleng))
        eyhmint = list(range(testleng))
        
        last_log_time = None
        
        for tm in range(teststart, testend):
            # get time info according to tm, time index
            realtime = datatime[tm]
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
                last_log_time = realtime
            
            # Get max and min temperature, location, and day/night info
            da = dataset.t2m.sel(time = str(realtime), latitude = slice(0.01, -90))
            da['longitude'] = xr.where(da['longitude'] > 180, da['longitude'] - 360, da['longitude'])

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
            
            gc.collect()

        save_results(eyhmtr, eyhmaxt, eyhmint, year)
        del eyhmtr, eyhmaxt, eyhmint
        gc.collect()
        
        logging.info(f"Completed processing for year {year}")
    except Exception as e:
        logging.error(f"Error processing year {year}: {e}")


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
def save_results(eyhmtr, eyhmaxt, eyhmint, year):
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
    outname = f'south_hemisphere_eachyear_maxTrange_{year}'
    filepath = os.path.join(outpath, outname)
    with open(filepath, 'wb') as f:
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

    path = r'/data_backup/share/ERA5_land/temperature/hourly/'
    outpath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_original_list/south_hemisphere/'
    os.makedirs(outpath, exist_ok=True)
    
    years = np.arange(1950, 2024)
    
    parallel_process(years)
    
    time_end = time.time()
    time_sum = time_end - time_start
    logging.info(f"Total processing time: {time_sum} seconds")
    
    
    
    
    


 