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
import datetime
from datetime import timedelta
import time
import logging
import traceback
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# this file is for calculating DSTR in globe based on MERRA2 data

# Initialize logging
logging.basicConfig(filename=r'/data1/fyliu/a_temperature_range/code/logging/data_processing_world_merra.log', 
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
        path_ls = os.listdir(path)
        str_year = str(year)
        sublist = [file for file in path_ls if str_year in file]
        sublist.sort()
        datelen = len(sublist)
        
        # create an empty array for each day hourly max T range
        # each column represents, year, month, day of month, hour, day of year, maxT, minT, max Trange
        # maxT, maxT lon, maxT lat, maxT day/night, localhour
        # minT, minT, minT lon, minT lat, minT day/night, localhour
        eydmtr = list(range(datelen))
        eydmaxt = list(range(datelen))
        eydmint = list(range(datelen))
            
        for datenum in range(datelen):
            
            filepath = os.path.join(path, sublist[datenum])
            logging.info(f"Processing file: {filepath}")
        
            # Get time info using netCDF
            dataset = nc.Dataset(filepath, 'r')
            datatime = dataset.variables['time']
            datatime = nc.num2date(datatime, datatime.units)
            dataset.close()
            
            # Read data using xarray
            dataset = xr.open_dataset(filepath)
            
            tmlen = len(datatime)
            edhmtr = np.zeros((tmlen, 8))
            edhmaxt = list(range(tmlen))
            edhmint = list(range(tmlen))
            
            for tm in range(tmlen):
                # get time info according to tm, time index
                realtime = datatime[tm]
                tempt_time = realtime.timetuple()
                ind = tempt_time.tm_hour
                
                edhmtr[ind, 0] = tempt_time.tm_year
                edhmtr[ind, 1] = tempt_time.tm_mon
                edhmtr[ind, 2] = tempt_time.tm_mday
                edhmtr[ind, 3] = tempt_time.tm_hour
                edhmtr[ind, 4] = tempt_time.tm_yday
                
                # Get max and min temperature, location, and day/night info
                da = dataset.T2M.sel(time=str(realtime))
    
                # Max temperature
                t = da.where(da == da.max(), drop=True).squeeze()
                notnan = t.to_numpy()[~np.isnan(t.to_numpy())]
                edhmtr[ind, 5] = notnan[0] - 273.15
                edhmaxt[ind] = process_temperature(t, realtime)
    
                # Min temperature
                t = da.where(da == da.min(), drop=True).squeeze()
                notnan = t.to_numpy()[~np.isnan(t.to_numpy())]
                edhmtr[ind, 6] = notnan[0] - 273.15
                edhmint[ind] = process_temperature(t, realtime)
    
                # Max temperature range
                edhmtr[ind, 7] = edhmtr[ind, 5] - edhmtr[ind, 6]
                
            maxTrange = edhmtr[:, 7].max()
            row_ind = np.where(edhmtr[:, 7] == maxTrange)
            daymtr = edhmtr[row_ind[0], :]
            daymaxt = [edhmaxt[i] for i in row_ind[0]]
            daymint = [edhmint[i] for i in row_ind[0]]
            
            eydmtr[datenum] = daymtr
            eydmaxt[datenum] = daymaxt
            eydmint[datenum] = daymint
            
            del edhmtr, edhmaxt, edhmint
            gc.collect()

        # save result
        # print(eydmtr, eydmaxt, eydmint)
        each_year = [eydmtr, eydmaxt, eydmint]
        outname = f'world_eachyear_maxTrange_{year}.pkl'
        filepath = os.path.join(outpath, outname)
        with open(filepath, 'wb') as f:
            pickle.dump(each_year, f)
        logging.info(f"Results saved for year {year}")
        
        del eydmtr, eydmaxt, eydmint
        gc.collect()
        
        logging.info(f"Completed processing for year {year}")
    except Exception as e:
        logging.error(f"Error processing year {year}: {e}")
        logging.error(traceback.format_exc())


def process_temperature(t, realtime):
    latlen = t.lat.size 
    lonlen = t.lon.size
    tnp = t.to_numpy()
    lonnp = t.lon.to_numpy()
    latnp = t.lat.to_numpy()
    
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
                except Exception as e:
                    logging.error(f"Error in is_daytime: {e}")
                    tempt[num, 3] = 3
                    tempt[num, 4] = 26
                num += 1

    return tempt


        
# Multiprocessing function
def parallel_process(year_range):
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(cal_dstr, year_range), total=len(year_range)):
            pass


if __name__ == "__main__":
    time_start = time.time()

    path = r'/data2/share/metero/temperature/MERRA2/'
    outpath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_original_list/world_merra/'
    os.makedirs(outpath, exist_ok=True)
    
    years = np.arange(1980, 2024)
    parallel_process(years)
    
    time_end = time.time()
    time_sum = time_end - time_start
    logging.info(f"Total processing time: {time_sum} seconds")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
