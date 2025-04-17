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
import traceback
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# this file is for calculating DSTR in China based on China meteorological forcing dataset

# Initialize logging
logging.basicConfig(filename=r'/data1/fyliu/a_temperature_range/code/logging/data_processing_china_CMF.log', 
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
        
        eydmtr, eydmaxt, eydmint = [], [], []
        
        for month in range(datelen):
        
            filepath = os.path.join(path, sublist[month])
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
            emhmtr = np.zeros((testleng, 8))
            emhmaxt = list(range(testleng))
            emhmint = list(range(testleng))
            
            for tm in range(teststart, testend):
                # get time info according to tm, time index
                realtime = datatime[tm]
                ind = tm - teststart
                tempt_time = realtime.timetuple()
                emhmtr[ind, 0] = tempt_time.tm_year
                emhmtr[ind, 1] = tempt_time.tm_mon
                emhmtr[ind, 2] = tempt_time.tm_mday
                emhmtr[ind, 3] = tempt_time.tm_hour
                emhmtr[ind, 4] = tempt_time.tm_yday
                
                # Get max and min temperature, location, and day/night info
                da = dataset.temp.sel(time=str(realtime))
    
                # Max temperature
                t = da.where(da == da.max(), drop=True).squeeze()
                notnan = t.to_numpy()[~np.isnan(t.to_numpy())]
                emhmtr[ind, 5] = notnan[0] - 273.15
                emhmaxt[ind] = process_temperature(t, realtime)
    
                # Min temperature
                t = da.where(da == da.min(), drop=True).squeeze()
                notnan = t.to_numpy()[~np.isnan(t.to_numpy())]
                emhmtr[ind, 6] = notnan[0] - 273.15
                emhmint[ind] = process_temperature(t, realtime)
    
                # Max temperature range
                emhmtr[ind, 7] = emhmtr[ind, 5] - emhmtr[ind, 6]
                
                gc.collect()
    
            emdmtr, emdmaxt, emdmint = cal_emdstr(emhmtr, emhmaxt, emhmint)
            eydmtr = eydmtr + emdmtr
            eydmaxt = eydmaxt + emdmaxt
            eydmint = eydmint + emdmint
            
            del emdmtr, emdmaxt, emdmint, emhmtr, emhmaxt, emhmint
            gc.collect()
        
        # save result
        # print(eydmtr, eydmaxt, eydmint)
        each_year = [eydmtr, eydmaxt, eydmint]
        outname = f'china_eachyear_maxTrange_{year}.pkl'
        filepath = os.path.join(outpath, outname)
        with open(filepath, 'wb') as f:
            pickle.dump(each_year, f)
        logging.info(f"Results saved for year {year}")
        
        del each_year, eydmtr, eydmaxt, eydmint
        gc.collect()

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


# Function to save results
def cal_emdstr(emhmtr, emhmaxt, emhmint):
    maxdoy = int(emhmtr[:, 4].max())
    mindoy = int(emhmtr[:, 4].min())
    doy = maxdoy - mindoy + 1
    
    emdmtr = list(range(doy))
    emdmaxt = list(range(doy))
    emdmint = list(range(doy))
    for date in range(mindoy, maxdoy + 1):
        row_ind = np.where(emhmtr[:, 4] == date)
        tempt = emhmtr[row_ind[0], :]
        # print(tempt)
        temptmax = [emhmaxt[ind] for ind in row_ind[0]]
        temptmin = [emhmint[ind] for ind in row_ind[0]]
        maxTrange = tempt[:, 7].max()
        row_ind = np.where(tempt[:, 7] == maxTrange)
        # print(row_ind)
        dateind = date - mindoy
        emdmtr[dateind] = tempt[row_ind[0], :]
        emdmaxt[dateind] = [temptmax[ind] for ind in row_ind[0]] 
        emdmint[dateind] = [temptmin[ind] for ind in row_ind[0]]
        del tempt, temptmax, temptmin
        gc.collect()
    
    return emdmtr, emdmaxt, emdmint

        
# Multiprocessing function
def parallel_process(year_range):
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(cal_dstr, year_range), total=len(year_range)):
            pass


if __name__ == "__main__":
    time_start = time.time()

    path = r'/data1/wzheng/CMF/CMFdata/Data_forcing_03hr_010deg/Temp/'
    outpath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_original_list/china_CMF/'
    os.makedirs(outpath, exist_ok=True)
   
    years = np.arange(1979, 2019)
    parallel_process(years)
    
    time_end = time.time()
    time_sum = time_end - time_start
    logging.info(f"Total processing time: {time_sum} seconds")
    
    
    
    
