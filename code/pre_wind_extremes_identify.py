# -*- coding: utf-8 -*-
import os
import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas as pd

# this file is for identifying extreme wind events

# def a function to combine outpath and outname
def combinename(outpath, outname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)

    return filepath


# def a function identify the target wind or rainfall
def isextreme(inpath, outpath, thr, key):
    # read path
    allfile = os.listdir(inpath)
    allfile.sort()
    for ff in allfile:
        num = 0
        result = np.zeros((1, 8))
        filepath = os.path.join(inpath, ff)
        # get time info use netCDF
        dataset = nc.Dataset(filepath, 'r')
        datatime = dataset.variables['time']
        datatime = nc.num2date(datatime, datatime.units)
        dataset.close()

        # read data use xarray
        dataset = xr.open_dataset(filepath, mask_and_scale=True)
        tmstart = 0
        tmend = len(datatime)
        for tm in range(tmstart, tmend):
            realtime = datatime[tm]
            da = dataset.wind.sel(time=str(realtime))

            # identify extreme rainfall or wind
            extreme = da.where(da >= thr)
            ext_stack = extreme.stack(x=['lon', 'lat'])
            ext_notnan = ext_stack[ext_stack.notnull()]
            datalen = len(ext_notnan)
            if datalen > 0:
                temptdata = np.zeros((datalen, 8))
                tempt_time = realtime.timetuple()
                temptdata[:, 0] = int(tempt_time.tm_year)
                temptdata[:, 1] = int(tempt_time.tm_mon)
                temptdata[:, 2] = int(tempt_time.tm_mday)
                temptdata[:, 3] = int(tempt_time.tm_hour)
                temptdata[:, 4] = int(tempt_time.tm_yday)
                for ind in range(datalen):
                    temptdata[ind, 5] = round(ext_notnan.x.values[ind][0], 2)
                    temptdata[ind, 6] = round(ext_notnan.x.values[ind][1], 2)
                    temptdata[ind, 7] = ext_notnan.values[ind]
                if num == 0:
                    result = temptdata
                    num += 1
                else:
                    result = np.vstack((result, temptdata))
        # save result
        df = pd.DataFrame(result, columns=['year', 'month', 'mday', 'hour', 'doy', 'lon', 'lat', 'value'])
        outpath1 = os.path.join(outpath, key)
        filename = key + '_' + ff[-9:-3] + '.csv'
        outname = combinename(outpath1, filename)
        df.to_csv(outname, sep=',')
        print(outname)

    return


# set path
windpath = r'/data1/wzheng/CMF/CMFdata/Data_forcing_03hr_010deg/Wind/'
outpath = r'/data1/fyliu/b_compound_hourly_rainfall_wind/data/'
os.makedirs(outpath, exist_ok=True)

# set threshold, 14 m/s
windth = 17.2
isextreme(windpath, outpath, windth, 'wind')