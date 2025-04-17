# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

# this file is for selecting the max/min/mean DSTR for each year of each scale and then combine then into a csv
# as well as radiation data

# def a function to combine outpath and outname
def combinename(outpath, outname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)
    
    return filepath
    
    

# def a function read csv data
# return, labels, label name, list, length=33, except Macau
# datals, combined data, list, length=n, according to the user and csvfile
def readcsv(path, datanum, istranspose, isindex):
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
        elif key[0]=='north':
            key = 'NH'
        elif key[0]=='south':
            key = 'SH'
        else:
            key = key[0].capitalize()
        labels[i] = key
        filepath = os.path.join(path, ff)
        # read data
        if isindex:
            data = pd.read_csv(filepath, sep=',', header=0, index_col=0)
        else:
            data = pd.read_csv(filepath, sep=',', header=0)
        data1 = data.iloc[:, 0:datanum]
        # save to list
        for j in range(datanum):
            datals[j].append(data1.iloc[:, j])
        i += 1

    # change to np.array
    for lsnum in range(datanum):
        combined_df = pd.concat(datals[lsnum], axis=1, ignore_index=True)
        combined_df.columns = labels
        datals[lsnum] = combined_df

    return labels, datals



# 1) multiyear DSTR data
path = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_dstr/'
datanum = 3
labels, datals = readcsv(path, datanum, 1, 1)
year_df = datals[0].iloc[:, 0].rename('year')
pro_data = datals[2]

max_values = pro_data.groupby(year_df).max().reset_index()
min_values = pro_data.groupby(year_df).min().reset_index()
mean_values = pro_data.groupby(year_df).mean().reset_index()

outdata = [max_values, min_values, mean_values]

# save csv
outpath = '/data1/fyliu/a_temperature_range/process_data/ridge_data/dstr'
key = ['max', 'min', 'mean']
for i in range(3):
    outname = key[i] + '_dstr.csv'
    filepath = combinename(outpath, outname)
    outdata[i].to_csv(filepath)
    print(filepath)
    
    
    
# 2) radiation & albedo
stmaxpath = f'/data1/fyliu/a_temperature_range/process_data/ridge_data/STmax_radiation/'
stminpath = f'/data1/fyliu/a_temperature_range/process_data/ridge_data/STmin_radiation/'
datanum = 9
labels1, datals1 = readcsv(stmaxpath, datanum, 1, 0)
labels2, datals2 = readcsv(stminpath, datanum, 1, 0)
outpath = f'/data1/fyliu/a_temperature_range/process_data/ridge_data/radiation'

varname = ['str', 'strd', 'ssrd', 'slhf', 'sshf', 'fal', 'str_earth', 'rad']
key = ['stmax', 'stmin', 'dstr']
for i, var in enumerate(varname):
    pro_data1 = datals1[i]
    pro_data2 = datals2[i]
    pro_data = pro_data1 - pro_data2
    
    pro_data_ls = [pro_data1, pro_data2, pro_data]
    for j in range(3):
        tempt = pro_data_ls[j].copy()
        tempt['year'] = np.arange(1950, 2024)
        outpath1 = os.path.join(outpath, key[j])
    
        # save csv
        outname = 'mean_' + key[j] + '_' + var +'.csv'
        filepath = combinename(outpath1, outname)
        tempt.to_csv(filepath)
        print(filepath)