# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import stats
import seaborn as sns


# this file is for calculating temporal trends and p value at each scale based on ERA5-Land
# which will be combined and compared
# the visualization of them are also plotted here, although most of them are not shown in text

# def a function to combine outpath and outname
def combinename(outpath, outname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)
    return filepath


# basic set
# font family and size
plt.rcParams['font.size'] = 10.5
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['legend.fontsize'] = 12
# line width
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

clr = ['maroon', 'dimgray', 'steelblue']

for baseyear in [1950, 1980]:

    # path
    figpath = f'/data1/fyliu/a_temperature_range/result/fig_tempo_trend_{str(2020 - baseyear)}_yr/'
    lspath = f'/data1/fyliu/a_temperature_range/process_data/tempo_trend_{str(2020 - baseyear)}_yr/'
    os.makedirs(lspath, exist_ok=True)
    os.makedirs(figpath, exist_ok=True)

    # =============================================================================================
    # =============================================================================================
    # read DSTR data, iterate by region
    datapath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_dstr/'
    fall = os.listdir(datapath)
    fall.sort()
    filelen = len(fall)

    for i in range(filelen):
        # use for fig content
        ffile = fall[i]
        key = ffile.lower().split('_')
        if key[0] in ['hong', 'inner', 'north', 'south']:
            key = key[0:2]
            key = [word.capitalize() for word in key]
            key = ' '.join(key)
        else:
            key = key[0].capitalize()
        # use for file safe
        ff = ffile.split('_')
        if ff[0] in ['Hong', 'Inner', 'north', 'south']:
            ff = ff[0:2]
            ff = '_'.join(ff)
        else:
            ff = ff[0]

        # dstr
        path = os.path.join(datapath, ffile)
        data = pd.read_csv(path, sep=',', header=0, index_col=0)
        data = data[data['year'] >= baseyear]

        # cal max, mean & min DSTR series
        group_data = data.groupby('year')['dstr'].agg(['max', 'mean', 'min'])
        group_data = group_data.reset_index()

        # plot DSTR temporal trend
        fig1 = plt.figure(figsize=(4, 4), dpi=300)
        grid = GridSpec(1, 1, figure=fig1, left=0.17, bottom=0.07, right=0.98, top=0.98, wspace=0, hspace=0)
        ax1 = plt.subplot(grid[0, 0])
        col = ['max', 'mean', 'min']
        text_ls = []
        slope_p_ls = np.zeros((3, 2))
        for ind in range(3):
            # cal slope, intercept, r2 and p value
            slope, intercept, corr, pvalue, sterr = stats.linregress(x=group_data['year'], y=group_data[col[ind]])
            slope_p_ls[ind, 0] = slope
            slope_p_ls[ind, 1] = pvalue
            # only plot mean DSTR
            if ind == 1:
                ax1.plot(group_data['year'], group_data[col[ind]], linestyle='-', color=clr[ind], linewidth=2, alpha=0.5,
                         label=col[ind])
                sns.regplot(x=group_data['year'], y=group_data[col[ind]], ci=95, marker='o', color=clr[ind], ax=ax1,
                            scatter_kws={'s': 8}, line_kws={'linewidth': 2})
                textname = 'Slope = ' + str(round(slope * 100, 2)) + ' (\u2103/100 yr), $P$ = ' + str(round(pvalue, 2))

                ax1.set_xlabel('')
                ax1.set_ylabel('DSTR (\u2103)')
                # add text
                xlim = ax1.get_xlim()
                ylim = ax1.get_ylim()
                xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
                ypos = ylim[0] + 0.05 * (ylim[1] - ylim[0])
                ax1.text(xpos, ypos, textname, verticalalignment='top', horizontalalignment='left', color=clr[ind+1], fontsize=9)

                # save fig1
                outname = ff + '_tempo_trend.png'
                filepath = combinename(figpath, outname)
                plt.savefig(filepath, dpi=300)
                print(filepath)
                plt.close()


        # save result
        df = pd.DataFrame(slope_p_ls, index=['max', 'mean', 'min'],
                          columns=['slope', 'P'])
        outname = ff + '_tempo_trend_p.csv'
        filepath = combinename(lspath, outname)
        df.to_csv(filepath, sep=',')
        print(filepath)


