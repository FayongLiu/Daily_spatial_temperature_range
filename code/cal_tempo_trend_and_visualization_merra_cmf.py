# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import stats
import seaborn as sns


# this file is for calculating temporal trends and p value at each scale based on MERRA2 and CMF
# which will be combined and compared
# the visualization of them are also plotted here, although most of them are not shown in text
# as well as the comparison between different dataset, i.e., Fig. S8


# def a function to combine outpath and outname
def combinename(outpath, outname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)
    return filepath


# def a function to add fignum
def addfignum(axn, fignum, ftsz_num):
    xlim = axn.get_xlim()
    ylim = axn.get_ylim()
    xpos = xlim[0]
    ypos = ylim[1]
    axn.text(xpos, ypos, fignum, verticalalignment='bottom', horizontalalignment='left', fontweight='bold',
             fontsize=ftsz_num)

    return


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
ftsz_num = 12

clr = ['maroon', 'dimgray', 'steelblue']

# path
figpath = r'/data1/fyliu/a_temperature_range/result/fig_tempo_trend_merra_cmf/'
lspath = r'/data1/fyliu/a_temperature_range/process_data/tempo_trend_merra_cmf/'
os.makedirs(lspath, exist_ok=True)
os.makedirs(figpath, exist_ok=True)

# =============================================================================================
# =============================================================================================
# read DSTR data, iterate by region
datapath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char_merra_cmf/multiyear_dstr/'
fall = os.listdir(datapath)
fall.sort()
filelen = len(fall)
syear = [1979, 1979, 1980, 1980]
eyear = [2019, 2019, 2024, 2024]
keyname = ['china_cmf', 'china', 'world_merrra', 'world']

for i in range(filelen):

    ffile = fall[i]
    ff = keyname[i]

    # dstr
    path = os.path.join(datapath, ffile)
    data = pd.read_csv(path, sep=',', header=0, index_col=0)
    data = data[(data['year'] >= syear[i]) & (data['year'] < eyear[i])]

    # cal max, mean & min DSTR series
    group_data = data.groupby('year')['dstr'].agg(['max', 'mean', 'min'])
    group_data = group_data.reset_index()

    # figure1
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
            ax1.text(xpos, ypos, textname, verticalalignment='top', horizontalalignment='left', color=clr[ind + 1], fontsize=9)
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


# comparison different dataset
data_ls = []
for i in range(filelen):
    ffile = fall[i]

    # dstr
    path = os.path.join(datapath, ffile)
    data = pd.read_csv(path, sep=',', header=0, index_col=0)
    data = data[(data['year'] >= syear[i]) & (data['year'] < eyear[i])]

    # cal max, mean & min DSTR series
    group_data = data.groupby('year')['dstr'].agg(['max', 'mean', 'min'])
    group_data = group_data.reset_index()

    data_ls.append(group_data)

fig1 = plt.figure(figsize=(8, 4), dpi=300)
grid = GridSpec(1, 2, figure=fig1, left=0.08, bottom=0.22, right=0.98, top=0.93, wspace=0.2, hspace=0)
ax_ls = [plt.subplot(grid[0, i]) for i in range(2)]
xpos = [[], []]
ypos = [[], []]
keyname = ['CMF', 'ERA5-Land', 'MERRA2', 'ERA5-Land']
fignum_ls = ['(a) China (1979-2018)', '(b) World (1980-2023)']
for ind in range(2):
    ax = ax_ls[ind]
    x1 = data_ls[ind * 2]['year']
    y1 = data_ls[ind * 2]['mean']
    x2 = data_ls[ind * 2 + 1]['year']
    y2 = data_ls[ind * 2 + 1]['mean']
    x = [x1, x2]
    y = [y1, y2]
    clr = ['dimgray', 'steelblue']
    lab = keyname[ind * 2:(ind + 1) * 2]
    text_ls = []
    for j in range(2):
        ax.plot(x[j], y[j], linestyle='-', color=clr[j], linewidth=2, alpha=0.5, label=lab[j])
        sns.regplot(x=x[j], y=y[j], ci=95, marker='o', color=clr[j], ax=ax, scatter_kws={'s': 8},
                    line_kws={'linewidth': 2})

        # cal slope, intercept, r2 and p value
        slope, intercept, corr, pvalue, sterr = stats.linregress(x=x[j], y=y[j])

        # add slope, intercept, r2 and p value
        textname = lab[j] + ', Slope = ' + str(round(slope * 100, 2)) + ' (\u2103/100 yr), $P$ = ' + str(round(pvalue, 2))
        text_ls.append(textname)
    xlim0 = ax.get_xlim()
    ylim0 = ax.get_ylim()
    for j in range(2):
        ypos0 = ylim0[0] - (0.2 + 0.08 * j) * (ylim0[1] - ylim0[0])
        ax.text(xlim0[0], ypos0, text_ls[j], color=clr[j], fontsize=9)

    ax.set_xlabel('Year')
    ax.set_ylabel('DSTR (\u2103)')
    ax.legend(ncol=1, loc='best', frameon=False, fontsize=9)

    addfignum(ax, fignum_ls[ind], ftsz_num)

# save fig1
filepath = r'/data1/fyliu/a_temperature_range/result/fig_main/Fig_S8_Trend_of_DSTR_based_on_different_dataset.png'
plt.savefig(filepath, dpi=300)
print(filepath)
plt.close()