# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import pandas as pd
import seaborn as sns

# this file plots temporal trend at different scale during 1980-2018

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
    xpos = xlim[0]  # + 0.03*(xlim[1] - xlim[0])
    ypos = ylim[1]  # - 0.03*(ylim[1] - ylim[0])
    axn.text(xpos, ypos, fignum, verticalalignment='bottom', horizontalalignment='left', fontweight='bold',
             fontsize=ftsz_num)

    return


# def a function plot temporal trend
def plot_tempo_trend_one(axname, trend_ls, clrname, ylab, lab, baseyear, endyear, pnum):
    unit = ' (\u2103/100 yr)'
    scale_f = 100

    x = np.array(range(baseyear, endyear + 1))

    for ind in range(1):
        # 74 years trend
        axname.plot(x, trend_ls[ind], linestyle='-', color=clrname[ind], linewidth=2, alpha=0.5, label=lab[ind])
        sns.regplot(x=x, y=trend_ls[ind], ci=95, marker='o', color=clrname[ind], ax=axname, scatter_kws={'s': 8},
                    line_kws={'linewidth': 2})
        # cal slope, intercept, r2 and p value
        slope, intercept, corr, pvalue, sterr = stats.linregress(x=x, y=trend_ls[ind])

        # add slope, intercept, r2 and p value
        textname = 'Slope = ' + str(round(slope * scale_f, 2)) + unit + ', $P$ = ' + str(round(pvalue, 2))
        xlim = axname.get_xlim()
        ylim = axname.get_ylim()
        xpos = xlim[0] + 0.05 * (xlim[1] - xlim[0])
        ypos = ylim[1] - 0.05 * (ylim[1] - ylim[0])
        axname.text(xpos, ypos, textname, verticalalignment='top', horizontalalignment='left', color='steelblue',
                    fontsize=ftsz)

    axname.set_ylabel(ylab)
    axname.set_xlabel('Year')
    axname.set_xlim(baseyear - 1, endyear + 1)
    axname.set_xticks(range(baseyear, endyear + 1, 10))

    addfignum(axname, pnum, ftsz_num)

    return


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
        if key == 'Tibet':
            labels[i] = 'Xizang'
        else:
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


# def a function plot bar
def plotbar(axname, slopedata, clrname, ylabelname, sigp, pnum, isfill):
    x = np.arange(len(slopedata))
    width = 0.35
    if isfill == 0:
        rects = axname.bar(x, slopedata, width, linewidth=1.5, color='none', edgecolor=clrname, alpha=alp)
    else:
        rects = axname.bar(x, slopedata, width, linewidth=1.5, color=clrname, edgecolor=clrname, alpha=alp)

    axname.axhline(y=0, xmin=np.min(x) - 1, xmax=np.max(x) + 1, color='k', linestyle='-', linewidth=1)

    # xylabel
    axname.set_xlim(np.min(x) - 1, np.max(x) + 1)
    axname.set_xticks(x)
    axname.set_xticklabels([])
    axname.set_ylabel(ylabelname)

    rectnum = 0
    yheight = axname.get_ylim()[1] - axname.get_ylim()[0]
    for rect in rects:
        if sigp[rectnum] < 0.05:
            if slopedata[rectnum] >= 0:
                axname.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005 * yheight, '*', ha='center')
            else:
                axname.text(rect.get_x() + rect.get_width() / 2, rect.get_height() - 0.05 * yheight, '*', ha='center')
        rectnum = rectnum + 1

    axname.text(0.5, 1.8, '* $P$ < 0.05')

    # set pnum
    addfignum(axname, pnum, ftsz_num)

    return


# basic set
# font family and size
plt.rcParams["font.family"] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
# line width
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
ftsz = 9
ftsz_num = 12
alp = 0.65

outpath = r'/data1/fyliu/a_temperature_range/result/fig_main/'

fig = plt.figure(figsize=(13, 7), dpi=300)
grid = GridSpec(2, 4, figure=fig, left=0.06, bottom=0.17, right=0.98, top=0.97, wspace=0.27, hspace=0.3)
ax_ls = [plt.subplot(grid[0, j]) for j in range(4)]
ax_ls.append(plt.subplot(grid[1, 0:]))

# plot DSTR, STmax, STmin trend
# =============================================================================================
# read DSTR data, iterate by region
varname = ['dstr', 'stmax', 'stmin']
all_dstr = {}
for baseyear, endyear in [(1980, 2018)]:

    # read DSTR data, iterate by region
    path1 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_dstr/'
    path2 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_stmax/'
    path3 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_stmin/'

    datapath = [path1, path2, path3]
    fall_ls = []
    for i in range(3):
        fall = os.listdir(datapath[i])
        fall.sort()
        filelen = len(fall)
        fall_ls.append(fall)

    for i in range(filelen):
        # use for fig content
        ffile = fall_ls[0][i]
        key = ffile.lower().split('_')
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

        data_ls = []
        for j in range(3):
            path = os.path.join(datapath[j], fall_ls[j][i])
            data = pd.read_csv(path, sep=',', header=0, index_col=0)
            data = data[(data['year'] >= baseyear) & (data['year'] <= endyear)]

            # cal max, mean & min DSTR series
            group_data_tempt = data.groupby('year')[varname[j]].mean()

            data_ls.append(group_data_tempt)

        group_data = pd.concat(data_ls, axis=1).reset_index()
        all_dstr[key] = group_data

# plot global, nh, sh and china trend
scale_name = ['World', 'SH', 'NH', 'China']
dstr_ls = [all_dstr.get(name) for name in scale_name]
clrname = ['dimgray', 'maroon', 'steelblue']
ylab = 'DSTR (\u2103)'
lab = ['DSTR', 'STmax', 'STmin']
pnum_ls = ['(a) World', '(b) SH', '(c) NH', '(d) China']

for i in range(4):
    axname = ax_ls[i]
    trend_ls = [dstr_ls[i][var] for var in varname]

    slope_arr = plot_tempo_trend_one(axname, trend_ls, clrname, ylab, lab, baseyear, endyear, pnum_ls[i])

# plot provincial trend
# ==========================================================================
path = r'/data1/fyliu/a_temperature_range/process_data/tempo_trend_40_yr/'
# 0-2 reperensts slope and P
datanum = 2
labels, datals = readcsv(path, datanum, 0)
slopels = datals[0][:33]
pls = datals[1][:33]

ylabel = 'Slope (\u2103/100 yr)'
plotbar(ax_ls[4], slopels[:, 1] * 100, clrname[0], ylabel, pls[:, 1], '(e) Provinces in China', 1)
ax_ls[4].set_xticklabels(labels[:33], rotation=-75)

outname = 'Fig S12 Trend of DSTR and factors at different scales during 1980 2018.png'
outname = outname.replace(' ', '_')
filename = combinename(outpath, outname)
plt.savefig(filename, dpi=300)
plt.close()
print(filename)




























