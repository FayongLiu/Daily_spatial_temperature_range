# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import stats


# this file is for comparison between different periods at each scale, as well as visualization based on MERRA2 and CFM
# only comparison between time series here

# def a function for changing test
def changetest(data1, data2):
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    absdiffer = mean2 - mean1
    reldiffer = (mean2 - mean1) / abs(mean1) * 100

    if len(data1) > len(data2):
        selected_indices = np.random.choice(len(data1), size=len(data2), replace=False)
        data1 = data1[selected_indices]
    elif len(data2) > len(data1):
        selected_indices = np.random.choice(len(data2), size=len(data1), replace=False)
        data2 = data2[selected_indices]
    # sighrank test
    statistic, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')

    return absdiffer, reldiffer, p_value


# def a function to combine outpath and outname
def combinename(outpath, outname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)
    return filepath


# def a function tranfer index to letter
def index_to_letter(index):
    base = ord('a') - 1
    if index <= 26:
        return chr(index + base)
    elif index <= 26 * 26:
        first_letter = chr((index - 1) // 26 + base)
        second_letter = chr((index - 1) % 26 + base)
        return first_letter + second_letter
    else:
        return ''


# def a function plot multi year change line
# return, mmmseries, 366*3 array, each columns, max, mean, min line
# mmmvalues, 1*12 array, max, mean, min, std values for each mmm line
def plotmulitline(axname1, axname2, axname3, linedata, clr, mk, labelname, ylabelname, pnum):
    # transfer to pandas array
    linedf = pd.DataFrame(linedata, columns=['year', 'doy', 'dss'])
    # cal max, mean, min series and value
    mmmseries = linedf.groupby('doy')['dss'].agg(['max', 'mean', 'min'])
    mmmvalue = mmmseries.agg(['max', 'mean', 'min', 'std'], axis=0)
    x = mmmseries.index.to_numpy().astype(int)
    mmmseries = mmmseries.values
    mmmvalue = mmmvalue.values.flatten('F')

    # plot mmm line
    mksz = 3
    axname1.plot(x, mmmseries[:, 0], linestyle='-', marker=mk, markersize=mksz, color=clr[1], linewidth=2,
                 alpha=0.5, label=labelname)
    axname2.plot(x, mmmseries[:, 1], linestyle='-', marker=mk, markersize=mksz, color=clr[0], linewidth=2,
                 alpha=0.5, label=labelname)
    axname3.plot(x, mmmseries[:, 2], linestyle='-', marker=mk, markersize=mksz, color=clr[2], linewidth=2,
                 alpha=0.5, label=labelname)

    # add seasonal line
    axname = [axname1, axname2, axname3]
    for i in range(3):
        for xvalue in [firsec, secthi, thifou, foufir]:
            axname[i].axvline(x=xvalue, color=clr[0], linestyle='--', linewidth=1.5, alpha=0.5)
        axname[i].set_xlim(0, 367)
        axname[i].set_xticks([firsec, secthi, thifou, foufir])
        axname[i].set_xticklabels([])
        ylabes = ylabelname[i] + ' (' + '\u2103' + ')'
        axname[i].set_ylabel(ylabes)

    # add pnum
    if len(pnum) != 0:
        for i in range(3):
            xlim = axname[i].get_xlim()
            ylim = axname[i].get_ylim()
            xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
            ypos = ylim[1] - 0.01 * (ylim[1] - ylim[0])
            axname[i].text(xpos, ypos, pnum[i], verticalalignment='top', horizontalalignment='left', fontweight='bold',
                           fontsize=12)

    return mmmseries, mmmvalue


# call function to plot
# set basic info
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
# seasonal day
firsec = 60
secthi = firsec + 92
thifou = secthi + 92
foufir = thifou + 91

clr1 = ['silver', 'lightcoral', 'lightblue']
clr2 = ['dimgray', 'brown', 'steelblue']

# =============================================================================================
# =============================================================================================
# read DSTR data, iterate by region
datapath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char_merra_cmf/multiyear_dstr/'
fall = os.listdir(datapath)
fall.sort()
filelen = len(fall)

keyname = ['china', 'china_cfm', 'world', 'world_meera']

for i in range(filelen):
    # use for fig content
    ffile = fall[i]

    # path
    for timestep in [10, 20, 30]:

        figpath = f'/data1/fyliu/a_temperature_range/result/fig_comparison_{timestep}_merra_cfm/'
        os.makedirs(figpath, exist_ok=True)
        # use for file safe
        ff = keyname[i]

        # dstr
        path = os.path.join(datapath, ffile)
        data = pd.read_csv(path, sep=',', header=0, index_col=0)
        data = data.values
        if i < 2:
            ind1 = (data[:, 0] >= 1979) & (data[:, 0] < (1979 + timestep))
            ind2 = (data[:, 0] > (2018 - timestep)) & (data[:, 0] <= 2018)
            fstlab = '1979-' + str(1978 + timestep)
            seclab = str(2019 - timestep) + '-2018'
        else:
            ind1 = (data[:, 0] >= 1980) & (data[:, 0] < (1980 + timestep))
            ind2 = (data[:, 0] > (2023 - timestep)) & (data[:, 0] <= 2023)
            fstlab = '1980-' + str(1980 + timestep)
            seclab = str(2024 - timestep) + '-2023'
        data1 = data[ind1, :]
        data2 = data[ind2, :]

        # figure1
        # plot compare DSTR
        fig1 = plt.figure(figsize=(8, 8), dpi=300)
        grid = GridSpec(3, 1, figure=fig1, left=0.08, bottom=0.05, right=0.98, top=0.92,
                        wspace=0, hspace=0.27)
        ax1 = plt.subplot(grid[0, 0])
        ax2 = plt.subplot(grid[1, 0])
        ax3 = plt.subplot(grid[2, 0])
        ylabelname = ['maxDSTR', 'meanDSTR', 'minDSTR']
        pnum = ['(a)', '(b)', '(c)']
        dataname = 'dstr'

        mmmseries1, mmmvalue1 = plotmulitline(ax1, ax2, ax3, data1, clr1, '.', fstlab, ylabelname, pnum)
        mmmseries2, mmmvalue2 = plotmulitline(ax1, ax2, ax3, data2, clr2, '.', seclab, ylabelname, '')

        # change test
        clen = 3
        abstext = list(np.arange(clen))
        reltext = list(np.arange(clen))
        # each column, absdiffer, relative differ and p value, each row, max, mean and min dstr/stmax/stmin
        changels = np.zeros((clen, 3))
        for j in range(clen):
            absdif, reldif, p = changetest(mmmseries1[:, j], mmmseries2[:, j])
            changels[j, 0] = absdif
            changels[j, 1] = reldif
            changels[j, 2] = p
            abstext[j] = 'Absolute change: ' + str(round(absdif, 2)) + '\u2103'
            reltext[j] = 'Relative change: ' + str(round(reldif, 2)) + '%' + ', P = ' + str(round(p, 2))

        # text
        ax1.text(1, ax1.get_ylim()[1] + 0.04 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), reltext[0])
        ax1.text(1, ax1.get_ylim()[1] + 0.14 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), abstext[0])
        ax2.text(1, ax2.get_ylim()[1] + 0.04 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]), reltext[1])
        ax2.text(1, ax2.get_ylim()[1] + 0.14 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]), abstext[1])
        ax3.text(1, ax3.get_ylim()[1] + 0.04 * (ax3.get_ylim()[1] - ax3.get_ylim()[0]), reltext[2])
        ax3.text(1, ax3.get_ylim()[1] + 0.14 * (ax3.get_ylim()[1] - ax3.get_ylim()[0]), abstext[2])

        # legend
        ax1.legend(ncol=2, bbox_to_anchor=(1.02, 0.95), loc='lower right', frameon=False)
        ax2.legend(ncol=2, bbox_to_anchor=(1.02, 0.95), loc='lower right', frameon=False)
        ax3.legend(ncol=2, bbox_to_anchor=(1.02, 0.95), loc='lower right', frameon=False)

        # add title and so on
        ax3.set_xticklabels(['Mar.', 'Jun.', 'Sep.', 'Dec.'])

        # save fig1
        outname = ff + '_compare_mmmchange.png'
        filepath = combinename(figpath, outname)
        plt.savefig(filepath, dpi=300)
        print(filepath)
        plt.close()






