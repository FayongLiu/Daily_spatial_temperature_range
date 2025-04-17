# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import pandas as pd
import geopandas as gpd


# this file is for calculating the characteristics of DSTR based on MERRA2 and CMF
# the results of them (i.e., at different scale) will be combined and compared
# as well as the visualization of them, although most of them are not show in the text
# including:
# 1) temoral changes of DSTR, STmax, STmin at each scale
# 2) hour distribution (i.e., when DSTR occur duing a day) at each scale
# 3) distribution of STmax and STmin at each scale


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


# def a function conbine data during specific period
# return, dstr, stmax, stmin, n*3 array, each columns, year, doy, dstr/stmax/stmin
# hourdis, n*3 array, each columns, year, doy, hour
# maxdis, mindis, n*7 array, each columns, stmax/stmin, lon, lat, d/n, localhour, doy, year
def combinedata(path, yearrange):
    # read file
    allfile = os.listdir(path)
    allfile.sort()

    # get DSTR, STmax, STmin, Hour distribution data, STmax location and STmin location data
    dstr = []
    stmax = []
    stmin = []
    hourdis = []
    maxdis = []
    mindis = []
    for ffilename in allfile:
        # get year
        if ffilename.endswith('.pkl'):
            f_tempt = os.path.splitext(ffilename)[0]
        else:
            f_tempt = ffilename
        year = int(f_tempt[-4:])
        if year not in yearrange:
            continue
        else:
            # read data
            ffilepath = os.path.join(path, ffilename)
            with open(ffilepath, "rb") as f:
                eachyear = pickle.load(f)
            # dstr, stmax, stmin, hourdis
            daynum = 1
            for day in eachyear[0]:
                if daynum == 1:
                    trtemp = day[0, [0, 4, 7]]
                    maxtemp = np.mean(day[:, [0, 4, 5]], axis=0)
                    mintemp = np.mean(day[:, [0, 4, 6]], axis=0)
                    hourtemp = day[:, [0, 4, 3]]
                    daynum = daynum + 1
                else:
                    trtemp = np.vstack((trtemp, day[0, [0, 4, 7]]))
                    maxtemp = np.vstack((maxtemp, np.mean(day[:, [0, 4, 5]], axis=0)))
                    mintemp = np.vstack((mintemp, np.mean(day[:, [0, 4, 6]], axis=0)))
                    hourtemp = np.vstack((hourtemp, day[:, [0, 4, 3]]))
                    daynum = daynum + 1
            dstr.append(trtemp)
            stmax.append(maxtemp)
            stmin.append(mintemp)
            hourdis.append(hourtemp)
            # stmax location
            daynum = 1
            isfirst = 1
            for day in eachyear[1]:
                for hour in day:
                    if isfirst == 1:
                        yeartemp = year * np.ones((len(hour[:, 0]), 1))
                        doytemp = daynum * np.ones((len(hour[:, 0]), 1))
                        loctemp = np.hstack((hour, doytemp, yeartemp))
                        isfirst = isfirst + 1
                    else:
                        yeartemp = year * np.ones((len(hour[:, 0]), 1))
                        doytemp = daynum * np.ones((len(hour[:, 0]), 1))
                        loctemp1 = np.hstack((hour, doytemp, yeartemp))
                        loctemp = np.vstack((loctemp, loctemp1))
                daynum = daynum + 1
            maxdis.append(loctemp)
            # stmin location
            daynum = 1
            isfirst = 1
            for day in eachyear[2]:
                for hour in day:
                    if isfirst == 1:
                        yeartemp = year * np.ones((len(hour[:, 0]), 1))
                        doytemp = daynum * np.ones((len(hour[:, 0]), 1))
                        loctemp = np.hstack((hour, doytemp, yeartemp))
                        isfirst = isfirst + 1
                    else:
                        yeartemp = year * np.ones((len(hour[:, 0]), 1))
                        doytemp = daynum * np.ones((len(hour[:, 0]), 1))
                        loctemp1 = np.hstack((hour, doytemp, yeartemp))
                        loctemp = np.vstack((loctemp, loctemp1))
                daynum = daynum + 1
            mindis.append(loctemp)

    # change to array
    # dstr
    daynum = 1
    for ls in dstr:
        if daynum == 1:
            temparray = ls
            daynum = daynum + 1
        else:
            temparray = np.vstack((temparray, ls))
            daynum = daynum + 1
    dstr = temparray
    # stmax
    daynum = 1
    for ls in stmax:
        if daynum == 1:
            temparray = ls
            daynum = daynum + 1
        else:
            temparray = np.vstack((temparray, ls))
            daynum = daynum + 1
    stmax = temparray
    # stmin
    daynum = 1
    for ls in stmin:
        if daynum == 1:
            temparray = ls
            daynum = daynum + 1
        else:
            temparray = np.vstack((temparray, ls))
            daynum = daynum + 1
    stmin = temparray
    # hourdis
    daynum = 1
    for ls in hourdis:
        if daynum == 1:
            temparray = ls
            daynum = daynum + 1
        else:
            temparray = np.vstack((temparray, ls))
            daynum = daynum + 1
    hourdis = temparray
    # maxdis
    daynum = 1
    for ls in maxdis:
        if daynum == 1:
            temparray = ls
            daynum = daynum + 1
        else:
            temparray = np.vstack((temparray, ls))
            daynum = daynum + 1
    maxdis = temparray
    # mindis
    daynum = 1
    for ls in mindis:
        if daynum == 1:
            temparray = ls
            daynum = daynum + 1
        else:
            temparray = np.vstack((temparray, ls))
            daynum = daynum + 1
    mindis = temparray
    return dstr, stmax, stmin, hourdis, maxdis, mindis


# def a function plot multi year change line
# return, mmmseries, 366*3 array, each columns, max, mean, min line
# mmmvalues, 1*12 array, max, mean, min, std values for each mmm line
def plotmulitline(axname, linedata, ylabelname, pnum):
    # plot each year
    yearnum = np.unique(linedata[:, 0])
    yearlen = len(yearnum)
    for i in range(yearlen):
        linedata1 = linedata[linedata[:, 0] == yearnum[i], 1:]
        # clr0 = ((yearlen+1-i)/(yearlen+2), (yearlen+1-i)/(yearlen+2), (yearlen+1-i)/(yearlen+2))
        axname.plot(linedata1[:, 0], linedata1[:, 1], color='darkgrey', linewidth=0.5, alpha=0.2)

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
    ll1, = axname.plot(x, mmmseries[:, 0], linestyle='-', color=clr[1], linewidth=1.5,
                       alpha=0.5, marker='.', markersize=mksz)
    ll2, = axname.plot(x, mmmseries[:, 1], linestyle='-', color=clr[0], linewidth=1.5,
                       alpha=0.5, marker='.', markersize=mksz)
    ll3, = axname.plot(x, mmmseries[:, 2], linestyle='-', color=clr[2], linewidth=1.5,
                       alpha=0.5, marker='.', markersize=mksz)
    # add seasonal line
    for xvalue in [firsec, secthi, thifou, foufir]:
        axname.axvline(x=xvalue, color=clr[0], linestyle='--', linewidth=1, alpha=0.5)
    # test mmm value
    tt1 = 'Max line (max, ' + str(round(mmmvalue[0], 2)) + ', mean, ' + str(round(mmmvalue[1], 2)) + ', min, ' + str(
        round(mmmvalue[2], 2)) + ', std, ' + str(round(mmmvalue[3], 2)) + ')'
    tt2 = 'Mean line (max, ' + str(round(mmmvalue[4], 2)) + ', mean, ' + str(round(mmmvalue[5], 2)) + ', min, ' + str(
        round(mmmvalue[6], 2)) + ', std, ' + str(round(mmmvalue[7], 2)) + ')'
    tt3 = 'Min line (max, ' + str(round(mmmvalue[8], 2)) + ', mean, ' + str(round(mmmvalue[9], 2)) + ', min, ' + str(
        round(mmmvalue[10], 2)) + ', std, ' + str(round(mmmvalue[11], 2)) + ')'
    # add legengd and so on
    axname.legend(handles=[ll1, ll2, ll3], labels=[tt1, tt2, tt3], bbox_to_anchor=(0, -0.07), loc='upper left',
                  frameon=False)
    axname.set_xlim(0, 367)
    axname.set_xticks([firsec, secthi, thifou, foufir])
    ylabes = ylabelname + ' (' + '\u2103' + ')'
    axname.set_ylabel(ylabes)
    xlim = axname.get_xlim()
    ylim = axname.get_ylim()
    xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
    ypos = ylim[1] - 0.01 * (ylim[1] - ylim[0])
    axname.text(xpos, ypos, pnum, verticalalignment='top', horizontalalignment='left', fontweight='bold', fontsize=12)

    return mmmseries, mmmvalue


# def a function calculate mode
def group_mode(group):
    return group.mode()


# def a function plot hour distribution
# return, modehour, pandas series, index, doy, columns, multiyear mode hour distribution
# hournum, 1*24array, each columns, 0-23 hour
def plothourdis(axname1, axname2, hourdata):
    # plot hour distribution
    axname1.plot(hourdata[:, 1], hourdata[:, 2], 'o', markersize=3, markerfacecolor='none',
                 markeredgewidth=0.5, markeredgecolor=clr[0], alpha=0.5)
    # add season line
    for xvalue in [firsec, secthi, thifou, foufir]:
        axname1.axvline(x=xvalue, color=clr[0], linestyle='--', linewidth=0.5, alpha=0.3)
        # add reference line
    for yvalue in range(0, 19, 6):
        axname1.axhline(y=yvalue, color=clr[0], linestyle='--', linewidth=0.5, alpha=0.3)
    # cal multiyear average hour dis
    hourdf = pd.DataFrame(hourdata, columns=['year', 'doy', 'hour'])
    # cal mode hour
    modehour = hourdf.groupby('doy')['hour'].apply(group_mode)
    modehour.plot(kind='line', x='doy', y='hour', ax=axname1, color=clr[2], marker='.', markersize=3, linestyle='-',
                  linewidth=1.5, alpha=0.5, label='Mode hour')

    # set xy axis & title
    axname1.set_xlim(0, 367)
    axname1.set_xticks([firsec, secthi, thifou, foufir])
    axname1.set_xticklabels(['Mar.', 'Jun.', 'Sep.', 'Dec.'])
    axname1.set_xlabel('Day of year')
    axname1.set_ylim(-1, 24)
    axname1.set_yticks([0, 6, 12, 18])
    axname1.set_ylabel('Hour of day')
    axname1.legend(bbox_to_anchor=(0, 0.94), loc='lower left', frameon=False)

    # cal the times of each hour
    x = np.arange(24)
    hournum = np.zeros(24)
    for hour in range(24):
        hournum[hour] = np.sum(hourdata[:, 2] == hour)
    # plot stack bin
    axname2.barh(x, hournum, color=clr[2], alpha=0.5)
    maxnum = 0
    for hour in range(24):
        text = str(int(hournum[hour]))
        axname2.text(hournum[hour], x[hour], text, ha='left', va='center', c=clr[0], fontsize=8)
        if maxnum < hournum[hour]:
            maxnum = hournum[hour]
    axname2.set_xlim(0, maxnum + 500)
    axname2.set_ylim(-1, 24)
    axname2.set_yticks([])
    axname2.set_xlabel('Frequency')

    return modehour, hournum


# def a function plot location and day/night
# return, dnperc, 2*1array, night£¬ day percentile, %
# lonnum, latnum, n*2array, lon/lat value, count
def plotloc(axname1, axname2, axname3, locdata, lab, mkr, locclr, barnum):
    # add basemap
    wid = 0.3
    ssz = 2
    alp = 0.3
    if pi >= 2:
        gdf = worldgdf
        min_lon, max_lon = -180, 180
        min_lat, max_lat = -90, 90
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]
        gdf.plot(ax=axname1, color='black', linewidth=1.0)
    else:
        gdf = chinagdf
        gdf.plot(ax=axname1, color='black', linewidth=1.0)

    # cal the percent of day or night
    dnperc = np.zeros((2, 1))
    for i in range(2):
        dnperc[i, 0] = np.sum(locdata[:, 3] == i)
    # tranfter to percent
    dnsum = np.sum(dnperc)
    for i in range(2):
        dnperc[i, 0] = dnperc[i, 0] / dnsum * 100

    # plot tmax tmin location
    datalen = len(locdata[:, 0])
    for j in range(datalen):
        axname1.plot(locdata[j, 1], locdata[j, 2], mkr, markerfacecolor=locclr,
                     markeredgecolor=locclr, markeredgewidth=wid,
                     markersize=ssz, alpha=alp)

    # cal the number of lon and lat
    unique_values, counts = np.unique(locdata[:, 1], return_counts=True)
    lonnum = np.column_stack((unique_values, counts))
    unique_values, counts = np.unique(locdata[:, 2], return_counts=True)
    latnum = np.column_stack((unique_values, counts))

    # plot lonnum and latnum
    width = 0.05
    axname2.bar(lonnum[:, 0] - width / 2, lonnum[:, 1], width, linewidth=0.5, facecolor=locclr, edgecolor=locclr,
                alpha=0.5)

    if barnum == 0:
        axname3.barh(latnum[:, 0] - width / 2, latnum[:, 1], width, linewidth=0.5, facecolor=locclr, edgecolor=locclr,
                     alpha=0.5, label='STmax')
    else:
        axname3.barh(latnum[:, 0] + width / 2, latnum[:, 1], width, linewidth=0.5, facecolor=locclr, edgecolor=locclr,
                     alpha=0.5, label='STmin')

    return dnperc, lonnum, latnum


# call function to plot
# set basic info
# path
datapath1 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_original_list/china/'
datapath2 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_original_list/china_CMF'

datapath3 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_original_list/world/'
datapath4 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_original_list/world_merra/'

datapath = [datapath1, datapath2, datapath3, datapath4]
syear = [1979, 1979, 1980, 1980]
eyear = [2019, 2019, 2024, 2024]
keyname = ['china', 'china_cmf', 'world', 'world_merrra']

figpath = r'/data1/fyliu/a_temperature_range/result/fig_DSTR_char_merra_cmf/'
lspath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char_merra_cmf/'
os.makedirs(lspath, exist_ok=True)
os.makedirs(figpath, exist_ok=True)

worldshp = r'/data1/fyliu/a_temperature_range/data/boundary/global_coastline_110m/global_coastline_110m.shp'
chinashp = r'/data1/fyliu/a_temperature_range/data/boundary/China_GS(2020)4619/China_provincial_line.shp'

# basic set
# font family and size
# plt.rcParams['font.family'] = 'Arial'
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

# color
clr = ['dimgray', 'maroon', 'steelblue', 'darkgreen']
# seasonal day
firsec = 60
secthi = firsec + 92
thifou = secthi + 92
foufir = thifou + 91

# read gdf
worldgdf = gpd.read_file(worldshp)
chinagdf = gpd.read_file(chinashp)

# read data, iterate by region
for pi in range(4):

    path = datapath[pi]
    ffile = keyname[pi]
    yearrange = list(range(syear[pi], eyear[pi]))

    # combine data
    dstr, stmax, stmin, hourdis, maxdis, mindis = combinedata(path, yearrange)
    # transfer UTC to Beijing time
    hourlen = len(hourdis[:, 2])
    if pi < 2:
        for i in range(hourlen):
            hourdis[i, 2] += 8
            if hourdis[i, 2] >= 24:
                hourdis[i, 2] -= 24
                # save result
    # dstr, stmax, stmin
    dss = [dstr, stmax, stmin, hourdis]
    dssstr = ['dstr', 'stmax', 'stmin', 'hourdis']
    for i in range(4):
        df = pd.DataFrame(dss[i], columns=['year', 'doy', dssstr[i]])
        outpath = lspath + 'multiyear_' + dssstr[i]
        outname = ffile + '_multiyear_' + dssstr[i] + '.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')
    # maxdis, mindis
    mm = [maxdis, mindis]
    mmstr = ['maxdis', 'mindis']
    col = ['stmax', 'stmin']
    for i in range(2):
        df = pd.DataFrame(mm[i], columns=[col[i], 'lon', 'lat', 'dayORnight', 'localhour', 'doy', 'year'])
        outpath = lspath + 'multiyear_' + mmstr[i]
        outname = ffile + '_multiyear_' + mmstr[i] + '.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')

    # ================================================
    # ================================================
    # figure1, plot multi year change line
    fig1 = plt.figure(figsize=(8, 8), dpi=300)
    grid = GridSpec(3, 1, figure=fig1, left=0.08, bottom=0.12, right=0.98, top=0.98,
                    wspace=0, hspace=0.6)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[2, 0])
    ylabelname = ['DSTR', 'STmax', 'STmin']
    pnum = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    mmmseries1, mmmvalue1 = plotmulitline(ax1, dstr, ylabelname[0], pnum[0])
    mmmseries2, mmmvalue2 = plotmulitline(ax2, stmax, ylabelname[1], pnum[1])
    mmmseries3, mmmvalue3 = plotmulitline(ax3, stmin, ylabelname[2], pnum[2])
    mmmseries = np.hstack((mmmseries1, mmmseries2, mmmseries3))
    mmmvalue = np.vstack((mmmvalue1, mmmvalue2, mmmvalue3)).T
    # add title and so on
    # tit = 'Multiyear DSTR change in ' + key
    # ax1.set_title(tit)
    ax1.set_xticklabels(['Mar.', 'Jun.', 'Sep.', 'Dec.'])
    ax2.set_xticklabels(['Mar.', 'Jun.', 'Sep.', 'Dec.'])
    ax3.set_xticklabels(['Mar.', 'Jun.', 'Sep.', 'Dec.'])
    # ax3.set_xlabel('Day of year')

    # save results
    # mmm series
    df = pd.DataFrame(mmmseries, index=np.arange(1, 367),
                      columns=['max DSTR', 'mean DSTR', 'min DSTR',
                               'max STmax', 'mean STmax', 'min STmax',
                               'max STmin', 'mean STmin', 'min STmin'])
    outpath = lspath + 'multiyear_mmmseries'
    outname = ffile + '_multiyear_mmmseries.csv'
    filepath = combinename(outpath, outname)
    df.to_csv(filepath, sep=',')
    # mmm values
    df = pd.DataFrame(mmmvalue, index=['maxmax', 'meanmax', 'minmax', 'stdmax', 'maxmean',
                                       'meanmean', 'minmean', 'stdmean', 'maxmin', 'meanmin', 'minmin', 'stdmin'],
                      columns=['DSTR', 'STmax', 'STmin'])
    outpath = lspath + 'multiyear_mmmvalue'
    outname = ffile + '_multiyear_mmmvalue.csv'
    filepath = combinename(outpath, outname)
    df.to_csv(filepath, sep=',')
    # fig1
    outname = ffile + '_multiyear_mmmchange.png'
    filepath = combinename(figpath, outname)
    plt.savefig(filepath, dpi=300)
    print(filepath)
    plt.close()

    # ================================================
    # ================================================
    # figure2, plot hour and location distribution
    fig2 = plt.figure(figsize=(8, 3), dpi=300)
    grid = GridSpec(1, 10, figure=fig2, left=0.12, bottom=0.15, right=0.98, top=0.92,
                    wspace=0, hspace=0.3)
    ax4 = plt.subplot(grid[0, 0:7])
    ax5 = plt.subplot(grid[0, 7:10])

    # plot hour distribution
    modehour, hournum = plothourdis(ax4, ax5, hourdis)

    # save result
    # modehour
    outpath = lspath + 'multiyear_modehour'
    outname = ffile + '_multiyear_modehour.csv'
    filepath = combinename(outpath, outname)
    modehour.to_csv(filepath, header=['hour'], sep=',')
    # hourmun
    df = pd.DataFrame(hournum, index=range(24), columns=['hournum'])
    outpath = lspath + 'multiyear_hournum_un'
    outname = ffile + '_multiyear_huornum_un.csv'
    filepath = combinename(outpath, outname)
    df.to_csv(filepath, sep=',')

    # save picture
    outname = ffile + '_multiyear_distribution_hour.png'
    filepath = combinename(figpath, outname)
    plt.savefig(filepath, dpi=300)
    print(filepath)
    plt.close()

    # ================================================
    # ================================================
    # figure3, plot location distribution
    # maxdis
    fig3 = plt.figure(figsize=(8, 4), dpi=300)
    grid = GridSpec(6, 10, figure=fig3, left=0.1, bottom=0.05, right=0.85, top=0.99,
                    wspace=0, hspace=0.3)
    ax6 = plt.subplot(grid[1:6, 0:9])
    ax7 = plt.subplot(grid[0, 0:9])
    ax8 = plt.subplot(grid[1:6, 9])

    dnperc1, lonnum1, latnum1 = plotloc(ax6, ax7, ax8, maxdis, 'STmax', 'o', clr[1], 0)
    dnperc2, lonnum2, latnum2 = plotloc(ax6, ax7, ax8, mindis, 'STmin', 'o', clr[2], 1)

    # set xylim and tick label
    if pi >= 2:
        ax6.set_xlim(-180, 180)
        ax6.set_ylim(-90, 90)
        ticks_x = np.arange(-180, 180.001, 60)
        labels_x = [f'{abs(round(tick, 1))}\xb0{"E" if tick > 0 else "W" if tick < 0 else " "}' for tick in ticks_x]
        ax6.set_xticks(ticks_x)
        ax6.set_xticklabels(labels_x)
        ticks_y = np.arange(-90, 90.001, 30)
        labels_y = [f'{abs(round(tick, 1))}\xb0{"N" if tick > 0 else "S" if tick < 0 else " "}' for tick in ticks_y]
        labels_y[0] = " "
        labels_y[-1] = " "
        ax6.set_yticks(ticks_y)
        ax6.set_yticklabels(labels_y)
    else:
        ticks_x = ax6.get_xticks()
        labels_x = [f'{abs(round(tick, 1))}\xb0{"E" if tick > 0 else "W" if tick < 0 else " "}' for tick in ticks_x]
        for ll in range(len(labels_x)):
            if ll % 2 != 0:
                labels_x[ll] = " "
        ax6.set_xticks(ticks_x)
        ax6.set_xticklabels(labels_x)
        ticks_y = ax6.get_yticks()
        labels_y = [f'{abs(round(tick, 1))}\xb0{"N" if tick > 0 else "S" if tick < 0 else " "}' for tick in ticks_y]
        labels_y[0] = " "
        labels_y[-1] = " "
        ax6.set_yticks(ticks_y)
        ax6.set_yticklabels(labels_y)

    for xvalue in ticks_x:
        ax6.axvline(x=xvalue, color=clr[0], linestyle='--', linewidth=0.5, alpha=0.3)
    for yvalue in ticks_y:
        ax6.axhline(y=yvalue, color=clr[0], linestyle='--', linewidth=0.5, alpha=0.3)

    xlim = ax6.get_xlim()
    ax7.set_xlim(xlim)
    ax7.set_xticks(ticks_x)
    ax7.set_xticklabels([])
    ticks_y = ax7.get_yticks()
    labels_y = [f'{str(int(tick)) if tick > 0 else " "}' for tick in ticks_y]
    ax7.set_yticks(ticks_y)
    ax7.set_yticklabels(labels_y, fontsize=8)
    ax7.set_ylabel('Frequency', fontsize=8)

    ylim = ax6.get_ylim()
    ax8.set_ylim(ylim)
    ticks_y = ax6.get_yticks()
    ax8.set_yticks(ticks_y)
    ax8.set_yticklabels([])
    ticks_x = ax8.get_xticks()
    labels_x = [f'{str(int(tick)) if tick > 0 else " "}' for tick in ticks_x]
    ax8.set_xticks(ticks_x)
    ax8.set_xticklabels(labels_x, fontsize=8)
    ax8.set_xlabel('Ffrequency', fontsize=8)
    # add title and legend
    # ax7.set_title('Geographic location of STmax and STmin in ' + key)
    # ax6.legend(ncol=2, bbox_to_anchor = (0.5, -0.05), loc='upper center', frameon=False)
    ax8.legend(ncol=1, bbox_to_anchor=(0.95, 1), loc='upper left', frameon=False)
    ax8.xaxis.tick_top()
    ax8.xaxis.set_label_position('top')

    # adjust ax position
    pos = ax6.get_position()
    pos1 = ax7.get_position()
    pos2 = ax8.get_position()
    ax7.set_position([pos.x0, pos.y0 + pos.height, pos.width, pos1.height])
    ax8.set_position([pos.x0 + pos.width, pos.y0, pos2.width, pos.height])

    # save result
    dll = [dnperc1, dnperc2, lonnum1, lonnum2, latnum1, latnum2]
    dllstr = ['STmax_dnnnum', 'STmin_dnnnum', 'STmax_lonnum', 'STmin_lonnum', 'STmax_latnum', 'STmin_latnum']
    for i in range(6):
        if i < 2:
            df = pd.DataFrame(dll[i], index=['night(%)', 'day(%)'], columns=[dllstr[i]])
        elif (i >= 2) & (i < 4):
            df = pd.DataFrame(dll[i], columns=['lon', 'lonnum'])
        else:
            df = pd.DataFrame(dll[i], columns=['lat', 'latnum'])
        outpath = lspath + 'multiyear_' + dllstr[i]
        outname = ffile + '_multiyear_' + dllstr[i] + '.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')

        # save picture
    outname = ffile + '_multiyear_distribution_location.png'
    filepath = combinename(figpath, outname)
    plt.savefig(filepath, dpi=300)
    print(filepath)
    plt.close()









































