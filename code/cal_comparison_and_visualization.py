# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import stats


# this file is for comparison between different periods at each scale, as well as visualization based on ERA5-Land
# including 3 timescale, 1950-1959 V.S. 2014-2023, 1950-1969 V.S. 2004-2023, 1950-1979 V.S. 1994-2023
# 1) the comparison of time series
# 2) the comparison of hour distribution
# 3) the comparison of STmax/STmin distribution


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
    axname1.plot(x, mmmseries[:, 0], linestyle='-', marker=mk, markersize=mksz, color=clr[1], linewidth=1,
                 alpha=0.5, label=labelname)
    axname2.plot(x, mmmseries[:, 1], linestyle='-', marker=mk, markersize=mksz, color=clr[0], linewidth=1,
                 alpha=0.5, label=labelname)
    axname3.plot(x, mmmseries[:, 2], linestyle='-', marker=mk, markersize=mksz, color=clr[2], linewidth=1,
                 alpha=0.5, label=labelname)

    # add seasonal line
    axname = [axname1, axname2, axname3]
    for i in range(3):
        for xvalue in [firsec, secthi, thifou, foufir]:
            axname[i].axvline(x=xvalue, color=clr[0], linestyle='--', linewidth=1, alpha=0.5)
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


# def a function plot hourdistribution
# return, modehour, pandas series, index, doy, columns, multiyear mode hour distribution
# hournum, 1*24array, each columns, 0-23 hour
def plothourdis(axname1, axname2, hourdata, clr, labelname, forl):
    # cal multiyear average hour dis
    hourdf = pd.DataFrame(hourdata, columns=['year', 'doy', 'hour'])
    # cal mode hour
    modehour = hourdf.groupby('doy')['hour'].apply(group_mode)
    modehour.plot(kind='line', x='doy', y='hour', ax=axname1, color=clr[2], marker='.', markersize=3, linestyle='-',
                  linewidth=1, alpha=0.5, label='Mode hour ' + '(' + labelname + ')')

    # add season line
    for xvalue in [firsec, secthi, thifou, foufir]:
        axname1.axvline(x=xvalue, color=clr2[0], linestyle='--', linewidth=0.5, alpha=0.3)
        # add reference line
    for yvalue in range(0, 19, 6):
        axname1.axhline(y=yvalue, color=clr2[0], linestyle='--', linewidth=0.5, alpha=0.3)

    # set xy axis & title
    axname1.set_xlim(0, 367)
    axname1.set_xticks([firsec, secthi, thifou, foufir])
    axname1.set_xticklabels(['Mar.', 'Jun.', 'Sep.', 'Dec.'])
    axname1.set_xlabel('Day of year')
    axname1.set_ylim(-1, 25)
    axname1.set_yticks(range(25))
    axname1.set_yticklabels(
        ['0', '', '', '', '', '', '6', '', '', '', '', '', '12', '', '', '', '', '', '18', '', '', '', '', '', '24'])
    axname1.set_ylabel('Hour of day')

    # cal the times of each hour
    x = np.arange(24)
    hournum = np.zeros(24)
    for hour in range(24):
        hournum[hour] = np.sum(hourdata[:, 2] == hour)
    # plot stack bin
    width = 0.4
    if forl == 1:
        axname2.barh(x - width / 2, hournum, width, color=clr[2], alpha=0.5, label=labelname)
        for hour in range(24):
            text = str(int(hournum[hour]))
            axname2.text(hournum[hour], x[hour] - width / 2, text, ha='left', va='center', c=clr[0])
    else:
        axname2.barh(x + width / 2, hournum, width, color=clr[2], alpha=0.5, label=labelname)
        for hour in range(24):
            text = str(int(hournum[hour]))
            axname2.text(hournum[hour], x[hour] + width / 2, text, ha='left', va='center', c=clr[0])
    maxnum = np.max(hournum)
    axname2.set_xlim(0, maxnum + 300)
    axname2.set_ylim(-1, 25)
    axname2.set_yticks(range(25))
    axname2.set_yticklabels([])
    axname2.set_xlabel('Frequency')

    return modehour, hournum


# def a function calculate mode
def group_mode(group):
    return group.mode()


# def a function plot location and day/night
# return, dnperc, 2*1array, night£¬ day percentile, %
# lonnum, latnum, n*2array, lon/lat value, count
def plotloc(axname1, axname2, axname3, locdata, pietext, locclr, barnum):
    # cal the percent of day or night
    dnperc = np.zeros((2, 1))
    for i in range(2):
        dnperc[i, 0] = np.sum(locdata[:, 3] == i)
    # tranfter to percent
    dnsum = np.sum(dnperc)
    for i in range(2):
        dnperc[i, 0] = dnperc[i, 0] / dnsum * 100

    # plot day or night
    axname3.pie(x=dnperc.flatten(), labels=['Without sunshine', 'With sunshine'], colors=['dimgray', 'white'],
                autopct='%.1f%%', wedgeprops={'linewidth': 1, 'edgecolor': 'k'})

    # cal the number of lon and lat
    unique_values, counts = np.unique(locdata[:, 1], return_counts=True)
    lonnum = np.column_stack((unique_values, counts))
    unique_values, counts = np.unique(locdata[:, 2], return_counts=True)
    latnum = np.column_stack((unique_values, counts))

    # plot lonnum and latnum
    width = 0.05
    if barnum == 0:
        axname1.bar(lonnum[:, 0] - width / 2, lonnum[:, 1], width, linewidth=0.5, facecolor=locclr, edgecolor=locclr,
                    alpha=0.5, label=fstlab)
        axname2.bar(latnum[:, 0] - width / 2, latnum[:, 1], width, linewidth=0.5, facecolor=locclr, edgecolor=locclr,
                    alpha=0.5, label=fstlab)
    else:
        axname1.bar(lonnum[:, 0] + width / 2, lonnum[:, 1], width, linewidth=0.5, facecolor=locclr, edgecolor=locclr,
                    alpha=0.5, label=seclab)
        axname2.bar(latnum[:, 0] + width / 2, latnum[:, 1], width, linewidth=0.5, facecolor=locclr, edgecolor=locclr,
                    alpha=0.5, label=seclab)

    axname1.set_ylabel('Frequency')
    axname2.set_ylabel('Frequency')
    axname3.set_title(pietext, fontweight='bold')
    xlab1 = pietext[4:9] + ' Lon'
    xlab2 = pietext[4:9] + ' Lat'
    axname1.set_xlabel(xlab1)
    axname2.set_xlabel(xlab2)

    return dnperc, lonnum, latnum


# call function to plot
# set basic info
fstyear_ls = [1959, 1969, 1979]
secyear_ls = [2014, 2004, 1994]
timestep_ls = [10, 20, 30]
for idx in range(3):
    fstyear = fstyear_ls[idx]
    secyear = secyear_ls[idx]
    timestep = timestep_ls[idx]
    fstlab = '1950-' + str(fstyear)
    seclab = str(secyear) + '-2023'

    # path
    figpath = f'/data1/fyliu/a_temperature_range/result/fig_comparison_{timestep}/'
    lspath = f'/data1/fyliu/a_temperature_range/process_data/comparison_{timestep}/'
    os.makedirs(figpath, exist_ok=True)
    os.makedirs(lspath, exist_ok=True)

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
    datapath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_dstr/'
    datapath1 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_stmax/'
    datapath2 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_stmin/'
    fall = os.listdir(datapath)
    fall.sort()
    fall1 = os.listdir(datapath1)
    fall1.sort()
    fall2 = os.listdir(datapath2)
    fall2.sort()
    filelen = len(fall)

    for i in range(filelen):
        # use for fig content
        ffile = fall[i]
        ffile1 = fall1[i]
        ffile2 = fall2[i]
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
        data = data.values
        ind1 = (data[:, 0] >= 1950) & (data[:, 0] <= fstyear)
        ind2 = (data[:, 0] >= secyear) & (data[:, 0] <= 2023)
        data1 = data[ind1, :]
        data2 = data[ind2, :]
        # stmax
        path = os.path.join(datapath1, ffile1)
        data = pd.read_csv(path, sep=',', header=0, index_col=0)
        data = data.values
        ind1 = (data[:, 0] >= 1950) & (data[:, 0] <= fstyear)
        ind2 = (data[:, 0] >= secyear) & (data[:, 0] <= 2023)
        data3 = data[ind1, :]
        data4 = data[ind2, :]
        # stmin
        path = os.path.join(datapath2, ffile2)
        data = pd.read_csv(path, sep=',', header=0, index_col=0)
        data = data.values
        ind1 = (data[:, 0] >= 1950) & (data[:, 0] <= fstyear)
        ind2 = (data[:, 0] >= secyear) & (data[:, 0] <= 2023)
        data5 = data[ind1, :]
        data6 = data[ind2, :]

        # figure1
        # plot compare DSTR
        fig1 = plt.figure(figsize=(8, 16), dpi=300)
        grid = GridSpec(9, 1, figure=fig1, left=0.08, bottom=0.02, right=0.98, top=0.97,
                        wspace=0, hspace=0.27)
        ax1 = plt.subplot(grid[0, 0])
        ax2 = plt.subplot(grid[1, 0])
        ax3 = plt.subplot(grid[2, 0])
        ax4 = plt.subplot(grid[3, 0])
        ax5 = plt.subplot(grid[4, 0])
        ax6 = plt.subplot(grid[5, 0])
        ax7 = plt.subplot(grid[6, 0])
        ax8 = plt.subplot(grid[7, 0])
        ax9 = plt.subplot(grid[8, 0])
        ylabelname = [['maxDSTR', 'meanDSTR', 'minDSTR'], ['maxSTmax', 'meanSTmax', 'minSTmax'],
                      ['maxSTmin', 'meanSTmin', 'minSTmin']]
        ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        pnum = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)'], ['(g)', '(h)', '(i)']]
        data = [data1, data2, data3, data4, data5, data6]
        dataname = ['dstr', 'stmax', 'stmin']
        for i in range(3):
            mmmseries1, mmmvalue1 = plotmulitline(ax[3 * i], ax[3 * i + 1], ax[3 * i + 2], data[2 * i], clr1, '.', fstlab,
                                                  ylabelname[i], pnum[i])
            mmmseries2, mmmvalue2 = plotmulitline(ax[3 * i], ax[3 * i + 1], ax[3 * i + 2], data[2 * i + 1], clr2, '.',
                                                  seclab, ylabelname[i], '')

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
            ax[3 * i].text(1, ax[3 * i].get_ylim()[1] + 0.04 * (ax[3 * i].get_ylim()[1] - ax[3 * i].get_ylim()[0]),
                           reltext[0])
            ax[3 * i].text(1, ax[3 * i].get_ylim()[1] + 0.14 * (ax[3 * i].get_ylim()[1] - ax[3 * i].get_ylim()[0]),
                           abstext[0])
            ax[3 * i + 1].text(1, ax[3 * i + 1].get_ylim()[1] + 0.04 * (
                        ax[3 * i + 1].get_ylim()[1] - ax[3 * i + 1].get_ylim()[0]), reltext[1])
            ax[3 * i + 1].text(1, ax[3 * i + 1].get_ylim()[1] + 0.14 * (
                        ax[3 * i + 1].get_ylim()[1] - ax[3 * i + 1].get_ylim()[0]), abstext[1])
            ax[3 * i + 2].text(1, ax[3 * i + 2].get_ylim()[1] + 0.04 * (
                        ax[3 * i + 2].get_ylim()[1] - ax[3 * i + 2].get_ylim()[0]), reltext[2])
            ax[3 * i + 2].text(1, ax[3 * i + 2].get_ylim()[1] + 0.14 * (
                        ax[3 * i + 2].get_ylim()[1] - ax[3 * i + 2].get_ylim()[0]), abstext[2])

            # legend
            ax[3 * i].legend(ncol=2, bbox_to_anchor=(1.02, 0.95), loc='lower right', frameon=False)
            ax[3 * i + 1].legend(ncol=2, bbox_to_anchor=(1.02, 0.95), loc='lower right', frameon=False)
            ax[3 * i + 2].legend(ncol=2, bbox_to_anchor=(1.02, 0.95), loc='lower right', frameon=False)

            # save mmm and change result
            # mmm series
            df = pd.DataFrame(np.hstack((mmmseries1, mmmseries2)), index=np.arange(1, 367),
                              columns=['f max', 'f mean', 'f min', 'l max', 'l mean', 'l min'])
            outpath = lspath + 'compare_mmmseries_' + dataname[i]
            outname = ff + '_compare_mmmseries_' + dataname[i] + '.csv'
            filepath = combinename(outpath, outname)
            df.to_csv(filepath, sep=',')
            # mmm values
            df = pd.DataFrame(np.vstack((mmmvalue1, mmmvalue2)).T,
                              index=['maxmax', 'meanmax', 'minmax', 'stdmax', 'maxmean',
                                     'meanmean', 'minmean', 'stdmean', 'maxmin', 'meanmin', 'minmin', 'stdmin'],
                              columns=['fdecade', 'ldecade'])
            outpath = lspath + 'compare_mmmvalue_' + dataname[i]
            outname = ff + '_compare_mmmvalue_' + dataname[i] + '.csv'
            filepath = combinename(outpath, outname)
            df.to_csv(filepath, sep=',')
            # abs, rel and p
            df = pd.DataFrame(changels, index=['max', 'mean', 'min'],
                              columns=['abschange', 'relchange', 'P'])
            outpath = lspath + 'compare_p_' + dataname[i]
            outname = ff + '_compare_p_' + dataname[i] + '.csv'
            filepath = combinename(outpath, outname)
            df.to_csv(filepath, sep=',')

            # add title and so on
        ax9.set_xticklabels(['Mar.', 'Jun.', 'Sep.', 'Dec.'])

        # save fig1
        outname = ff + '_compare_mmmchange.png'
        filepath = combinename(figpath, outname)
        plt.savefig(filepath, dpi=300)
        print(filepath)
        plt.close()

    # =============================================================================================
    # =============================================================================================
    # read hourdis data, iterate by region
    datapath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_hourdis/'
    fall = os.listdir(datapath)
    fall.sort()
    for ffile in fall:
        # use for fig content
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

        # hourdis
        path = os.path.join(datapath, ffile)
        data = pd.read_csv(path, sep=',', header=0, index_col=0)
        data = data.values
        ind1 = (data[:, 0] >= 1950) & (data[:, 0] <= fstyear)
        ind2 = (data[:, 0] >= secyear) & (data[:, 0] <= 2023)
        data1 = data[ind1, :]
        data2 = data[ind2, :]

        # figure2
        # plot compare hour distribution
        fig3 = plt.figure(figsize=(8, 6), dpi=300)
        grid = GridSpec(1, 10, figure=fig3, left=0.07, bottom=0.08, right=0.98, top=0.98,
                        wspace=0, hspace=0.3)
        ax4 = plt.subplot(grid[0, 0:7])
        ax5 = plt.subplot(grid[0, 7:10])

        # plot hour distribution
        modehour1, hournum1 = plothourdis(ax4, ax5, data1, clr1, fstlab, 1)
        modehour2, hournum2 = plothourdis(ax4, ax5, data2, clr2, seclab, 0)
        ax4.legend(ncol=2, loc='upper left', frameon=False)
        ax5.legend(ncol=1, loc='upper right', frameon=False)

        # change test
        clen = 1
        abstext = list(np.arange(clen))
        reltext = list(np.arange(clen))
        # each column, absdiffer, relative differ and p value, each row, hournum
        changels = np.zeros((clen, 3))
        for i in range(clen):
            absdif, reldif, p = changetest(hournum1, hournum2)
            changels[i, 0] = absdif
            changels[i, 1] = reldif
            changels[i, 2] = p
            abstext[i] = 'Absolute change: ' + str(round(absdif, 2)) + 'h'
            reltext[i] = 'Relative change: ' + str(round(reldif, 2)) + '%' + ', P = ' + str(round(p, 2))

        # text
        ax4.text(1, ax4.get_ylim()[1] - 0.98 * (ax4.get_ylim()[1] - ax4.get_ylim()[0]), reltext[0])
        ax4.text(1, ax4.get_ylim()[1] - 0.93 * (ax4.get_ylim()[1] - ax4.get_ylim()[0]), abstext[0])

        # save results
        # modehour
        modehour = pd.concat([modehour1, modehour2], axis=1)
        outpath = lspath + 'compare_modehour'
        outname = ff + '_compare_modehour.csv'
        filepath = combinename(outpath, outname)
        modehour.to_csv(filepath, header=['fhour', 'lhour'], sep=',')
        # hourmun
        df = pd.DataFrame(np.vstack((hournum1, hournum2)).T, index=range(24), columns=['fhournum', 'lhournum'])
        outpath = lspath + 'compare_hournum_un'
        outname = ff + '_compare_huornum_un.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')
        # abs, rel and p
        df = pd.DataFrame(changels, index=['hourdis'],
                          columns=['abschange', 'relchange', 'P'])
        outpath = lspath + 'compare_p_hourdis'
        outname = ff + '_compare_p_hourdis.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')
        # figure
        outname = ff + '_compare_distribution_hour.png'
        filepath = combinename(figpath, outname)
        plt.savefig(filepath, dpi=300)
        print(filepath)
        plt.close()

    # =============================================================================================
    # =============================================================================================
    # read maxdis and mindis data, iterate by region
    datapath = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_maxdis/'
    datapath1 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_mindis/'
    fall = os.listdir(datapath)
    fall.sort()
    fall1 = os.listdir(datapath1)
    fall1.sort()
    filelen = len(fall)
    for i in range(filelen):
        # use for fig content
        ffile = fall[i]
        ffile1 = fall1[i]
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

        # maxdis
        path = os.path.join(datapath, ffile)
        data = pd.read_csv(path, sep=',', header=0, index_col=0)
        data = data.values
        ind1 = (data[:, 6] >= 1950) & (data[:, 6] <= fstyear)
        ind2 = (data[:, 6] >= secyear) & (data[:, 6] <= 2023)
        data1 = data[ind1, :]
        data2 = data[ind2, :]
        # mindis
        path1 = os.path.join(datapath1, ffile1)
        data = pd.read_csv(path1, sep=',', header=0, index_col=0)
        data = data.values
        ind1 = (data[:, 6] >= 1950) & (data[:, 6] <= fstyear)
        ind2 = (data[:, 6] >= secyear) & (data[:, 6] <= 2023)
        data3 = data[ind1, :]
        data4 = data[ind2, :]

        # figure4
        fig4 = plt.figure(figsize=(8, 8), dpi=300)
        grid = GridSpec(4, 10, figure=fig4, left=0.09, bottom=0.06, right=0.95, top=0.95,
                        wspace=0.6, hspace=0.7)
        ax1 = plt.subplot(grid[0, 0:7])
        ax2 = plt.subplot(grid[1, 0:7])
        ax3 = plt.subplot(grid[2, 0:7])
        ax4 = plt.subplot(grid[3, 0:7])
        ax5 = plt.subplot(grid[0, 7:10])
        ax6 = plt.subplot(grid[1, 7:10])
        ax7 = plt.subplot(grid[2, 7:10])
        ax8 = plt.subplot(grid[3, 7:10])

        clr1 = ['silver', 'lightcoral', 'lightblue']
        clr2 = ['dimgray', 'brown', 'steelblue']
        dnperc1, lonnum1, latnum1 = plotloc(ax1, ax2, ax5, data1, '(e) STmax (' + fstlab + ')', clr1[1], 0)
        dnperc2, lonnum2, latnum2 = plotloc(ax1, ax2, ax6, data2, '(f) STmax (' + seclab + ')', clr2[1], 1)
        dnperc3, lonnum3, latnum3 = plotloc(ax3, ax4, ax7, data3, '(g) STmin (' + fstlab + ')', clr1[2], 0)
        dnperc4, lonnum4, latnum4 = plotloc(ax3, ax4, ax8, data4, '(h) STmin (' + seclab + ')', clr2[2], 1)

        # change test
        clen = 4
        abstext = list(np.arange(clen))
        reltext = list(np.arange(clen))
        # each column, absdiffer, relative differ and p value, each row, hournum
        changedata1 = [lonnum1[:, 1], latnum1[:, 1], lonnum3[:, 1], latnum3[:, 1]]
        changedata2 = [lonnum2[:, 1], latnum2[:, 1], lonnum4[:, 1], latnum4[:, 1]]
        changels = np.zeros((clen, 3))
        for i in range(clen):
            absdif, reldif, p = changetest(changedata1[i], changedata2[i])
            changels[i, 0] = absdif
            changels[i, 1] = reldif
            changels[i, 2] = p
            abstext[i] = 'Absolute change: ' + str(round(absdif, 2)) + '\xb0'
            reltext[i] = 'Relative change: ' + str(round(reldif, 2)) + '%' + ', P = ' + str(round(p, 2))

        # adjust xlable and text
        axname = [ax1, ax2, ax3, ax4]
        for i in range(4):
            ticks_x = axname[i].get_xticks()
            if i % 2 == 0:
                labels_x = [f'{abs(round(tick, 1))}\xb0{"E" if tick > 0 else "W" if tick < 0 else " "}' for tick in ticks_x]
            else:
                labels_x = [f'{abs(round(tick, 1))}\xb0{"N" if tick > 0 else "S" if tick < 0 else " "}' for tick in ticks_x]
            axname[i].set_xticks(ticks_x)
            axname[i].set_xticklabels(labels_x)
            axname[i].text(axname[i].get_xlim()[0],
                           axname[i].get_ylim()[1] + 0.04 * (axname[i].get_ylim()[1] - axname[i].get_ylim()[0]), reltext[i])
            axname[i].text(axname[i].get_xlim()[0],
                           axname[i].get_ylim()[1] + 0.2 * (axname[i].get_ylim()[1] - axname[i].get_ylim()[0]), abstext[i])
            axname[i].legend(ncol=1, bbox_to_anchor=(1.02, 0.9), loc='lower right', frameon=False)

        # add pnum
        # adjust xlable and text
        pnum = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        axname = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
        for i in range(4):
            xlim = axname[i].get_xlim()
            ylim = axname[i].get_ylim()
            xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
            ypos = ylim[1] - 0.01 * (ylim[1] - ylim[0])
            axname[i].text(xpos, ypos, pnum[i], verticalalignment='top', horizontalalignment='left', fontweight='bold',
                           fontsize=12)

        # save results
        # dnperc
        df = pd.DataFrame(np.hstack((dnperc1, dnperc2, dnperc3, dnperc4)), index=['night', 'day'],
                          columns=['fSTmax', 'lSTmax', 'fSTmin', 'lSTmin'])
        outpath = lspath + 'compare_dnperc'
        outname = ff + '_compare_dnperc.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')
        # lonnum
        df = pd.concat([pd.DataFrame(lonnum1), pd.DataFrame(lonnum2), pd.DataFrame(lonnum3), pd.DataFrame(lonnum4)], axis=1)
        df.columns = ['fSTmax lon', 'count', 'lSTmax lon', 'count', 'fSTmin lon', 'count', 'lSTmin lon', 'count']
        outpath = lspath + 'compare_lonnum'
        outname = ff + '_compare_lonnum.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')
        # latnum
        df = pd.concat([pd.DataFrame(latnum1), pd.DataFrame(latnum2), pd.DataFrame(latnum3), pd.DataFrame(latnum4)], axis=1)
        df.columns = ['fSTmax lat', 'count', 'lSTmax lat', 'count', 'fSTmin lat', 'count', 'lSTmin lat', 'count']
        outpath = lspath + 'compare_latnum'
        outname = ff + '_compare_latnum.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')
        # abs, rel and p
        df = pd.DataFrame(changels, index=['STmax lon', 'STmax lat', 'STmin lon', 'STmin lat'],
                          columns=['abschange', 'relchange', 'P'])
        outpath = lspath + 'compare_p_locdis'
        outname = ff + '_compare_p_locdis.csv'
        filepath = combinename(outpath, outname)
        df.to_csv(filepath, sep=',')
        # figure
        outname = ff + '_compare_distribution_location.png'
        filepath = combinename(figpath, outname)
        plt.savefig(filepath, dpi=300)
        print(filepath)
        plt.close()









































