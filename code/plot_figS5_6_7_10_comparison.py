# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# this file is for plotting the comparion between different periodes
# including fig S5-7, S10

# def a function to combine outpath and outname
def combinename(outpath, outname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)
    return filepath


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
def plotbar(axname, absdata, reldata, clrname, ylabelname1, ylabelname2, sigp, pnum, isfill, titname, legloc):
    x = np.arange(len(absdata))
    width = 0.35
    if isfill == 0:
        rects1 = axname.bar(x - width / 2, absdata, width, linewidth=1.5, color='none', edgecolor=clrname[0], alpha=0.5)
        axname1 = axname.twinx()
        rects2 = axname1.bar(x + width / 2, reldata, width, linewidth=1.5, color='none', edgecolor=clrname[1],
                             alpha=0.5)
    else:
        rects1 = axname.bar(x - width / 2, absdata, width, linewidth=1.5, color=clrname[0], edgecolor=clrname[0],
                            alpha=0.5)
        axname1 = axname.twinx()
        rects2 = axname1.bar(x + width / 2, reldata, width, linewidth=1.5, color=clrname[1], edgecolor=clrname[1],
                             alpha=0.5)
    axname.axhline(y=0, xmin=np.min(x) - 1, xmax=np.max(x) + 1, color='k', linestyle='-', linewidth=1)

    # xlabel
    axname.set_xlim(np.min(x) - 1, np.max(x) + 1)
    axname.set_xticks(x)
    axname.set_xticklabels([])

    # ylabel
    axname.tick_params(axis='y', colors=clrname[0])
    axname1.tick_params(axis='y', colors=clrname[1])

    axname.set_ylabel(ylabelname1)
    axname.yaxis.label.set_color(clrname[0])
    axname1.set_ylabel(ylabelname2)
    axname1.yaxis.label.set_color(clrname[1])
    axname.spines['left'].set_color(clrname[0])
    axname1.spines['right'].set_color(clrname[1])
    # set y lim
    ax1_ylim = axname.get_ylim()
    ax2_ylim = axname1.get_ylim()
    ratio1 = np.floor(ax2_ylim[1] / ax1_ylim[1])
    if ratio1 == 0:
        ratio1 = 1
    if ax1_ylim[0] == 0:
        ratio2 = ratio1
    else:
        ratio2 = np.floor(ax2_ylim[0] / ax1_ylim[0])
    if ratio2 == 0:
        ratio2 = 1
    ratio = min(ratio1, ratio2)
    axname.set_ylim(min(ax1_ylim[0], ax2_ylim[0]) / ratio, max(ax1_ylim[1], ax2_ylim[1]) / ratio)
    axname1.set_ylim(min(ax1_ylim[0], ax2_ylim[0]), max(ax1_ylim[1], ax2_ylim[1]))
    # set legend
    axname.legend([rects1, rects2], ['Absolute change', 'Relative change'], ncol=1, loc=legloc, frameon=False)

    rectnum = 0
    yheight = axname.get_ylim()[1] - axname.get_ylim()[0]
    for rect in rects1:
        if sigp[rectnum] < 0.01:
            if absdata[rectnum] >= 0:
                axname.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005 * yheight, '**', ha='center')
            else:
                axname.text(rect.get_x() + rect.get_width() / 2, rect.get_height() - 0.05 * yheight, '**', ha='center')
        elif sigp[rectnum] >= 0.01 and sigp[rectnum] < 0.05:
            if absdata[rectnum] >= 0:
                axname.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.005 * yheight, '*', ha='center')
            else:
                axname.text(rect.get_x() + rect.get_width() / 2, rect.get_height() - 0.05 * yheight, '*', ha='center')
        rectnum = rectnum + 1

    # set pnum
    xlim = axname.get_xlim()
    ylim = axname.get_ylim()
    xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
    ypos = ylim[1] - 0.01 * (ylim[1] - ylim[0])
    axname.text(xpos, ypos, pnum, verticalalignment='top', horizontalalignment='left', fontweight='bold', fontsize=12)

    # set title
    axname.set_title(titname, fontweight='bold')

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

# seasonal day
firsec = 60
secthi = firsec + 92
thifou = secthi + 92
foufir = thifou + 91

for timestep in [10, 20, 30]:
    basepath = r'/data1/fyliu/a_temperature_range/process_data/comparison_' + str(timestep)
    outpath = r'/data1/fyliu/a_temperature_range/result/fig_main/comparison_' + str(timestep)
    os.makedirs(outpath, exist_ok=True)

    # ==========================================================================
    # figure1, dstr
    path = os.path.join(basepath, 'compare_p_dstr')
    # 0-3 reperensts abs change, rel change and P
    datanum = 3
    labels, datals = readcsv(path, datanum, 0)
    absls = datals[0]
    rells = datals[1]
    pls = datals[2]

    fig1 = plt.figure(figsize=(8, 9), dpi=300)
    grid = GridSpec(3, 1, figure=fig1, left=0.09, bottom=0.12, right=0.91, top=0.99, wspace=0, hspace=0)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[2, 0])
    axname = [ax1, ax2, ax3]
    clrname = [['dimgray', 'maroon'], ['dimgray', 'tan'], ['dimgray', 'steelblue']]
    pnum = ['(a) maxDSTR', '(b) meanDSTR', '(c) minDSTR']

    for i in range(3):
        ylabel0 = 'Absolute change (' + '\u2103' + ')'
        ylabel1 = 'Relative change (%)'
        plotbar(axname[i], absls[:, i], rells[:, i], clrname[i], ylabel0, ylabel1, pls[:, i], pnum[i], 1, '',
                'upper right')

    ax3.set_xticklabels(labels, rotation=-75)

    # save figrue
    outname = 'compare_dstr.png'
    filepath = combinename(outpath, outname)
    plt.savefig(filepath, dpi=300)
    print(filepath)
    plt.close()

    # ==========================================================================
    # figure2, hourdis
    path = os.path.join(basepath, 'compare_p_hourdis')
    # 0-3 reperensts abs change, rel change and P
    datanum = 3
    labels, datals = readcsv(path, datanum, 0)
    absls = datals[0].T[0]
    rells = datals[1].T[0]
    pls = datals[2].T[0]

    fig2 = plt.figure(figsize=(8, 3), dpi=300)
    ax1 = fig2.add_subplot(111)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.99, bottom=0.34)
    clrname = ['dimgray', 'steelblue']
    ylabelname = 'Hour distribution'
    ylabel0 = 'Absolute change (h)'
    ylabel1 = 'Relative change (%)'

    x = np.arange(len(absls))
    width = 0.35
    rects1 = ax1.bar(x - width / 2, absls, width, linewidth=1.5, facecolor=clrname[0], edgecolor=clrname[0], alpha=0.5)
    plotbar(ax1, absls, rells, clrname, ylabel0, ylabel1, pls, '', 1, '', 'upper right')

    ax1.set_xticklabels(labels, rotation=-75)

    # save figrue
    outname = 'compare_hourdis.png'
    filepath = combinename(outpath, outname)
    plt.savefig(filepath, dpi=300)
    print(filepath)
    plt.close()

    # ==========================================================================
    # figure3, locdis
    path = os.path.join(basepath, 'compare_p_locdis')
    # 0-3 reperensts abs change, rel change and P
    datanum = 3
    labels, datals = readcsv(path, datanum, 0)
    absls = datals[0]
    rells = datals[1]
    pls = datals[2]

    fig3 = plt.figure(figsize=(8, 12), dpi=300)
    grid = GridSpec(4, 1, figure=fig3, left=0.08, bottom=0.09, right=0.92, top=0.97, wspace=0, hspace=0.15)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[2, 0])
    ax4 = plt.subplot(grid[3, 0])
    axname = [ax1, ax2, ax3, ax4]
    clrname = [['dimgray', 'maroon'], ['dimgray', 'maroon'], ['dimgray', 'steelblue'], ['dimgray', 'steelblue']]
    ylabelname = ['STmax without sunshine', 'STmax with sunshine', 'STmin without sunshine', 'STmin with sunshine']
    titname = ['(a) STmax lon', '(b) STmax lat', '(c) STmin lon', '(d) STmin lat']
    isfill = np.array([1, 0, 1, 0])

    for i in range(4):
        # ylabel0 = ylabelname[i] + ' AC (%)'
        # ylabel1 = ylabelname[i] + ' RC (%)'
        ylabel0 = 'Absolute change (%)'
        ylabel1 = 'Relative change (%)'
        plotbar(axname[i], absls[:, i], rells[:, i], clrname[i], ylabel0, ylabel1, pls[:, i], '', isfill[i], titname[i],
                'upper right')

    ax4.set_xticklabels(labels, rotation=-75)

    # save figrue
    outname = 'compare_locdis.png'
    filepath = combinename(outpath, outname)
    plt.savefig(filepath, dpi=300)
    print(filepath)
    plt.close()

    # ==========================================================================
    # figure4, locdis, seasonal dynamics
    path = os.path.join(basepath, 'compare_mmmvalue_dstr')
    # 0-2 reperensts f decade and l decade
    datanum = 2
    labels, datals = readcsv(path, datanum, 0)
    fdata = datals[0][:, [3, 7, 11]]
    ldata = datals[1][:, [3, 7, 11]]
    absls = ldata - fdata
    rells = (ldata - fdata) / abs(fdata) * 100
    pls = np.ones(37)

    fig4 = plt.figure(figsize=(8, 9), dpi=300)
    grid = GridSpec(3, 1, figure=fig4, left=0.09, bottom=0.12, right=0.91, top=0.99, wspace=0, hspace=0)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[2, 0])
    axname = [ax1, ax2, ax3]
    clrname = [['dimgray', 'maroon'], ['dimgray', 'tan'], ['dimgray', 'steelblue']]
    pnum = ['(a) std maxDSTR', '(b) std meanDSTR', '(c) std minDSTR']

    for i in range(3):
        ylabel0 = 'Absolute change (' + '\u2103' + ')'
        ylabel1 = 'Relative change (%)'
        plotbar(axname[i], absls[:, i], rells[:, i], clrname[i], ylabel0, ylabel1, pls, pnum[i], 1, '', 'upper right')

    ax3.set_xticklabels(labels, rotation=-75)

    # save figrue
    outname = 'compare_dstr_std.png'
    filepath = combinename(outpath, outname)
    plt.savefig(filepath, dpi=300)
    print(filepath)
    plt.close()

    # ==========================================================================
    # figure5, day or night
    path = os.path.join(basepath, 'compare_dnperc')
    # 0-4 reperensts fstmax, lstmax, ftmin, lstmin
    datanum = 4
    labels, datals = readcsv(path, datanum, 0)
    fstmax = datals[0]
    lstmax = datals[1]
    fstmin = datals[2]
    lstmin = datals[3]
    abs1 = lstmax - fstmax
    rel1 = (lstmax - fstmax) / abs(fstmax) * 100
    abs2 = lstmin - fstmin
    rel2 = (lstmin - fstmin) / abs(fstmin) * 100
    absls = np.hstack((abs1, abs2))
    rells = np.hstack((rel1, rel2))
    pls = np.ones(37)

    fig5 = plt.figure(figsize=(8, 12), dpi=300)
    grid = GridSpec(4, 1, figure=fig5, left=0.08, bottom=0.09, right=0.92, top=0.97, wspace=0, hspace=0.15)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[2, 0])
    ax4 = plt.subplot(grid[3, 0])
    axname = [ax1, ax2, ax3, ax4]
    clrname = [['dimgray', 'maroon'], ['dimgray', 'maroon'], ['dimgray', 'steelblue'], ['dimgray', 'steelblue']]
    ylabelname = ['STmax without sunshine', 'STmax with sunshine', 'STmin without sunshine', 'STmin with sunshine']
    titname = ['(a) STmax without sunshine', '(b) STmax with sunshine', '(c) STmin without sunshine',
               '(d) STmin with sunshine']
    isfill = np.array([1, 0, 1, 0])

    for i in range(4):
        # ylabel0 = ylabelname[i] + ' AC (%)'
        # ylabel1 = ylabelname[i] + ' RC (%)'
        ylabel0 = 'Absolute change (%)'
        ylabel1 = 'Relative change (%)'
        plotbar(axname[i], absls[:, i], rells[:, i], clrname[i], ylabel0, ylabel1, pls, '', isfill[i], titname[i],
                'upper left')

    ax4.set_xticklabels(labels, rotation=-75)

    # save figrue
    outname = 'compare_dnperc.png'
    filepath = combinename(outpath, outname)
    plt.savefig(filepath, dpi=300)
    print(filepath)
    plt.close()


