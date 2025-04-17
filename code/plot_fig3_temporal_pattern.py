# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# this file is for plotting temporal pattern of DSTR
# including hour distribution, daytime or nighttime distribution, and month distribution


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


def add_right_cax(ax, pad, width):
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


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

pathdir = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/'
outpath = r'/data1/fyliu/a_temperature_range/result/fig_main/'
os.makedirs(outpath, exist_ok=True)
clrname = ['maroon', 'dimgray', 'steelblue']

xrot = -70

# ========================================================================
# ========================================================================
# plot day and night percentage, hour and month distribution
grid_col = 11
fig2 = plt.figure(figsize=(8, 11), dpi=300)
grid = GridSpec(11, 11, figure=fig2, left=0.08, bottom=0.065, right=0.94, top=0.99, wspace=0, hspace=0.3)
ax2 = plt.subplot(grid[0:6, 0:grid_col - 1])
ax1 = plt.subplot(grid[6:8, 0:grid_col])
ax3 = plt.subplot(grid[8:11, 0:grid_col - 1])

# plot day or night percentage
path = os.path.join(pathdir, 'multiyear_STmax_dnnnum')
datanum = 1
labels, datals = readcsv(path, datanum, 0)
stmax_dn = datals[0]

path = os.path.join(pathdir, 'multiyear_STmin_dnnnum')
datanum = 1
labels, datals = readcsv(path, datanum, 0)
stmin_dn = datals[0]

x = np.arange(37)
clr1 = [i / 255 for i in [8, 58, 122]]
clr2 = [i / 255 for i in [150, 0, 0]]
ax1.plot(x, stmax_dn[:, 1], '-o', linewidth=1.5, color=clr2, label='STmax', alpha=0.5)
ax1.plot(x, stmin_dn[:, 1], '-o', linewidth=1.5, color=clr1, label='STmin', alpha=0.5)
ax1.axhline(y=50, xmin=0, xmax=1, linestyle='--', linewidth=1, color='dimgray', alpha=0.5)
ax1.text(37, 50, '50%', va='center', ha='left')
ax1.set_xlim(-0.5, 36.5)
ax1.set_xticks(np.arange(0, 37))
ax1.set_xticklabels(labels, rotation=xrot, ha='left')
ax1.set_ylabel('Within sunshine (%)')
# ax1.legend(ncol=2, loc='lower left', frameon=False)
ax1.legend(ncol=1, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
ypos = ylim[1] - 0.01 * (ylim[1] - ylim[0])
ax1.text(xpos, ypos, '(b)', fontsize=12, fontweight='bold', va='top', ha='left')
pos = ax1.get_position().bounds
ax1.set_position([pos[0], pos[1], pos[2] * 0.9, pos[3]])

# ========================================================================
# plot hour distribution
path = os.path.join(pathdir, 'multiyear_hournum_un')
# 0-1 represents hournum
datanum = 1
labels, datals = readcsv(path, datanum, 0)
datals = datals[0]

sumls = np.sum(datals, axis=1)
m, n = np.shape(datals)
for i in range(m):
    for j in range(n):
        datals[i, j] = datals[i, j] / sumls[i]

# plot matrix on ax2
pos1 = ax2.get_position().bounds
ax2.set_position([pos1[0], pos1[1] + 0.06, pos1[2], pos1[3]])
h1 = ax2.imshow(datals.T, cmap='Blues', interpolation='nearest')

# Add colorbar for ax2
cax = add_right_cax(ax2, pad=0.03, width=0.02)
c1 = fig2.colorbar(h1, cax=cax, orientation='vertical')
c1.set_ticks([0, 0.05, 0.1, 0.15, 0.2])
c1.set_ticklabels(['0%', '5%', '10%', '15%', '20%'])
c1.set_label('Hour distribution of DSTR')

# Add xyticks, labels, and other settings for ax2
# ax2.set_ylabel('Hour of day')
ax2.set_ylim(-0.5, 23.5)
ax2.set_yticks(np.arange(0, 24, 3))
ax2.set_yticklabels(['0h', '3h', '6h', '9h', '12h', '15h', '18h', '21h'])
ax2.set_xlim(-0.5, 36.5)
ax2.set_xticks(np.arange(0, 37))
ax2.set_xticklabels(labels, rotation=xrot, ha='left')

xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
ypos = ylim[1] - 0.01 * (ylim[1] - ylim[0])
ax2.text(xpos, ypos, '(a)', fontsize=12, fontweight='bold', va='top', ha='left')

pos = ax1.get_position().bounds
ax1.set_position([pos[0], pos[1] + 0.05, pos1[2], pos[3]])

# ========================================================================
# plot monthly distribution of DSTR
path = os.path.join(pathdir, 'multiyear_dstr')
datanum = 3
allfile = os.listdir(path)
allfile.sort()
result_list = []
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
    filepath = os.path.join(path, ff)
    # read data
    data = pd.read_csv(filepath, sep=',', header=0, index_col=0)
    data1 = data.iloc[:, 0:datanum].copy()
    data1['date'] = pd.to_datetime(data1['year'].astype(int).astype(str), format='%Y') + pd.to_timedelta(
        data1['doy'].astype(int) - 1, unit='D')
    data1['month'] = data1['date'].dt.month
    monthly_avg = data1.groupby('month')['dstr'].mean().reset_index(name='monthly_mean')
    dstr_anomaly = (monthly_avg['monthly_mean'] - monthly_avg['monthly_mean'].mean()) / monthly_avg[
        'monthly_mean'].std()
    dstr_anomaly = dstr_anomaly.to_frame(name=key)
    result_list.append(dstr_anomaly)

final_result = pd.concat(result_list, axis=1)
final_result.rename(columns={'Tibet': 'Xizang'}, inplace=True)

# clr1 = [i/255 for i in [162, 192, 217]]
# clr2 = [i/255 for i in [192, 130, 130]]
clr1 = 'steelblue'
clr2 = 'maroon'
darken_factor = 0.93

clr3 = [i / 255 * darken_factor for i in [250, 137, 87]]
clr4 = [i / 255 * darken_factor for i in [230, 81, 57]]
clr5 = [i / 255 * darken_factor for i in [253, 230, 195]]

colors = [(0, 'white'), (0.25, clr5), (0.5, clr3), (0.75, clr4), (1, clr2)]
cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors)

h2 = ax3.imshow(final_result, cmap=cmap_custom, interpolation='nearest')  # 'OrRd', interpolation='nearest')

# xpos, ypos = np.meshgrid(np.arange(final_result.shape[1]), np.arange(final_result.shape[0]))
# h2 = ax3.scatter(xpos, ypos, c=final_result.values.flatten(), s=50, cmap=cmap_custom, edgecolors='dimgray', marker='o', alpha=0.9)

# Add colorbar for ax3 (circular heatmap)
cax = add_right_cax(ax3, pad=0.03, width=0.02)
c2 = fig2.colorbar(h2, cax=cax, orientation='vertical')
c2.set_ticks([-2, -1, 0, 1, 2])
c2.set_ticklabels(['-2', '-1', '0', '1', '2'])
c2.set_label('Standard anomaly of DSTR')

ax3.set_ylim(-0.5, 11.5)
ax3.set_yticks(np.arange(0, 12))
ax3.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

ax3.set_xlim(-0.5, 36.5)
ax3.set_xticks(np.arange(0, final_result.shape[1]))
ax3.set_xticklabels(final_result.columns, rotation=xrot, ha='left')

xlim = ax3.get_xlim()
ylim = ax3.get_ylim()
xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
ypos = ylim[1] - 0.01 * (ylim[1] - ylim[0])
ax3.text(xpos, ypos, '(c)', fontsize=12, fontweight='bold', va='top', ha='left')

# Save the figure
outname = 'Fig_3_Tempora_pattern_of_DSTR_at_different_scales.png'
filepath = combinename(outpath, outname)
plt.savefig(filepath, dpi=300)
print(filepath)
plt.close()






















