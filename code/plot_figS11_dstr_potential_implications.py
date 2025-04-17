# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import pandas as pd
import seaborn as sns

# this file is for plotting potential implications of DSTR, i.e., fig S11

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
    xpos = xlim[0] + 0.03 * (xlim[1] - xlim[0])
    ypos = ylim[1] - 0.03 * (ylim[1] - ylim[0])
    axn.text(xpos, ypos, fignum, verticalalignment='top', horizontalalignment='left', fontweight='bold',
             fontsize=ftsz_num)

    return


# plot all scale, multiyear
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
alp = 0.7

# read dstr data
dstr_path = r'/data1/fyliu/a_temperature_range/process_data/ridge_data/dstr/mean_dstr.csv'
dstr_df = pd.read_csv(dstr_path, index_col=0)
dstr = dstr_df.mean().reset_index()
dstr.columns = ['pro', 'dstr_1950_2023']
dstr = dstr.iloc[1:]

dstr1 = dstr_df[dstr_df['year'] == 2023].T
dstr1 = dstr1.reset_index()
dstr1.columns = ['pro', 'dstr_2023']
dstr1 = dstr1.iloc[1:]

dstr2 = dstr_df[(dstr_df['year'] >= 1979) & (dstr_df['year'] <= 2018)].mean().reset_index()
dstr2.columns = ['pro', 'dstr_1979_2018']
dstr2 = dstr2.iloc[1:]

dstr3 = dstr_df[(dstr_df['year'] >= 1993) & (dstr_df['year'] <= 2023)].mean().reset_index()
dstr3.columns = ['pro', 'dstr_1993_2023']
dstr3 = dstr3.iloc[1:]

# read extreme wind data
wind_path = r'/data1/fyliu/a_temperature_range/process_data/correlation/province_wind_extremes_counts.csv'
wind = pd.read_csv(wind_path, index_col=0)
wind = wind.mean().reset_index()
wind.columns = ['pro', 'Extreme winds']
wind = wind.iloc[1:]

# read other extremes data
ext_path = r'/data1/fyliu/a_temperature_range/data/Chinese Climate Physical Risk Index (CCPRI).xlsx'
ext = pd.read_excel(ext_path, sheet_name='Province level')
ext = ext.groupby('Province').mean().reset_index()
ext = ext[['Province', 'LTD', 'HTD', 'ERD', 'EDD', 'Climate Physical Risk Index (CPRI)']]
ext = ext.rename(columns={'Province': 'pro', 'Climate Physical Risk Index (CPRI)': 'CPRI'})

# read plant & animal diversity data
diver_path = r'/data1/fyliu/a_temperature_range/data/biodiversity.xlsx'
diver = pd.read_excel(diver_path)
diver = diver[['pro', 'Animalia species', 'Plantae species']]

# read weather station data
sta_path = r'/data1/fyliu/a_temperature_range/data/station.xls'
sta = pd.read_excel(sta_path, sheet_name='count1')

# read area
area_path = r'/data1/fyliu/a_temperature_range/process_data/correlation/area_dem_lon_lat_plant.xlsx'
area = pd.read_excel(area_path)
area = area[['pro', 'Area (km2)']]

merged_df = dstr.merge(dstr1, on='pro', how='inner') \
    .merge(dstr2, on='pro', how='inner') \
    .merge(dstr3, on='pro', how='inner') \
    .merge(sta, on='pro', how='inner') \
    .merge(wind, on='pro', how='inner') \
    .merge(ext, on='pro', how='inner') \
    .merge(diver, on='pro', how='inner') \
    .merge(area, on='pro', how='inner')

# plot
fig1 = plt.figure(figsize=(10, 10), dpi=300)
grid = GridSpec(3, 3, figure=fig1, left=0.07, bottom=0.05, right=0.98, top=0.98,
                wspace=0.27, hspace=0.2)
ax_ls = [plt.subplot(grid[i, j]) for i in range(3) for j in range(3)]
fignum_ls = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

for i in range(9):
    ax = ax_ls[i]
    if i == 0:
        xvar = 'dstr_1979_2018'
    elif (i > 0) and (i < 6):
        xvar = 'dstr_1993_2023'
    elif i == 8:
        xvar = 'dstr_1950_2023'
    else:
        xvar = 'dstr_2023'

    tempt2 = merged_df[xvar].values
    tempt1 = merged_df.iloc[:, 5 + i].values
    # cal r2 and p value
    corr, pvalue = stats.pearsonr(tempt2, tempt1)

    #    ax.plot(tempt2, tempt1, 'o', markersize=5, markerfacecolor='dimgray', markeredgewidth=1, markeredgecolor='dimgray', alpha = 0.5)
    if pvalue < 0.05:
        clr = 'steelblue'
    else:
        clr = 'dimgray'
    p = sns.regplot(x=tempt2, y=tempt1, ci=95, marker='o', color=clr, ax=ax)

    # add x, y label and text
    ylabelname = merged_df.columns[5 + i]
    ax.set_ylabel(ylabelname)
    # ax.set_xlabel('DSTR/Area (\u2103/km$^2$)')
    ax.set_xlabel('DSTR (\u2103)')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))
    textname = '$R^2$ = ' + str(round(corr, 2)) + ', $P$ = ' + str(round(pvalue, 2))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xpos = xlim[1] - 0.55 * (xlim[1] - xlim[0])
    ypos = ylim[0] + 0.1 * (ylim[1] - ylim[0])
    ax.text(xpos, ypos, textname, verticalalignment='top', horizontalalignment='left')
    addfignum(ax, fignum_ls[i], ftsz_num)

    if i == 0:
        for j in range(len(tempt2)):
            this_pro = merged_df['pro'].values[j]
            ax.text(tempt2[j] + 1, tempt1[j], this_pro, va='center', ha='left', color='steelblue', fontsize=7.5)

# save picture
outpath = r'/data1/fyliu/a_temperature_range/result/fig_main'
outname = 'Fig S11 Linear fit between DSTR and extreme events biodiversity and number of meteorological stations at provincial scale.png'
outname = outname.replace(' ', '_')
filepath = combinename(outpath, outname)
plt.savefig(filepath, dpi=300)
print(filepath)
plt.close()
