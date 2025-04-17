# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# this file is for plot Fig. 1. Multiyear DSTR at different spatial scales and driving factors


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


# def a function read csv data
# return, labels, label name, list, length=33, except Macau
# datals, combined data, list, length=n, according to the user and csvfile
def readcsv(path, datanum):
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


fig_outpath = r'/data1/fyliu/a_temperature_range/result/fig_main/'
dstr_ridge_outpath = r'/data1/fyliu/a_temperature_range/result/result_dstr_pattern_ridge/'
os.makedirs(fig_outpath, exist_ok=True)
os.makedirs(dstr_ridge_outpath, exist_ok=True)

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

fig1 = plt.figure(figsize=(13, 12), dpi=300)
grid = GridSpec(3, 3, figure=fig1, left=0.06, bottom=0.05, right=0.95, top=0.98,
                wspace=0.23, hspace=0.28)
ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[0, 1])
ax3 = plt.subplot(grid[0, 2])
ax4 = plt.subplot(grid[1, 0:3])
ax5 = plt.subplot(grid[2, 0])
ax6 = plt.subplot(grid[2, 1])
ax7 = plt.subplot(grid[2, 2])

# fig (a)
# DSTR data of different scale
# ========================================================================================
path = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_dstr/'
datanum = 3
labels, datals = readcsv(path, datanum)
year_df = datals[0].iloc[:, 0]
dstr = datals[2]
year_filtered = year_df[year_df.between(1950, 2023)]
fiter_dstr = dstr.loc[year_filtered.index]
pro_data = fiter_dstr.iloc[:, 0:33].values
pro_lab = labels[0:33]
pro_data_flat = pro_data.flatten()
nation_data = fiter_dstr.iloc[:, 33].values
hemi_data = fiter_dstr.iloc[:, 34:36].values
hemi_lab = labels[34:36]
hemi_data_flat = hemi_data.flatten()
glo_data = fiter_dstr.iloc[:, 36].values
scale_data = [pro_data_flat, nation_data, hemi_data_flat, glo_data]
labels = ['Provincial', 'National', 'Hemispheric', 'Global']
colors = ['steelblue', 'rosybrown', 'maroon', 'dimgray']

sns.boxplot(
    data=scale_data,
    palette=colors,
    ax=ax1,
    boxprops=dict(alpha=alp),
    flierprops=dict(marker='+', markersize=5)
)

ax1.set_ylabel('DSTR (\u2103)')
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(labels, rotation=0)
addfignum(ax1, '(a)', ftsz_num)

# save csv
means = np.array([np.mean(data) for data in scale_data])
std_devs = np.array([np.std(data) for data in scale_data])

results_df = pd.DataFrame({
    'Scale': labels,
    'Mean': means,
    'Std': std_devs,
    'Mean - Std': means - std_devs,
    'Mean + Std': means + std_devs
})

# save csv
outname = 'dstr_different_scale_1950_2023.csv'
filename = combinename(dstr_ridge_outpath, outname)
results_df.to_csv(filename)
print(filename)

# fig (b)
# corelation between area & dstr
# ========================================================================================
path1 = r'/data1/fyliu/a_temperature_range/process_data/ridge_data/dstr/mean_dstr.csv'
path2 = r'/data1/fyliu/a_temperature_range/process_data/correlation/area_dem_lon_lat_plant.xlsx'
data1 = pd.read_csv(path1, index_col=0)
data1 = data1[data1['year'].between(1950, 2023)]
data1 = data1.mean()
data1 = data1.drop('year').rename_axis('pro').rename('DSTR')

data2 = pd.read_excel(path2)
data2 = data2.iloc[:, :6]

# merge them
merdata = pd.merge(data1, data2, on='pro')
merdata.columns = ['pro', 'DSTR', 'Area', 'Elevation_range', 'Longitude_range', 'Latitude_range', 'Sea_distance_range']

for col in merdata.columns:
    merdata[col] = pd.to_numeric(merdata[col], errors='coerce')

x1 = merdata['Area'].iloc[:33]
y1 = merdata['DSTR'].iloc[:33]

# linear fit
p = sns.regplot(x=x1, y=y1, ci=95, marker='o', color='steelblue', ax=ax2)
# cal slope, intercept, r2 and p value
slope, intercept, corr, pvalue, sterr = stats.linregress(x=x1, y=y1)
# add x, y label and text
ax2.set_xlabel('Area (km$^2$)')
ax2.set_ylabel('DSTR (\u2103)')
ax2.ticklabel_format(style='sci', scilimits=(-2, 2), axis='x')

# add slope, intercept, r2 and p value
textname = f'$R^2$ = {round(corr, 2)}, $P$ = {round(pvalue, 2)}'
xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
xpos = xlim[0] + 0.53 * (xlim[1] - xlim[0])
ypos = ylim[0] + 0.1 * (ylim[1] - ylim[0])
ax2.text(xpos, ypos, textname, verticalalignment='top', horizontalalignment='left', color='steelblue')
addfignum(ax2, '(b) Provincial', ftsz_num)

# fig (c)
# plot absolute dstr and dstr/area on hemispherical scale
# ========================================================================================
# add STmax and STmin
path1 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_stmax/'
path2 = r'/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_stmin/'
datanum = 3
labels1, datals1 = readcsv(path1, datanum)
labels2, datals2 = readcsv(path2, datanum)
year_df = datals1[0].iloc[:, 0]
stmax = datals1[2]
stmin = datals2[2]
year_filtered = year_df[year_df.between(1950, 2023)]
fiter_stmax = stmax.loc[year_filtered.index]
fiter_stmin = stmin.loc[year_filtered.index]
hemi_stmax = fiter_stmax.iloc[:, 34:36].values
hemi_stmin = fiter_stmin.iloc[:, 34:36].values
hemi_lab = labels1[34:36]
data_ls = [hemi_data, hemi_stmax, hemi_stmin]
colors = ['dimgray', 'maroon', 'steelblue']
lab = ['DSTR', 'STmax', 'STmin']

for i in range(3):
    dstr_data = data_ls[i]
    clr = colors[i]
    linewd = 1.5
    sns.boxplot(
        data=dstr_data,
        ax=ax3,
        color=clr,
        width=0.3,
        boxprops=dict(facecolor='none', edgecolor=clr, linewidth=linewd),
        whiskerprops=dict(color=clr, linewidth=linewd),
        capprops=dict(color=clr, linewidth=linewd),
        medianprops=dict(color=clr, linewidth=linewd),
        flierprops=dict(marker='+', markersize=5, markeredgecolor=clr)
    )

    # save csv
    results = []
    for idx, label in enumerate(hemi_lab):
        dstr_values = dstr_data[:, idx]
        mean_dstr = np.mean(dstr_values)
        std_dstr = np.std(dstr_values)

        results.append({
            'Scale': label,
            'Mean': mean_dstr,
            'Std': std_dstr,
            'Mean - Std': mean_dstr - std_dstr,
            'Mean + Std': mean_dstr + std_dstr
        })

    results_df = pd.DataFrame(results)

    outname = lab[i] + '_hemispherical_1950_2023.csv'
    filename = combinename(dstr_ridge_outpath, outname)
    results_df.to_csv(filename)
    print(filename)

ax3.set_ylabel('Temperature (\u2103)')
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['North hemisphere', 'South hemisphere'])
# ax3.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
addfignum(ax3, '(c) Hemispheric', ftsz_num)

ypos = [76, 41, -41]
for i in range(3):
    ax3.text(0.2, ypos[i], lab[i], color=colors[i])


# fig (d)
# plot absolute dstr and dstr/area on provincial scale
# ========================================================================================
def plot_2y_box(axname, dstr_data, area_data, xlab, clr, ftsz_num, pnum, rot, ha):
    data_len = dstr_data.shape[1]
    dstr_per_area = dstr_data.copy()
    for ii in range(data_len):
        dstr_per_area[:, ii] = dstr_per_area[:, ii] / area_data[ii]

    positions = np.arange(data_len)

    linewd = 1.5
    # y1 box
    sns.boxplot(
        data=dstr_data,
        ax=axname,
        color=clr[0],
        width=0.3,
        positions=positions - 0.25,
        boxprops=dict(facecolor='none', edgecolor=clr[0], linewidth=linewd),
        whiskerprops=dict(color=clr[0], linewidth=linewd),
        capprops=dict(color=clr[0], linewidth=linewd),
        medianprops=dict(color=clr[0], linewidth=linewd),
        flierprops=dict(marker='+', markersize=5, markeredgecolor=clr[0])
    )
    axname.set_ylabel('DSTR (\u2103)')

    axname.yaxis.label.set_color(clr[0])
    axname.tick_params(axis='y', colors=clr[0])
    axname.set_xticks(np.arange(33))
    axname.set_xticklabels(xlab, rotation=rot, ha=ha)

    ax_copy = axname.twinx()
    sns.boxplot(
        data=dstr_per_area,
        ax=ax_copy,
        color=clr[1],
        width=0.3,
        positions=positions + 0.25,
        boxprops=dict(facecolor='none', edgecolor=clr[1], linewidth=linewd),
        whiskerprops=dict(color=clr[1], linewidth=linewd),
        capprops=dict(color=clr[1], linewidth=linewd),
        medianprops=dict(color=clr[1], linewidth=linewd),
        flierprops=dict(marker='+', markersize=5, markeredgecolor=clr[1])
    )

    ax_copy.set_ylabel('DSTR/Area (\u2103/km$^2$)')
    #ax_copy.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
    ax_copy.set_yscale('log')

    ax_copy.yaxis.label.set_color(clr[1])
    ax_copy.tick_params(axis='y', colors=clr[1])

    addfignum(axname, pnum, ftsz_num)

    # save csv
    results = []
    for idx, label in enumerate(xlab):
        dstr_values = dstr_data[:, idx]
        mean_dstr = np.mean(dstr_values)
        std_dstr = np.std(dstr_values)

        dstr_per_area_values = dstr_per_area[:, idx]
        mean_dstr_per_area = np.mean(dstr_per_area_values)
        std_dstr_per_area = np.std(dstr_per_area_values)

        results.append({
            'Scale': label,
            'Mean': mean_dstr,
            'Std': std_dstr,
            'Mean - Std': mean_dstr - std_dstr,
            'Mean + Std': mean_dstr + std_dstr,
            'Mean/Area': mean_dstr_per_area,
            'Std/Area': std_dstr_per_area,
            '(Mean - Std)/Area': mean_dstr_per_area - std_dstr_per_area,
            '(Mean + Std)/Area': mean_dstr_per_area + std_dstr_per_area
        })

    results_df = pd.DataFrame(results)

    outname = 'dstr_' + pnum[4:] + '_1950_2023.csv'
    filename = combinename(dstr_ridge_outpath, outname)
    results_df.to_csv(filename)
    print(filename)

    return


clr = ['dimgray', 'steelblue']
area_data = merdata['Area'].values
plot_2y_box(ax4, pro_data, area_data, pro_lab, clr, ftsz_num, '(d) Provincial', -45, 'left')


# fig (e), (f), (g)
# Ridge rege=ression
# ========================================================================================
# def a function for redge regression
def ridgeReg(df, outpath, out_key, ftsz_num, lab1, axname, plot_pred, plot_con, pnum, clr):
    # standardlization
    transfer = StandardScaler()
    df1 = transfer.fit_transform(df)

    # divie data into training data (80%) and test data (20%)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        df1[:, 1:],
        df1[:, 0],
        test_size=0.2,
        random_state=1234)

    # set different Lambda
    Lambdas = np.logspace(-5, 3, 200)
    # create empty list for result
    ridge_cofficients = []
    # iterate for Lambda
    for Lambda in Lambdas:
        ridge = Ridge(alpha=Lambda)
        # cal coefficiency
        ridge.fit(x_train, y_train)
        ridge_cofficients.append(ridge.coef_)

    # 10-fold cross-validation method
    ridge_cv = RidgeCV(alphas=Lambdas, scoring='neg_mean_squared_error', cv=10)
    ridge_cv.fit(x_train, y_train)
    ridge_best_Lambda = ridge_cv.alpha_

    # use best Lambda to fit
    ridge = Ridge(alpha=ridge_best_Lambda)
    ridge.fit(x_train, y_train)
    coeff = pd.DataFrame(index=['Intercept'] + df.columns.tolist()[1:],
                         data=[ridge.intercept_] + ridge.coef_.tolist())
    print('Coefficient:')
    print(coeff)
    print("                                                                   ")
    # save result
    outname = out_key + '_coefficients.csv'
    filepath = combinename(outpath, outname)
    coeff.to_csv(filepath, sep=',')

    # cal relative contribution
    con = coeff.iloc[1:, 0]
    con = con.abs()
    con = 100 * con / con.sum()
    print('Relative contributions:')
    print(con)
    print("                                                                   ")

    # prediction
    ridge_train = ridge.predict(x_train)
    R2_train = r2_score(y_train, ridge_train)
    ridge_test = ridge.predict(x_test)
    R2_test = r2_score(y_test, ridge_test)
    print('Training and testing R$^2$:')
    print(R2_train, R2_test)
    # plot
    xrot = -30
    yrot = -45
    sz = 30

    if plot_pred:
        ax = axname[0]
        # predictions
        ax.scatter(ridge_train, y_train, s=sz, c='maroon', marker='o', alpha=alp, label='Train')
        ax.scatter(ridge_test, y_test, s=sz, c='steelblue', marker='o', alpha=alp, label='Test')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], color='grey', linestyle='--', linewidth=2, label='1:1 line')
        textname1 = 'Train $R^2$ = ' + str(round(R2_train, 2))
        textname2 = 'Test $R^2$ = ' + str(round(R2_test, 2))

        xpos = xlim[0] + 0.01 * (xlim[1] - xlim[0])
        ypos = ylim[1] - 0.1 * (ylim[1] - ylim[0])
        ypos1 = ylim[1] - 0.2 * (ylim[1] - ylim[0])
        ax.text(xpos, ypos, textname1, verticalalignment='top', horizontalalignment='left')
        ax.text(xpos, ypos1, textname2, verticalalignment='top', horizontalalignment='left')
        ax.set_xlabel('Simulations')
        ax.set_ylabel('Observations')
        ax.legend(ncol=1, loc='lower right', frameon=False)
        addfignum(ax, pnum[1], ftsz_num)

    if plot_con:
        ax = axname[1]
        # plot contributions
        x = range(len(con))

        ax.barh(x, con, color=clr, edgecolor='k', alpha=alp, height=0.8)
        # ax9.barh(x, con, color='none', edgecolor=clr, alpha=alp, height =0.8)
        ax.set_ylim(-0.5, len(lab1) - 1)
        ax.set_yticks(range(len(lab1) - 1))
        ax.set_yticklabels(lab1[1:], rotation=yrot, va='center')
        ax.set_xlabel('Contributions (%)')
        addfignum(ax, pnum[2], ftsz_num)
        for i in range(len(con)):
            text = str(round(con.iloc[i], 1))
            ax.text(con.iloc[i] + 1, i, text, ha='left', va='center', c='dimgray')
        ax.set_xlim(0, max(con) + 10)

        if plot_pred == 0:
            textname1 = 'Train $R^2$ = ' + str(round(R2_train, 2))
            textname2 = 'Test $R^2$ = ' + str(round(R2_test, 2))
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xpos = xlim[0] + 0.6 * (xlim[1] - xlim[0])
            ypos = ylim[0] + 0.2 * (ylim[1] - ylim[0])
            ypos1 = ylim[0] + 0.1 * (ylim[1] - ylim[0])
            ax.text(xpos, ypos, textname1, verticalalignment='top', horizontalalignment='left')
            ax.text(xpos, ypos1, textname2, verticalalignment='top', horizontalalignment='left')

    return


merdata1 = merdata.copy().iloc[:33]
columns_to_normalize = ['pro', 'DSTR', 'Elevation_range', 'Longitude_range', 'Latitude_range', 'Sea_distance_range']
merdata1[columns_to_normalize] = merdata1[columns_to_normalize].div(merdata1['Area'], axis=0)
df0 = merdata1[['DSTR', 'Latitude_range', 'Longitude_range', 'Elevation_range', 'Sea_distance_range']]

lab1 = ['DSTR', 'Lat', 'Lon', 'Ele', 'Sea dis']
axname = [ax5, ax6]
pnum = ['', '(e) DSTR-Location', '(f) DSTR-Location']
clr = ['steelblue', 'rosybrown', 'maroon', 'lightgray']
ridgeReg(df0, dstr_ridge_outpath, 'DSTR_Location', ftsz_num, lab1, axname, 1, 1, pnum, clr)

ridge_data = merdata[['Area', 'DSTR']].iloc[:33]
radiation_dir = r'/data1/fyliu/a_temperature_range/process_data/ridge_data/radiation/dstr'
radiation_path = os.listdir(radiation_dir)
for ffile in radiation_path:
    varname = ffile[10:-4]
    tempt = pd.read_csv(os.path.join(radiation_dir, ffile), index_col=0)
    tempt = tempt.mean()
    ridge_data[varname] = tempt.values[:33]

columns_to_normalize = ['ssrd', 'fal', 'strd', 'str_earth', 'sshf', 'slhf']
ridge_data[columns_to_normalize] = ridge_data[columns_to_normalize].div(ridge_data['Area'], axis=0)
df1 = ridge_data[columns_to_normalize]
df = pd.concat([df0['DSTR'].reset_index(drop=True), df1.reset_index(drop=True)], axis=1)

lab1 = ['DSTR', 'Es', 'Albedo', 'Ea', 'Eg', 'SH', 'LH']
axname = ['', ax7]
pnum = ['', '', '(g) DSTR-Energy']
clr = ['steelblue', 'lightsteelblue', 'maroon', 'rosybrown', 'dimgray', 'lightgray']

ridgeReg(df, dstr_ridge_outpath, 'Lat_Radiation', ftsz_num, lab1, axname, 0, 1, pnum, clr)

pos = ax4.get_position()
ax4.set_position([pos.x0, pos.y0 + 0.015, pos.width, pos.height])

outname = 'Fig_1_Multiyear_DSTR_at_different_spatial_scales_and_driving_factors.png'
filepath = combinename(fig_outpath, outname)
plt.savefig(filepath, dpi=300)
print(filepath)
plt.close()








































