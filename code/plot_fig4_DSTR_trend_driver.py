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
from matplotlib.ticker import MaxNLocator


# this file is for plotting dstr trend and driving factors, i.e., fig 4


# def a function to combine fig_outpath and outname
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
        axname.yaxis.set_major_locator(MaxNLocator(5))

    axname.set_ylabel(ylab)
    axname.set_xlim(baseyear - 1, endyear + 1)
    axname.set_xticks(range(baseyear, endyear + 1, 15))

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

    axname.text(0.5, 0.8, '* $P$ < 0.05')

    # set pnum
    addfignum(axname, pnum, ftsz_num)

    return


# def a function for redge regression
def ridgeReg(df, ls_outpath, out_key, ftsz, ftsz_num, lab1, axname, pnum, clr):
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
    filepath = combinename(ls_outpath, outname)
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

    # plot contributions
    yrot = -45
    x = range(len(con))
    axname.barh(x, con, color=clr, edgecolor='k', alpha=alp, height=0.8)
    # axname.barh(x, con, color='none', edgecolor=clr, alpha=alp, height =0.8)
    axname.set_ylim(-0.5, len(lab1) - 1.5)
    axname.set_yticks(range(len(lab1) - 1))
    axname.set_yticklabels(lab1[1:], rotation=yrot, va='center')
    axname.set_xlabel('Contributions (%)')
    addfignum(axname, pnum, ftsz_num)
    for i in range(len(con)):
        text = str(round(con.iloc[i], 1))
        axname.text(con.iloc[i] + 1, i, text, ha='left', va='center', c='dimgray')
    axname.set_xlim(0, max(con) + 10)

    textname1 = 'Train $R^2$ = ' + str(round(R2_train, 2))
    textname2 = 'Test $R^2$ = ' + str(round(R2_test, 2))
    xlim = axname.get_xlim()
    ylim = axname.get_ylim()
    xpos = xlim[0] + 0.55 * (xlim[1] - xlim[0])
    ypos = ylim[0] + 0.7 * (ylim[1] - ylim[0])
    ypos1 = ylim[0] + 0.6 * (ylim[1] - ylim[0])
    axname.text(xpos, ypos, textname1, verticalalignment='top', horizontalalignment='left')
    axname.text(xpos, ypos1, textname2, verticalalignment='top', horizontalalignment='left')

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

fig_outpath = r'/data1/fyliu/a_temperature_range/result/fig_main/'
ls_outpath = r'/data1/fyliu/a_temperature_range/result/result_dstr_trend_ridge/'

fig = plt.figure(figsize=(13, 10), dpi=300)
grid = GridSpec(3, 4, figure=fig, left=0.06, bottom=0.05, right=0.98, top=0.97, wspace=0.25, hspace=0.36)
ax_ls = [plt.subplot(grid[0, j]) for j in range(4)]
ax_ls.append(plt.subplot(grid[1, 0:]))
for j in range(4):
    ax_ls.append(plt.subplot(grid[2, j]))

# plot DSTR, STmax, STmin trend
# =============================================================================================
# read DSTR data, iterate by region
varname = ['dstr', 'stmax', 'stmin']
all_dstr = {}
for baseyear, endyear in [(1950, 2023)]:

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
lab = ['DSTR', 'STmax', 'STmin']
pnum_ls = ['(a) World', '(b) SH', '(c) NH', '(d) China']

for i in range(4):
    if i == 0:
        ylab = 'DSTR (\u2103)'
    else:
        ylab = ''
    axname = ax_ls[i]
    trend_ls = [dstr_ls[i][var] for var in varname]

    slope_arr = plot_tempo_trend_one(axname, trend_ls, clrname, ylab, lab, baseyear, endyear, pnum_ls[i])

# plot provincial trend
# ==========================================================================
path = r'/data1/fyliu/a_temperature_range/process_data/tempo_trend_70_yr/'
# 0-2 reperensts slope and P
datanum = 2
labels, datals = readcsv(path, datanum, 0)
slopels = datals[0][:33]
pls = datals[1][:33]

ylabel = 'Slope (\u2103/100 yr)'
axname = ax_ls[4]
plotbar(axname, slopels[:, 1] * 100, clrname[0], ylabel, pls[:, 1], '(e) Provinces in China', 1)
axname.set_xticklabels(labels[:33], rotation=-75)
pos = axname.get_position()
axname.set_position([pos.x0, pos.y0 + 0.04, pos.width, pos.height])


# read radiation
radiation = []
for para_var in ['STmax', 'STmin']:
    key1 = para_var.lower()
    radiation_dir = f'/data1/fyliu/a_temperature_range/process_data/ridge_data/radiation/{key1}'
    radiation_path = os.listdir(radiation_dir)
    rad = {}
    for ffile in radiation_path:
        varname = ffile[6 + len(para_var):-4]
        tempt = pd.read_csv(os.path.join(radiation_dir, ffile), index_col=0)
        rad[varname] = tempt

    var_ls = ['ssrd', 'fal', 'strd', 'str_earth', 'sshf', 'slhf']

    all_rad_ls = {}
    all_scale_name = rad.get('ssrd').columns[0:-1]
    for scale in all_scale_name:
        scale_data = []
        for var in var_ls:
            tempt = rad.get(var)[scale].rename(var)
            scale_data.append(tempt)
        scale_data = pd.concat(scale_data, axis=1)
        scale_data['year'] = np.arange(1950, 2024)
        all_rad_ls[scale] = scale_data
    radiation.append(all_rad_ls)

# plot ridge regression
clr = ['steelblue', 'lightsteelblue', 'maroon', 'rosybrown', 'dimgray', 'lightgray']
labels = ['SH', 'NH', 'Guizhou', 'Heilongjiang']
pnum_ls = ['(f) World/SH-STmin', '(g) NH-STmin', '(h) Guizhou-STmax', '(i) Heilongjiang-STmin']
rad_num = [0, 0, 1, 0]
for ii in range(len(labels)):
    print(labels[ii])
    if rad_num[ii]:
        all_rad_ls = radiation[0]
        para_var = 'STmax'
    else:
        all_rad_ls = radiation[1]
        para_var = 'STmin'

    axname = ax_ls[ii+5]
    dstr_data = all_dstr.get(labels[ii])[key1]
    rad_data = all_rad_ls.get(labels[ii])
    df = pd.concat([dstr_data.reset_index(drop=True), rad_data[var_ls].reset_index(drop=True)], axis=1)
    out_key = labels[ii] + '_' + para_var + '_Radiation'
    lab1 = [para_var, 'Es', 'Albedo', 'Ea', 'Eg', 'SH', 'LH']
    ridgeReg(df, ls_outpath, out_key, ftsz, ftsz_num, lab1, axname, pnum_ls[ii], clr)

    print('=========================================================\n')
    print('=========================================================\n')

outname = 'Fig_4_Trend_of_DSTR_and_driving_factors_at_different_scales.png'
filename = combinename(fig_outpath, outname)
plt.savefig(filename, dpi=300)
plt.close()
print(filename)




























