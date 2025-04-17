Data and code for Daily spatial temperature range: Spatiotemporal pattern and climate change response

Any question, please contact Fayong Liu (liufayong3375@igsnrr.ac.cn)

1. Data
1.1. ERA5-Land hourly t2m data (1950-2023)
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
1.2. MERRA2 t2m data (1980-2023)
https://disc.gsfc.nasa.gov/datasets/M2I1NXASM_5.12.4/summary?keywords=MERRA2%20inst1_2d_asm
1.3. China meteorological forcing dataset (1979-2018) 
https://data.tpdc.ac.cn/en/data/8028b944-daaa-4511-8769-965612652c49
1.4. ERA5-Land monthly data of albedo, solar shortwave radiation, atmospheric longwave radiation, net ground longwave radiation, sensible heat flux and latent heat flux
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview
1.5. World boundary
https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-coastline/
1.6. National and provincial boundary
http://bzdt.ch.mnr.gov.cn/
1.7. China meteorological station, and DEM 1km
https://www.resdc.cn/data.aspx?DATAID=349
1.8. Catalogue of Life China: 2023 Annual Checklist
http://www.sp2000.org.cn/
1.9. Chinese Climate Physical Risk Index (CCPRI)
https://figshare.com/articles/dataset/Climate_Physical_Risk_Index_CPRI_/25562229/1 

Note that data 1.1.-1.4. are not provided here due to their large size, you can download them from the link below each data, and data 1.5.-1.9. see in folder, ./data/
2. Code, need to replace the path when used, see in folder, ./code/
2.1. Download and preprocess data
1) Downloading ERA5-Land data
./download/, this folder provides some  codes downloading the data from ERA5-Land, and other data can be downloaded directly from the path provided in the 2.1 Data section

2) Preprocessing data for ridge regression
pre_ridge_xxxx.py

3) Preprocessing data for wind extremes, for potential implications of DSTR
pre_wind_extremes_xxxx.py

2.2. Calculating, some  code here are also provide visualization function, but most of the figure here are just the processing result, not show in the main text
1) Calculate daily spatial temperature range
cal_eachyear_daily_temperature_range_xxxx(region)_xxxx(dataset, default=ERA5-Land).py

2) Calulate the basic characteristics of DSTR (time series, hour distribuiton, and STmax/STmin distribution), as well as visualizations of them
cal_dstr_characteristics_and_visulizations_xxxx(dataset, default=ERA5-Land).py, the distribution of STmax/STmin were later used in Fig.2, S2-S4

3) Calculate temporal trend and p values, as well as visualizations of them, and  the comparison between different datasets
cal_tempo_trend_and_visualization_xxxx(dataset, default=ERA5-Land).py, some of the results were later used in Fig.S8-S9

4) Calculate comparison between different periods and p values, as well as visualizations of them, as a supplementation of the temporal trend
cal_comparison_and_visualization_xxxx(dataset, default=ERA5-Land).py

2.3. plot
plot_figx_xxxx.py
