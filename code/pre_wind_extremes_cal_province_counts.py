# -*- coding: utf-8 -*-
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import os
import glob


# this file is for calculating wind extremes frequency in each province in China

# here calculate at each province
# province boundary
proshp = r'/data1/fyliu/a_temperature_range/data/boundary/China_GS(2020)4619/China_provincial_polygon.shp'
conpath = r'/data1/fyliu/a_temperature_range/data/boundary/China_GS(2020)4619/China_province_code.xlsx'
# set f(x) table
mapping_df = pd.read_excel(conpath)
pro_maps = {}
for index, row in mapping_df.iterrows():
   pro_maps[str(row['DZM'])] = row['key']

# read gdf
progdf = gpd.read_file(proshp, encoding='utf-8')
progdf['pro'] = progdf['DZM'].map(pro_maps)
gdf = progdf
base_directory = r'/data1/fyliu/b_compound_hourly_rainfall_wind/data/wind/'
output_directory = r'/data1/fyliu/b_compound_hourly_rainfall_wind/data/wind_province/'
all_file = os.listdir(base_directory)
all_file.sort()
for excel_file in all_file:
   event_data = pd.read_csv(os.path.join(base_directory, excel_file))
   event_data['Date'] = pd.to_datetime(event_data[['year', 'month', 'mday']].rename(columns={'mday': 'day'}))
   event_data['geometry'] = event_data.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
   event_gdf = gpd.GeoDataFrame(event_data, geometry='geometry', crs=gdf.crs)
   event_with_province = gpd.sjoin(event_gdf, gdf, how='left', op='within')
   event_data['Province'] = event_with_province['pro']
   event_counts = event_data.groupby(['Date', 'Province']).size().reset_index(name='Event Count')
   event_counts_pivot = event_counts.pivot(index='Date', columns='Province', values='Event Count')
   event_counts_pivot = event_counts_pivot.fillna(0)
   event_counts_pivot = event_counts_pivot.reset_index()
   output_file = os.path.join(output_directory, f'pro_{excel_file}')
   event_counts_pivot.to_csv(output_file)
   print(f"Processed and saved: {output_file}\n")


# here combine them
input_directory = r'/data1/fyliu/b_compound_hourly_rainfall_wind/data/wind_province/'
all_files = glob.glob(os.path.join(input_directory, "*.csv"))
df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)
combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df = combined_df.fillna(0)
combined_df['Year'] = combined_df['Date'].dt.year
print(combined_df.head())
output_file = r'/data1/fyliu/b_compound_hourly_rainfall_wind/data/province_wind_extremes_details.csv'
combined_df.to_csv(output_file, index=False)
print(output_file)
combined_df = combined_df.drop(columns=['Date'])
combined_df = combined_df.drop(columns=['Unnamed: 0'])
province_year_counts = combined_df.groupby('Year').sum().reset_index()
print(province_year_counts.head())
output_file = r'/data1/fyliu/b_compound_hourly_rainfall_wind/data/province_wind_extremes_counts.csv'
province_year_counts.to_csv(output_file)
print(output_file)
