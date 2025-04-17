# -*- coding: utf-8 -*-
import xarray as xr
import pandas as pd
import numpy as np
import os

# this file is for calculating the area-frequency-weighted averages radiations at each scale

# def a function to combine outpath and outname
def combinename(outpath, outname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filepath = os.path.join(outpath, outname)

    return filepath


# Extract multiple variables based on CSV lat/lon
def get_nc_data_for_st(df, nc_data_1, nc_data_2, variables):
    """Extract data for multiple variables from NetCDF for each lat/lon in the input dataframe."""
    results = []
    for index, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']

        # Use .sel() with method='nearest' to find the nearest grid point in the NetCDF dataset
        selected_point = nc_data_1.sel(latitude=lat, longitude=lon, method='nearest')
        selected_area = nc_data_2.sel(lat=lat, lon=lon, method='nearest')

        # Initialize a dictionary to store the data for this row
        data = {
            'lat': lat,
            'lon': lon,
            'area': selected_area['area'].values,
            'count': row['count']  # Include the occurrence count from the CSV
        }

        # Loop through all the variables and extract their values
        for var in variables:
            data[var] = selected_point[var].values

        # Append the result to the list
        results.append(data)

    return pd.DataFrame(results)


def weighted_average(df, value_columns):
    """Calculate weighted averages for multiple variables based on a combined weight (area * count), ignoring NaN values."""
    results = {}
    # Create a new combined weight: area * count
    combined_weight = df['count'] * df['area']

    for value_column in value_columns:
        # Filter out NaN values in both the values and the combined weights
        valid_data = df[[value_column]].copy()
        valid_data['combined_weight'] = combined_weight

        # Drop rows where either the value or the weight is NaN
        valid_data = valid_data.dropna(subset=[value_column, 'combined_weight'])

        if len(valid_data) > 0:
            # Calculate the weighted values and sum of combined weights for non-NaN entries
            weighted_values = valid_data[value_column] * valid_data['combined_weight']
            sum_combined_weights = valid_data['combined_weight'].sum()

            if sum_combined_weights > 0:
                weighted_avg = weighted_values.sum() / sum_combined_weights
            else:
                weighted_avg = np.nan  # Handle the case where sum of weights is zero
        else:
            weighted_avg = np.nan  # If no valid data, return NaN

        # Store the result for this variable
        results[value_column] = weighted_avg

    return results


# Load NetCDF data files using xarray
rad_ds = xr.open_dataset(r'/data_backup/share/ERA5_land/radiation/monthly/merged_annual.nc')
rad_ds['longitude'] = xr.where(rad_ds['longitude'] > 180, rad_ds['longitude'] - 360, rad_ds['longitude'])
rad_ds = rad_ds.sortby('longitude')

area_ds = xr.open_dataset(r'/data1/fyliu/a_temperature_range/data/area_mask/area_0.1deg.nc')

# Load CSV data
stmax_dir = r'/data1/fyliu/a_temperature_range/process_data/ridge_data/STmax_loc_frequency_1950_2023/'
stmin_dir = r'/data1/fyliu/a_temperature_range/process_data/ridge_data/STmin_loc_frequency_1950_2023/'
stmaxpath = os.listdir(stmax_dir)
stmaxpath.sort()
stminpath = os.listdir(stmin_dir)
stminpath.sort()

variables = ['str', 'strd', 'ssrd', 'slhf', 'sshf', 'fal']

filelen = len(stmaxpath)
stmax_outpath = f'/data1/fyliu/a_temperature_range/process_data/ridge_data/STmax_radiation'
stmin_outpath = f'/data1/fyliu/a_temperature_range/process_data/ridge_data/STmin_radiation'
for i in range(filelen):
    # use for fig content
    stmaxfile = stmaxpath[i]
    ff = stmaxfile.split('_')
    if ff[0] in ['Hong', 'Inner', 'north', 'south']:
        ff = ff[0:2]
        ff = '_'.join(ff)
    else:
        ff = ff[0]
    stminfile = stminpath[i]
    csv_stmax = pd.read_csv(os.path.join(stmax_dir, stmaxfile))
    csv_stmin = pd.read_csv(os.path.join(stmin_dir, stminfile))

    # Get filtered data for STmax and STmin based on CSV lat/lon
    stmax_data = get_nc_data_for_st(csv_stmax, rad_ds, area_ds, variables)
    stmin_data = get_nc_data_for_st(csv_stmin, rad_ds, area_ds, variables)

    # Calculate weighted averages for all variables
    variables_to_average = variables  # You can include area or other variables if needed
    stmax_averages = weighted_average(stmax_data, variables_to_average)
    stmin_averages = weighted_average(stmin_data, variables_to_average)

    # Combine the results into a summary DataFrame for both STmax and STmin
    summary_stmax = pd.DataFrame(stmax_averages)  # .T.reset_index().rename(columns={'index': 'Variable'})
    summary_stmax['str_earth'] = summary_stmax['str'] - summary_stmax['strd']
    summary_stmax['rad'] = summary_stmax['ssrd'] * (1 - summary_stmax['fal']) + summary_stmax['str'] + \
                           summary_stmax['slhf'] + summary_stmax['sshf']
    summary_stmax['year'] = np.arange(1950, 2024)

    summary_stmin = pd.DataFrame(stmin_averages)  # .T.reset_index().rename(columns={'index': 'Variable'})
    summary_stmin['str_earth'] = summary_stmin['str'] - summary_stmin['strd']
    summary_stmin['rad'] = summary_stmin['ssrd'] * (1 - summary_stmin['fal']) + summary_stmin['str'] + \
                           summary_stmin['slhf'] + summary_stmin['sshf']
    summary_stmin['year'] = np.arange(1950, 2024)

    # Save the summary data to CSV files
    stmax_outname = f'{ff}_radiation_stmax.csv'
    stmin_outname = f'{ff}_radiation_stmin.csv'
    summary_stmax.to_csv(combinename(stmax_outpath, stmax_outname), index=False)
    summary_stmin.to_csv(combinename(stmin_outpath, stmin_outname), index=False)

    print(f"{stmax_outname} have been saved to CSV files.")
    print(f"{stmin_outname} have been saved to CSV files.")









