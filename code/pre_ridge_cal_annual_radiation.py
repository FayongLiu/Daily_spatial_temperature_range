import os
import xarray as xr
import matplotlib.pyplot as plt

# this file is for merging monthly radiation data to annual scale

# See this part for merging data
# ================================================================================================
# ================================================================================================
# Define file paths
albedo_file = "/data_backup/share/ERA5_land/radiation/monthly/global_albedo_01deg_1950_2023.nc"
radiation_files = [
    "/data_backup/share/ERA5_land/radiation/monthly/global_radiation_01deg_1950_2001.nc",
    "/data_backup/share/ERA5_land/radiation/monthly/global_radiation_01deg_2002_2023.nc"
]

# Output file path
output_file = "/data_backup/share/ERA5_land/radiation/monthly/merged_annual.nc"
# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Function to compute the annual sum for radiation variables
def process_radiation_file(file_path):
    ds = xr.open_dataset(file_path)
    # Compute annual sums for all variables in the dataset
    annual_sum = ds.resample(valid_time='1YE').sum(skipna=True)
    ds.close()  # Close the dataset to free memory
    return annual_sum

# Compute annual sums for each radiation file and merge them
annual_radiation_datasets = [process_radiation_file(f) for f in radiation_files]
merged_radiation = xr.concat(annual_radiation_datasets, dim='valid_time')

# Open the albedo dataset and compute the annual mean
albedo_ds = xr.open_dataset(albedo_file)
annual_albedo = albedo_ds.resample(valid_time='1YE').mean(skipna=True)
albedo_ds.close()  # Close to free memory

# Align radiation and albedo datasets along the time axis
aligned_radiation = merged_radiation.reindex(valid_time=annual_albedo.valid_time, method='nearest')

# Merge the processed albedo and radiation datasets
final_dataset = xr.merge([annual_albedo, aligned_radiation], compat='override')

# Save the merged dataset to a new NetCDF file
final_dataset.to_netcdf(output_file)

print(f"Merged dataset saved to: {output_file}")

