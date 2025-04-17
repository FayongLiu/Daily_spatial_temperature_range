# -*- coding: utf-8 -*-
import pandas as pd
import os

# this file is for counting the frequency of the STmax/STmin location
# i.e., for a given location, how many times they occcured duting a given period

# Folder paths for STmin and STmax data
min_folder = '/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_mindis/'
max_folder = '/data1/fyliu/a_temperature_range/process_data/DSTR_char/multiyear_maxdis/'

# Define the year range (1950-2023)
start_year = 1950
end_year = 2023

# Generate output folder paths dynamically based on the year range
output_folder_min = f'/data1/fyliu/a_temperature_range/process_data/ridge_data/STmin_loc_frequency_{start_year}_{end_year}/'
output_folder_max = f'/data1/fyliu/a_temperature_range/process_data/ridge_data/STmax_loc_frequency_{start_year}_{end_year}/'

# Create output folders if they don't exist
os.makedirs(output_folder_min, exist_ok=True)
os.makedirs(output_folder_max, exist_ok=True)

# Get all CSV files in each folder
min_files = [f for f in os.listdir(min_folder) if f.endswith('.csv')]
max_files = [f for f in os.listdir(max_folder) if f.endswith('.csv')]

# Function to calculate frequency of coordinates within each file for a given year range
def calculate_frequency(file_list, input_folder, output_folder, start_year, end_year):
    for file in file_list:
        # Read each CSV file
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path, index_col=0)

        # Ensure the 'year' column exists
        if 'year' not in df.columns:
            print(f"Skipping {file}: No 'year' column found.")
            continue

        # Filter data within the specified year range
        df_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

        # Group by lat and lon, and count occurrences
        coord_frequency = df_filtered.groupby(['lat', 'lon']).size().reset_index(name='count')

        # Modify output file name to include the year range
        output_file = os.path.join(
            output_folder,
            f"{file.replace('.csv', '')}_{start_year}_{end_year}_loc_frequency.csv"
        )

        # Save the frequency result to a new CSV file
        coord_frequency.to_csv(output_file, index=False)

        print(f"Processed {file} for years {start_year}-{end_year} and saved to {output_file}")

# Calculate and save coordinate frequency for STmin data within the year range
calculate_frequency(min_files, min_folder, output_folder_min, start_year, end_year)

# Calculate and save coordinate frequency for STmax data within the year range
calculate_frequency(max_files, max_folder, output_folder_max, start_year, end_year)

print("Processing completed.")
