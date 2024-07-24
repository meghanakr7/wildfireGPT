import os
import pandas as pd

# Define the path to the data folder
data_folder = './data'

# Define the date range
start_date = '20210701'
end_date = '20210730'

# List all files in the data folder
files = os.listdir(data_folder)

# Function to check if a filename is within the specified date range
def is_within_date_range(filename, start_date, end_date):
    if not filename.endswith('.csv'):
        return False
    file_date = filename.split('.')[0]
    return start_date <= file_date <= end_date

# Process each file in the date range
for file in files:
    if is_within_date_range(file, start_date, end_date):
        file_path = os.path.join(data_folder, file)
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Drop the LAT and LON columns if they exist
        df = df.drop(columns=['LAT', 'LON'], errors='ignore')
        # Save the modified dataframe back to the CSV file
        df.to_csv(file_path, index=False)
        print(f'Processed {file}')

print('All specified files have been processed.')