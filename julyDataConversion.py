import os
import pandas as pd
import numpy as np

# List of features to drop (with zero importance)
features_to_drop = [
    'FWI', 'VPD', 'HT', 'T', 'FWI_1_days_ago', 'VPD_1_days_ago', 'P_1_days_ago', 
    'FWI_2_days_ago', 'VPD_2_days_ago', 'P_2_days_ago', 
    'FWI_3_days_ago', 'VPD_3_days_ago', 'P_3_days_ago', 
    'FWI_4_days_ago', 'VPD_4_days_ago', 'P_4_days_ago', 
    'FWI_5_days_ago', 'VPD_5_days_ago', 'P_5_days_ago', 
    'FWI_6_days_ago', 'VPD_6_days_ago', 'P_6_days_ago', 
    'FWI_7_days_ago', 'VPD_7_days_ago', 'P_7_days_ago',
    'Nearest_3', 'Nearest_4', 'Nearest_5'
]

# Directory containing the CSV files
input_dir = './data/'
output_dir = './processed_data/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file
for day in range(1, 32):
    file_name = f'202107{day:02d}.csv'
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(input_path):
        # Load the data
        data = pd.read_csv(input_path)

        # Drop the zero importance columns
        data_dropped = data.drop(columns=features_to_drop, errors='ignore')

        # Drop rows where any FRP_X_days_ago columns have a value of -999.0
        frp_columns = [f'FRP_{i}_days_ago' for i in range(1, 8)]
        data_dropped = data_dropped[~data_dropped[frp_columns].isin([-999.0]).any(axis=1)]

        # Convert the specified FRP columns to their logarithmic form using np.log10(column + 1e-2)
        for col_name in frp_columns:
            if col_name in data_dropped.columns:
                data_dropped[col_name] = np.log10(data_dropped[col_name] + 1e-2)

        # Save the processed data to a new file
        data_dropped.to_csv(output_path, index=False)
        print(f"Processed file saved to {output_path}")

print("Processing complete.")