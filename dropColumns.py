import pandas as pd

# Load the wildfire_converted.csv file
data_path = './wildfire_converted.csv'
data = pd.read_csv(data_path)

# List of features to drop (with zero importance and the specified FRP columns)
features_to_drop = [
    'FWI', 'VPD', 'HT', 'T', 'FWI_1_days_ago', 'VPD_1_days_ago', 'P_1_days_ago', 
    'FWI_2_days_ago', 'VPD_2_days_ago', 'P_2_days_ago', 
    'FWI_3_days_ago', 'VPD_3_days_ago', 'P_3_days_ago', 
    'FWI_4_days_ago', 'VPD_4_days_ago', 'P_4_days_ago', 
    'FWI_5_days_ago', 'VPD_5_days_ago', 'P_5_days_ago', 
    'FWI_6_days_ago', 'VPD_6_days_ago', 'P_6_days_ago', 
    'FWI_7_days_ago', 'VPD_7_days_ago', 'P_7_days_ago',
    'Nearest_3', 'Nearest_4', 'Nearest_5',
    'FRP_1_days_ago', 'FRP_2_days_ago', 'FRP_3_days_ago', 'FRP_4_days_ago', 'FRP_5_days_ago', 'FRP_6_days_ago', 'FRP_7_days_ago'
]

# Drop the zero importance columns
data_dropped = data.drop(columns=features_to_drop)

# Rename the log_FRP_X_days_ago columns to FRP_X_days_ago
columns_to_rename = {
    'log_FRP_1_days_ago': 'FRP_1_days_ago',
    'log_FRP_2_days_ago': 'FRP_2_days_ago',
    'log_FRP_3_days_ago': 'FRP_3_days_ago',
    'log_FRP_4_days_ago': 'FRP_4_days_ago',
    'log_FRP_5_days_ago': 'FRP_5_days_ago',
    'log_FRP_6_days_ago': 'FRP_6_days_ago',
    'log_FRP_7_days_ago': 'FRP_7_days_ago'
}

data_dropped = data_dropped.rename(columns=columns_to_rename)

# Save the new dataframe to another file
output_path = 'wildfire_filtered.csv'
data_dropped.to_csv(output_path, index=False)

print(f"Filtered data saved to {output_path}")