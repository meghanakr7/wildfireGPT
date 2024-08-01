import pandas as pd
import numpy as np

# Step 1: Read the input CSV file
input_file = 'wildfire.csv'
print("Reading input file...")
df = pd.read_csv(input_file)
print("Input file read successfully.")

# Step 2: Convert the FRP values to their log values
print("Converting FRP values to log values...")
for i in range(1, 8):
    column_name = f'FRP_{i}_days_ago'
    log_column_name = f'log_{column_name}'
    df[log_column_name] = np.log10(df[column_name] + 1e-2)
    print(f"Converted {column_name} to {log_column_name}")

# Step 3: Find the minimum and maximum FRP values for each day
min_max_frp_values = {}
print("Calculating minimum and maximum FRP values for each day...")
for i in range(1, 8):
    column_name = f'FRP_{i}_days_ago'
    min_frp = df[column_name].min()
    max_frp = df[column_name].max()
    min_max_frp_values[column_name] = (min_frp, max_frp)
    print(f"Minimum {column_name}: {min_frp}")
    print(f"Maximum {column_name}: {max_frp}")

# Step 4: Save the converted columns along with all other data to a new CSV file
output_file = 'wildfire_converted.csv'
print("Saving converted data to output file...")
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")

# Print the minimum and maximum FRP values for confirmation
print("Summary of minimum and maximum FRP values for each day:")
for column_name, (min_frp, max_frp) in min_max_frp_values.items():
    print(f"{column_name} - Minimum: {min_frp}, Maximum: {max_frp}")
