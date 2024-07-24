import pandas as pd
import requests
from io import StringIO

# Load the data from the provided URL
url = "https://geobrain.csiss.gmu.edu/wildfire_site/data/20210701/firedata_20210701_predicted.txt"
response = requests.get(url, verify=False)  # Disable SSL verification

# Check if the request was successful
if response.status_code == 200:
    data = StringIO(response.text)
    # Correctly specify the delimiter as a comma
    df = pd.read_csv(data, delimiter=',')

    # Print the column names to identify the correct column name
    print("Column names:", df.columns)

    # Remove the column, assuming the exact name from the printed list
    column_to_remove = 'Predicted_FRP'  # Adjust this if the column name differs
    if column_to_remove in df.columns:
        df.drop(columns=[column_to_remove], inplace=True)
        print(f"Column '{column_to_remove}' removed.")
    else:
        print(f"Column '{column_to_remove}' not found.")

    # Save the data to a CSV file
    output_filename = "2021_July_1.csv"
    df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")
else:
    print("Failed to download the file")