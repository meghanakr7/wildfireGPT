import pandas as pd
import os

# Directory paths
data_dir = './processed_data'
predictions_dir = './predictions'
output_dir = './comparison_results'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Date range
start_date = pd.to_datetime('2021-07-01')
end_date = pd.to_datetime('2021-07-30')

# Loop over each date in the range
for date in pd.date_range(start_date, end_date):
    prediction_file = os.path.join(predictions_dir, date.strftime('%Y%m%d') + '.csv')
    data_file = os.path.join(data_dir, (date + pd.Timedelta(days=1)).strftime('%Y%m%d') + '.csv')
    
    # Check if both files exist
    if os.path.exists(prediction_file) and os.path.exists(data_file):
        # Load the prediction and data files
        pred_df = pd.read_csv(prediction_file)
        data_df = pd.read_csv(data_file)
        
        # Extract Predicted_FRP and FRP_1_days_ago
        predicted_frp = pred_df['Predicted_FRP']
        actual_frp_1_day_ago = data_df['FRP_1_days_ago']
        
        # Create a DataFrame for the comparison
        comparison = pd.DataFrame({
            'Predicted_FRP': predicted_frp,
            'FRP_1_days_ago': actual_frp_1_day_ago
        })
        
        # Save the comparison to a new file
        output_file = os.path.join(output_dir, f'comparison_{date.strftime("%Y%m%d")}.csv')
        comparison.to_csv(output_file, index=False)
        
        print(f'Comparison for {date.strftime("%Y-%m-%d")} saved to {output_file}')

print('All comparisons have been processed and saved.')
