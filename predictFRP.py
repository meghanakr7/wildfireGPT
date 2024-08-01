import os
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor

# Load the trained model
model_save_path = 'trained_tabnet_model_fold_3.zip'
tabnet_model = TabNetRegressor()
tabnet_model.load_model(model_save_path)

# Directory containing the processed CSV files
input_dir = './processed_data/'
output_dir = './predictions/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file
for day in range(1, 32):
    file_name = f'202107{day:02d}.csv'
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(input_path):
        # Load the processed data
        data = pd.read_csv(input_path)

        # Separate features
        X_new = data.drop(columns=['FRP'], errors='ignore')

        # Make predictions
        predictions = tabnet_model.predict(X_new.values)

        # Add the predictions to the dataframe
        data['Predicted_FRP'] = predictions

        # Save the predictions to a new file
        data.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

print("Prediction process complete.")