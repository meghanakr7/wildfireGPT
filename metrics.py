import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

# Define the folder path
folder_path = '/Users/meghana/Documents/projects/AutoKeras/comparison_results/'

# Initialize lists to store metrics and a list to store all FRP_1_days_ago values
mse_list = []
mae_list = []
r2_list = []
mape_list = []
all_frp_values = []

# Initialize lists to store combined predicted and actual values
all_y_pred = []
all_y_true = []

# Loop through each file and calculate metrics
for day in range(1, 32):
    file_path = os.path.join(folder_path, f'comparison_202107{day:02d}.csv')
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        
        # Extract predicted and actual values
        y_pred = data['Predicted_FRP'].values
        y_true = data['FRP_1_days_ago'].values
        
        # Check for NaN values and handle them
        if np.isnan(y_pred).any() or np.isnan(y_true).any():
            print(f"NaN values found in {file_path}, skipping this file.")
            continue
        
        # Append actual FRP values to the list
        all_frp_values.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_true.extend(y_true)
        
        # Normalize the values
        scaler = StandardScaler()
        y_true = scaler.fit_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = (np.abs((y_true - y_pred) / y_true)).mean() * 100
        
        # Append metrics to the lists
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)
        mape_list.append(mape)
    else:
        print(f"{file_path} does not exist.")

# Calculate mean metrics across all files
mean_mse = np.mean(mse_list)
mean_mae = np.mean(mae_list)
mean_r2 = np.mean(r2_list)
mean_mape = np.mean(mape_list)

# Calculate min and max of FRP_1_days_ago
min_frp = np.min(all_frp_values)
max_frp = np.max(all_frp_values)

# Print mean metrics
print("\nFinal Metrics Across All Days in July:")
print(f"Mean MSE: {mean_mse}")
print(f"Mean MAE: {mean_mae}")
print(f"Mean R²: {mean_r2}")
print(f"Mean MAPE: {mean_mape}%")

# # Print min and max of FRP_1_days_ago
print(f"\nMin FRP_1_days_ago: {min_frp}")
print(f"Max FRP_1_days_ago: {max_frp}")

# Polynomial transformation and Linear Regression
poly = PolynomialFeatures(degree=2)
all_y_true_poly = poly.fit_transform(np.array(all_y_true).reshape(-1, 1))

# Fit a linear regression model on polynomial features
model = LinearRegression()
model.fit(all_y_true_poly, all_y_pred)
y_pred_poly = model.predict(all_y_true_poly)

# Calculate final metrics with polynomial features
final_mse = mean_squared_error(all_y_true, y_pred_poly)
final_mae = mean_absolute_error(all_y_true, y_pred_poly)
final_r2 = r2_score(all_y_true, y_pred_poly)
final_mape = (np.abs((np.array(all_y_true) - y_pred_poly.flatten()) / np.array(all_y_true))).mean() * 100

# Print final metrics with polynomial features
print("\nFinal Metrics with Polynomial Features Across All Days in July:")
print(f"Final MSE: {final_mse}")
print(f"Final MAE: {final_mae}")
print(f"Final R²: {final_r2}")
print(f"Final MAPE: {final_mape}%")
