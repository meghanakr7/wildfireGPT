import os
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Load the data
data_path = './wildfire_filtered.csv'
data = pd.read_csv(data_path)

# Separate features and target
X = data.drop(columns=['FRP']).values
y = data['FRP'].values.reshape(-1, 1)

# Define TabNet model with better hyperparameters
tabnet_model = TabNetRegressor(
    n_d=64, # Width of the decision prediction layer
    n_a=64, # Width of the attention embedding for each mask
    n_steps=5, # Number of steps in the architecture
    gamma=1.5, # Relaxation parameter
    lambda_sparse=1e-3, # Sparsity regularization
    optimizer_fn=torch.optim.Adam, # Optimizer
    optimizer_params=dict(lr=2e-2), # Optimizer parameters
    mask_type='sparsemax' # Masking function to use (sparsemax or entmax)
)

# Implement k-fold cross-validation
kf = KFold(n_splits=5)
fold = 1
mse_list = []
mae_list = []
r2_list = []
mape_list = []

# Measure time for a single fold
start_time = time.time()

for train_index, valid_index in kf.split(X):
    print(f"Training fold {fold}...")
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    # Train the model
    tabnet_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Save the model for each fold
    model_save_path = f'trained_tabnet_model_fold_{fold}'
    tabnet_model.save_model(model_save_path)
    print(f"Trained model for fold {fold} saved to {model_save_path}")
    
    # Predict and evaluate
    y_pred = tabnet_model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    mape = (abs((y_valid - y_pred) / y_valid)).mean() * 100
    
    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)
    mape_list.append(mape)
    
    print(f"Fold {fold} - MSE: {mse}, MAE: {mae}, R²: {r2}, MAPE: {mape}%")
    fold += 1

# Measure end time
end_time = time.time()
fold_time = end_time - start_time
print(f"Time taken for one fold: {fold_time} seconds")

# Calculate mean metrics across all folds
mean_mse = np.mean(mse_list)
mean_mae = np.mean(mae_list)
mean_r2 = np.mean(r2_list)
mean_mape = np.mean(mape_list)

print(f"Mean MSE: {mean_mse}")
print(f"Mean MAE: {mean_mae}")
print(f"Mean R²: {mean_r2}")
print(f"Mean MAPE: {mean_mape}%")

# Estimate total time for k folds
k = 5
estimated_total_time = fold_time * k
print(f"Estimated total time for {k}-fold cross-validation: {estimated_total_time} seconds")
