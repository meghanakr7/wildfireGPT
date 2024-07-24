import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

def train_model():
    # Load the dataset
    dataset = pd.read_csv('5000.csv')

    # Check if the target column should be removed
    remove_target_column = True  # Set this based on your requirements

    if remove_target_column:
        X = dataset.drop(columns=['FRP'])
        y = dataset['FRP']
    else:
        X = dataset.copy()
        y = None  # If target is not removed, adjust as necessary

    # Assuming 'best_tabnet_model.zip' contains a TabNet model
    model = TabNetRegressor()
    model.load_model('best_tabnet_model.zip')

    # Convert to numpy arrays for training
    X_train = X.values
    if y is not None:
        y_train = y.values.reshape(-1, 1)

    # Train the model
    if y is not None:
        model.fit(
            X_train, y_train,
            max_epochs=50, patience=10,
            batch_size=1024, virtual_batch_size=128,
            num_workers=1, drop_last=False
        )

    # Save the trained model
    model.save_model('trained_tabnet_model')

if __name__ == '__main__':
    train_model()