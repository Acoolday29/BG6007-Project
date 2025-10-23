import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load cleaned dataset
def load_cleaned_data():
  df = pd.read_csv('cleaned_data.csv')

  return df

# Define feature as RGB, labels as SpO2
def features_and_labels(df):
  features = df[['R', 'G', 'B']].values
  labels = df['corresponding_SpO2'].values

  return features, labels

# Train random forest model
def train_RF(features, labels):
  # Divide 80% training group and 20% testing group
  x_train, x_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=0.2, 
        random_state=46)
  
  # Create RF model
  rf_model = RandomForestRegressor(
    n_estimators = 100,
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    random_state = 46,
    n_jobs = 1)

  # Train RF model
  rf_model.fit(x_train, y_train)

  # Train/test predictions
  y_train_pred = rf_model.predict(x_train)
  y_test_pred = rf_model.predict(x_test)

  # Evaluate model by MAE, MSE and R^2
  train_mae = mean_absolute_error(y_train, y_train_pred)
  test_mae = mean_absolute_error(y_test, y_test_pred)
  
  train_mse = mean_squared_error(y_train, y_train_pred)
  test_mse = mean_squared_error(y_test, y_test_pred)
  
  train_r2 = r2_score(y_train, y_train_pred)
  test_r2 = r2_score(y_test, y_test_pred)

  # Check for overfitting
  overfit_gap = train_mae - test_mae
  if overfit_gap < -0.5:
    print("Overfitting may exist")
  else:
    print("Pass overfitting test")

  print(f"training set - MAE: {train_mae:.3f}, MSE: {train_mse:.3f}, R²: {train_r2:.3f}")
  print(f"testing set - MAE: {test_mae:.3f}, MSE: {test_mse:.3f}, R²: {test_r2:.3f}")
  return rf_model, x_train, x_test, y_train, y_test, y_train_pred, y_test_pred

if __name__ == "__main__":
  df = load_cleaned_data()
  features, labels = features_and_labels(df)
  rf_model, x_train, x_test, y_train, y_test, y_train_pred, y_test_pred = train_RF(features, labels)


  
