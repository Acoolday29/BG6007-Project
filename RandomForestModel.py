import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load cleaned dataset
def load_cleaned_data():
  print("This is data after cleaning:")
  df = pd.read_csv('cleaned_data.csv')

  return df

# Define feature as RGB, labels as SpO2
def features_and_labels(df):
  features = df[['R', 'G', 'B']].values
  labels = df['corresponding_SpO2'].values

  return features, labels

# Train random forest model
def train_RF(x_train, y_train, x_test, y_test):
  # Create RF model
  rf_model = RandomForestRegressor(
    n_estimator = 100,
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
  
