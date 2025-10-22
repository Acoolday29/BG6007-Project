import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

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
