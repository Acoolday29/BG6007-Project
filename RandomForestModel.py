import pandas as pd
import numpy as np

def load_cleaned_data():
  print("This is data after cleaning:")
  df = pd.read_csv('cleaned_data.csv')

  return df

def features_and_labels(df):
  features = df[['R', 'G', 'B']].values
