"""Data preprocessing and splitting for fraud detection"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config

def load_data():
    """Load raw data"""
    print("Loading data...")
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Data loaded: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess and split data"""
    # TODO: Implement preprocessing
    pass

if __name__ == "__main__":
    df = load_data()
    print(df.head())
