"""
Data preprocessing and splitting for fraud detection
Handles time-based splitting to prevent data leakage
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import config

def load_data():
    """Load raw credit card fraud dataset"""
    print("Loading data...")
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Data loaded successfully: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    return df

def time_based_split(df, test_size=0.2, val_size=0.1):
    """
    Split data based on time to prevent data leakage
    In production, you don't have future fraud labels
    
    Returns:
        train_df, val_df, test_df
    """
    # Sort by time
    df = df.sort_values('Time').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\nTime-based split:")
    print(f"Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    print(f"Val:   {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    print(f"Test:  {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
    
    print(f"\nFraud distribution:")
    print(f"Train: {train_df['Class'].sum()} frauds ({train_df['Class'].mean()*100:.3f}%)")
    print(f"Val:   {val_df['Class'].sum()} frauds ({val_df['Class'].mean()*100:.3f}%)")
    print(f"Test:  {test_df['Class'].sum()} frauds ({test_df['Class'].mean()*100:.3f}%)")
    
    return train_df, val_df, test_df

def get_normal_transactions(df):
    """
    Extract only normal transactions for unsupervised training
    This simulates real-world scenario where we train without fraud labels
    """
    normal_df = df[df['Class'] == 0].copy()
    print(f"\nExtracted {len(normal_df)} normal transactions for training")
    return normal_df

def scale_features(train_df, val_df, test_df, features_to_scale):
    """
    Scale features using StandardScaler
    Fit only on training data to prevent data leakage
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    scaler.fit(train_df[features_to_scale])
    
    # Transform all sets
    train_df[features_to_scale] = scaler.transform(train_df[features_to_scale])
    val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])
    test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
    
    # Save scaler
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    joblib.dump(scaler, os.path.join(config.MODEL_SAVE_PATH, 'scaler.pkl'))
    print(f"\nScaler saved to {config.MODEL_SAVE_PATH}scaler.pkl")
    
    return train_df, val_df, test_df, scaler

def prepare_data_for_training():
    """
    Main pipeline: load, split, extract normal, and scale
    """
    # Load data
    df = load_data()
    
    # Time-based split
    train_df, val_df, test_df = time_based_split(
        df, 
        test_size=config.TEST_SIZE, 
        val_size=config.VAL_SIZE
    )
    
    # Extract normal transactions from training set for unsupervised learning
    train_normal = get_normal_transactions(train_df)
    
    # Features to scale (Amount needs scaling, V1-V28 are already scaled from PCA)
    features_to_scale = ['Amount']
    
    # Scale features
    train_normal_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_normal, val_df, test_df, features_to_scale
    )
    
    # Save processed data
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    
    train_normal_scaled.to_csv(
        os.path.join(config.PROCESSED_DATA_PATH, 'train_normal.csv'), 
        index=False
    )
    val_scaled.to_csv(
        os.path.join(config.PROCESSED_DATA_PATH, 'val.csv'), 
        index=False
    )
    test_scaled.to_csv(
        os.path.join(config.PROCESSED_DATA_PATH, 'test.csv'), 
        index=False
    )
    
    print(f"\nProcessed data saved to {config.PROCESSED_DATA_PATH}")
    
    return train_normal_scaled, val_scaled, test_scaled

if __name__ == "__main__":
    # Run preprocessing pipeline
    train_normal, val, test = prepare_data_for_training()
    
    print("\n" + "="*50)
    print("Data preprocessing complete!")
    print("="*50)
    print(f"\nTrain (normal only): {train_normal.shape}")
    print(f"Validation: {val.shape}")
    print(f"Test: {test.shape}")
