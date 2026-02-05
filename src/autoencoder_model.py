"""
Fast Autoencoder using Sklearn - completes in minutes instead of hours
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import evaluate_model, plot_roc_pr_curves, find_optimal_threshold

def load_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    train = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'train_normal.csv'))
    val = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'val.csv'))
    test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'test.csv'))
    
    # Sample for faster training
    train = train.sample(n=min(50000, len(train)), random_state=config.RANDOM_STATE)
    print(f"Training samples: {len(train):,}")
    print(f"Test samples: {len(test):,}")
    
    return train, val, test

def prepare_features(df):
    """Prepare features"""
    X = df.drop(columns=['Time', 'Class'], errors='ignore').values
    y = df['Class'].values if 'Class' in df.columns else None
    return X, y

def train_autoencoder(X_train):
    """Train autoencoder using sklearn MLPRegressor"""
    print("\n" + "="*60)
    print("TRAINING AUTOENCODER (SKLEARN)")
    print("="*60)
    
    print(f"\nArchitecture: Input({X_train.shape[1]}) -> 32 -> 10 -> 32 -> Output({X_train.shape[1]})")
    print("This will take 2-3 minutes...\n")
    
    model = MLPRegressor(
        hidden_layer_sizes=(32, 10, 32),  # Encoder-Bottleneck-Decoder
        activation='relu',
        max_iter=50,
        random_state=config.RANDOM_STATE,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5
    )
    
    model.fit(X_train, X_train)  # Autoencoder: input = output
    print("\n✓ Training complete!")
    
    return model

def compute_reconstruction_error(model, X):
    """Compute reconstruction error as anomaly score"""
    X_pred = model.predict(X)
    mse = np.mean((X - X_pred) ** 2, axis=1)
    return mse

def main():
    """Main training and evaluation pipeline"""
    
    # Load data
    train, val, test = load_data()
    
    # Prepare features
    X_train, _ = prepare_features(train)
    X_test, y_test = prepare_features(test)
    
    print(f"\nFeature dimensions: {X_train.shape[1]}")
    
    # Train
    model = train_autoencoder(X_train)
    
    # Save model
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'autoencoder.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("COMPUTING RECONSTRUCTION ERRORS")
    print("="*60)
    
    reconstruction_errors = compute_reconstruction_error(model, X_test)
    
    # Find optimal threshold
    threshold = find_optimal_threshold(y_test, reconstruction_errors, metric='f1')
    
    # Make predictions
    y_pred = (reconstruction_errors > threshold).astype(int)
    
    # Evaluate
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    test_metrics = evaluate_model(
        y_test, y_pred, reconstruction_errors,
        "Autoencoder - Test"
    )
    
    # Plot curves
    plot_roc_pr_curves(
        y_test, reconstruction_errors,
        "Autoencoder",
        save_path="notebooks/autoencoder_curves.png"
    )
    
    # Save threshold
    joblib.dump(threshold, os.path.join(config.MODEL_SAVE_PATH, 'autoencoder_threshold.pkl'))
    
    print("\n" + "="*60)
    print("AUTOENCODER TRAINING COMPLETE!")
    print("="*60)
    
    return model, test_metrics, threshold

if __name__ == "__main__":
    model, metrics, threshold = main()
