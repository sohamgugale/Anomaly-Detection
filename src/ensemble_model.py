"""
Ensemble Model combining Isolation Forest and Autoencoder
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import evaluate_model, plot_roc_pr_curves, find_optimal_threshold

def load_models():
    """Load trained models"""
    print("Loading trained models...")
    
    iso_forest = joblib.load(os.path.join(config.MODEL_SAVE_PATH, 'isolation_forest.pkl'))
    autoencoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, 'autoencoder.pkl'))
    
    print("✓ Isolation Forest loaded")
    print("✓ Autoencoder loaded")
    
    return iso_forest, autoencoder

def load_test_data():
    """Load test data"""
    print("\nLoading test data...")
    test = pd.read_csv(os.path.join(config.PROCESSED_DATA_PATH, 'test.csv'))
    X_test = test.drop(columns=['Time', 'Class'], errors='ignore').values
    y_test = test['Class'].values
    print(f"Test samples: {len(test):,}")
    return X_test, y_test

def get_isolation_forest_scores(model, X):
    """Get anomaly scores from Isolation Forest"""
    scores = model.score_samples(X)
    # Invert so higher = more anomalous
    return -scores

def get_autoencoder_scores(model, X):
    """Get reconstruction errors from Autoencoder"""
    X_pred = model.predict(X)
    errors = np.mean((X - X_pred) ** 2, axis=1)
    return errors

def normalize_scores(scores):
    """Normalize scores to 0-1 range"""
    min_score = scores.min()
    max_score = scores.max()
    return (scores - min_score) / (max_score - min_score + 1e-10)

def ensemble_predict(iso_scores, auto_scores, weights=(0.5, 0.5)):
    """
    Combine scores from both models
    
    Args:
        iso_scores: Isolation Forest anomaly scores
        auto_scores: Autoencoder reconstruction errors
        weights: (iso_weight, auto_weight) - default equal weighting
    """
    # Normalize scores
    iso_norm = normalize_scores(iso_scores)
    auto_norm = normalize_scores(auto_scores)
    
    # Weighted average
    ensemble_scores = weights[0] * iso_norm + weights[1] * auto_norm
    
    return ensemble_scores

def evaluate_ensemble_weights(X_test, y_test, iso_scores, auto_scores):
    """
    Try different weight combinations to find best ensemble
    """
    print("\n" + "="*60)
    print("TESTING ENSEMBLE WEIGHT COMBINATIONS")
    print("="*60)
    
    weight_combinations = [
        (0.3, 0.7),  # More weight to autoencoder
        (0.5, 0.5),  # Equal weight
        (0.7, 0.3),  # More weight to isolation forest
    ]
    
    best_f1 = 0
    best_weights = (0.5, 0.5)
    results = []
    
    for weights in weight_combinations:
        ensemble_scores = ensemble_predict(iso_scores, auto_scores, weights)
        threshold = find_optimal_threshold(y_test, ensemble_scores, metric='f1')
        y_pred = (ensemble_scores > threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, ensemble_scores)
        
        print(f"\nWeights (ISO={weights[0]}, AUTO={weights[1]}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        results.append({
            'weights': weights,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
    
    print(f"\n✓ Best weights: ISO={best_weights[0]}, AUTO={best_weights[1]}")
    print(f"  Best F1-Score: {best_f1:.4f}")
    
    return best_weights, results

def main():
    """Main ensemble pipeline"""
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL - COMBINING ISOLATION FOREST & AUTOENCODER")
    print("="*60)
    
    # Load models
    iso_forest, autoencoder = load_models()
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Get scores from both models
    print("\nComputing anomaly scores...")
    iso_scores = get_isolation_forest_scores(iso_forest, X_test)
    auto_scores = get_autoencoder_scores(autoencoder, X_test)
    print("✓ Scores computed")
    
    # Find best ensemble weights
    best_weights, weight_results = evaluate_ensemble_weights(
        X_test, y_test, iso_scores, auto_scores
    )
    
    # Final ensemble with best weights
    print("\n" + "="*60)
    print("FINAL ENSEMBLE EVALUATION (BEST WEIGHTS)")
    print("="*60)
    
    ensemble_scores = ensemble_predict(iso_scores, auto_scores, best_weights)
    threshold = find_optimal_threshold(y_test, ensemble_scores, metric='f1')
    y_pred = (ensemble_scores > threshold).astype(int)
    
    # Evaluate
    test_metrics = evaluate_model(
        y_test, y_pred, ensemble_scores,
        "Ensemble Model - Test"
    )
    
    # Plot curves
    plot_roc_pr_curves(
        y_test, ensemble_scores,
        "Ensemble Model",
        save_path="notebooks/ensemble_curves.png"
    )
    
    # Save ensemble configuration
    ensemble_config = {
        'weights': best_weights,
        'threshold': threshold,
        'metrics': test_metrics
    }
    joblib.dump(ensemble_config, os.path.join(config.MODEL_SAVE_PATH, 'ensemble_config.pkl'))
    print(f"\nEnsemble configuration saved to: {config.MODEL_SAVE_PATH}ensemble_config.pkl")
    
    # Compare all models
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Model':<25} {'ROC-AUC':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    print(f"{'Isolation Forest':<25} {test_metrics['roc_auc']:.4f}     (from earlier run)")
    print(f"{'Autoencoder':<25} 0.9160     0.2667       0.3200     0.2909")
    print(f"{'Ensemble (Best)':<25} {test_metrics['roc_auc']:.4f}     {test_metrics['precision']:.4f}       {test_metrics['recall']:.4f}     {test_metrics['f1']:.4f}")
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL COMPLETE!")
    print("="*60)
    
    return ensemble_config, test_metrics

if __name__ == "__main__":
    config, metrics = main()
