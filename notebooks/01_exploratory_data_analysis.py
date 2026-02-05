"""
Exploratory Data Analysis for Fraud Detection
Run this to understand the dataset characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
import config

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_explore():
    """Load data and show basic statistics"""
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    print("\n1. DATASET OVERVIEW")
    print("-" * 60)
    print(f"Total transactions: {len(df):,}")
    print(f"Features: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n2. CLASS DISTRIBUTION")
    print("-" * 60)
    class_counts = df['Class'].value_counts()
    print(f"Normal transactions: {class_counts[0]:,} ({class_counts[0]/len(df)*100:.2f}%)")
    print(f"Fraudulent transactions: {class_counts[1]:,} ({class_counts[1]/len(df)*100:.2f}%)")
    print(f"Imbalance ratio: 1:{class_counts[0]//class_counts[1]}")
    
    print("\n3. TIME ANALYSIS")
    print("-" * 60)
    print(f"Time range: {df['Time'].min():.0f}s to {df['Time'].max():.0f}s")
    print(f"Duration: {(df['Time'].max() - df['Time'].min())/3600:.1f} hours")
    
    print("\n4. AMOUNT STATISTICS")
    print("-" * 60)
    print(df.groupby('Class')['Amount'].describe())
    
    print("\n5. MISSING VALUES")
    print("-" * 60)
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    
    print("\n6. FEATURE CORRELATIONS WITH FRAUD")
    print("-" * 60)
    correlations = df.corr()['Class'].sort_values(ascending=False)
    print("\nTop 5 positive correlations:")
    print(correlations.head(6)[1:])  # Exclude Class itself
    print("\nTop 5 negative correlations:")
    print(correlations.tail(5))
    
    return df

def visualize_distributions(df):
    """Create visualization of key distributions"""
    print("\n7. GENERATING VISUALIZATIONS")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Class distribution
    ax1 = axes[0, 0]
    class_counts = df['Class'].value_counts()
    ax1.bar(['Normal', 'Fraud'], class_counts.values, color=['green', 'red'])
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.set_yscale('log')
    for i, v in enumerate(class_counts.values):
        ax1.text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # 2. Amount distribution by class
    ax2 = axes[0, 1]
    normal_amounts = df[df['Class'] == 0]['Amount']
    fraud_amounts = df[df['Class'] == 1]['Amount']
    ax2.hist([normal_amounts, fraud_amounts], bins=50, label=['Normal', 'Fraud'], 
             color=['green', 'red'], alpha=0.7)
    ax2.set_xlabel('Transaction Amount')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Transaction Amount Distribution')
    ax2.legend()
    ax2.set_xlim(0, 500)  # Focus on typical range
    
    # 3. Time distribution of fraud
    ax3 = axes[1, 0]
    fraud_time = df[df['Class'] == 1]['Time'] / 3600  # Convert to hours
    ax3.hist(fraud_time, bins=48, color='red', alpha=0.7)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Fraud Count')
    ax3.set_title('Fraud Distribution Over Time')
    
    # 4. Feature correlation heatmap (top features)
    ax4 = axes[1, 1]
    top_features = df.corr()['Class'].abs().sort_values(ascending=False).head(11).index
    corr_matrix = df[top_features].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Feature Correlation Matrix (Top Features)')
    
    plt.tight_layout()
    plt.savefig('notebooks/eda_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: notebooks/eda_visualization.png")
    plt.show()

def analyze_pca_features(df):
    """Analyze PCA-transformed features"""
    print("\n8. PCA FEATURES ANALYSIS")
    print("-" * 60)
    
    v_features = [f'V{i}' for i in range(1, 29)]
    
    # Compare normal vs fraud for each V feature
    print("\nFeatures with largest mean difference (Normal vs Fraud):")
    differences = {}
    for feature in v_features:
        normal_mean = df[df['Class'] == 0][feature].mean()
        fraud_mean = df[df['Class'] == 1][feature].mean()
        differences[feature] = abs(fraud_mean - normal_mean)
    
    top_diff = sorted(differences.items(), key=lambda x: x[1], reverse=True)[:5]
    for feature, diff in top_diff:
        print(f"{feature}: {diff:.4f}")

if __name__ == "__main__":
    # Run EDA
    df = load_and_explore()
    visualize_distributions(df)
    analyze_pca_features(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print("\nKey Insights:")
    print("1. Highly imbalanced dataset (0.17% fraud)")
    print("2. PCA features V1-V28 already normalized")
    print("3. Amount feature needs scaling")
    print("4. No missing values")
    print("5. Time-based patterns exist in fraud distribution")
