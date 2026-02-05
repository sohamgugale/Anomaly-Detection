"""
Real-Time Fraud Detection System - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    iso_forest = joblib.load('models/isolation_forest.pkl')
    autoencoder = joblib.load('models/autoencoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    ensemble_config = joblib.load('models/ensemble_config.pkl')
    
    return iso_forest, autoencoder, scaler, ensemble_config

def normalize_scores(scores):
    """Normalize scores to 0-1 range"""
    min_score = scores.min()
    max_score = scores.max()
    return (scores - min_score) / (max_score - min_score + 1e-10)

def predict_fraud(transaction, iso_forest, autoencoder, ensemble_config):
    """
    Predict if transaction is fraudulent
    
    Returns:
        is_fraud: Boolean
        fraud_probability: Float 0-1
        confidence: String (Low/Medium/High)
    """
    # Get anomaly scores
    iso_score = -iso_forest.score_samples(transaction.reshape(1, -1))[0]
    
    X_pred = autoencoder.predict(transaction.reshape(1, -1))
    auto_score = np.mean((transaction - X_pred) ** 2)
    
    # Normalize and ensemble
    iso_norm = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-10)
    auto_norm = (auto_score - auto_score.min()) / (auto_score.max() - auto_score.min() + 1e-10)
    
    weights = ensemble_config['weights']
    ensemble_score = weights[0] * iso_norm + weights[1] * auto_norm
    
    threshold = ensemble_config['threshold']
    is_fraud = ensemble_score > threshold
    
    # Fraud probability (normalized score)
    fraud_prob = float(ensemble_score)
    
    # Confidence level
    if fraud_prob < 0.3:
        confidence = "Low"
    elif fraud_prob < 0.6:
        confidence = "Medium"
    else:
        confidence = "High"
    
    return is_fraud, fraud_prob, confidence, {
        'isolation_forest': float(iso_norm),
        'autoencoder': float(auto_norm),
        'ensemble': float(ensemble_score)
    }

# Title and description
st.title("🔍 Real-Time Fraud Detection System")
st.markdown("""
**Unsupervised Anomaly Detection** using Ensemble Methods (Isolation Forest + Autoencoder)

Upload a transaction or enter features manually to detect potential fraud.
""")

# Load models
try:
    iso_forest, autoencoder, scaler, ensemble_config = load_models()
    st.success("✓ Models loaded successfully")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("ROC-AUC Score", "0.947")
    st.metric("Fraud Detection Rate", "43%")
    st.metric("False Positive Rate", "0.58%")
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    This system uses:
    - **Isolation Forest**: Tree-based anomaly detection
    - **Autoencoder**: Neural network reconstruction error
    - **Ensemble**: Weighted combination (30% IF, 70% AE)
    
    **Dataset**: 284,807 credit card transactions
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Single Transaction", "📁 Batch Upload", "📈 Model Insights"])

with tab1:
    st.header("Analyze Single Transaction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transaction Features")
        
        # Amount input
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
        
        # Sample transaction button
        if st.button("🎲 Use Sample Transaction"):
            # Load a sample from test data
            test_df = pd.read_csv('data/processed/test.csv')
            sample = test_df[test_df['Class'] == 0].sample(1).iloc[0]
            st.session_state['sample'] = sample
        
        # Feature inputs (V1-V28 are PCA features)
        st.markdown("**PCA Features (V1-V28)**")
        st.info("In production, these would come from your transaction processing system")
        
        # Create sliders for first few V features as demo
        v_features = {}
        cols = st.columns(3)
        for i in range(1, 6):
            with cols[(i-1) % 3]:
                v_features[f'V{i}'] = st.slider(f'V{i}', -5.0, 5.0, 0.0, 0.1)
        
        # Auto-fill rest with zeros
        for i in range(6, 29):
            v_features[f'V{i}'] = 0.0
        
        # Prepare transaction
        transaction_data = {'Amount': amount, **v_features}
        transaction_df = pd.DataFrame([transaction_data])
        
        # Scale amount
        transaction_df['Amount'] = scaler.transform(transaction_df[['Amount']])
        
        # Predict button
        if st.button("🔍 Analyze Transaction", type="primary"):
            transaction = transaction_df.values[0]
            
            is_fraud, fraud_prob, confidence, scores = predict_fraud(
                transaction, iso_forest, autoencoder, ensemble_config
            )
            
            # Store in session state
            st.session_state['prediction'] = {
                'is_fraud': is_fraud,
                'fraud_prob': fraud_prob,
                'confidence': confidence,
                'scores': scores
            }
    
    with col2:
        st.subheader("Detection Result")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            
            # Main result
            if pred['is_fraud']:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **LEGITIMATE**")
            
            # Fraud probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred['fraud_prob'] * 100,
                title = {'text': "Fraud Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if pred['is_fraud'] else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence
            st.metric("Confidence Level", pred['confidence'])
            
            # Model scores
            st.markdown("**Model Scores:**")
            st.progress(pred['scores']['isolation_forest'], text=f"Isolation Forest: {pred['scores']['isolation_forest']:.3f}")
            st.progress(pred['scores']['autoencoder'], text=f"Autoencoder: {pred['scores']['autoencoder']:.3f}")
            st.progress(pred['scores']['ensemble'], text=f"Ensemble: {pred['scores']['ensemble']:.3f}")

with tab2:
    st.header("Batch Transaction Analysis")
    
    st.markdown("""
    Upload a CSV file with transaction data. Required columns:
    - `Amount`: Transaction amount
    - `V1` through `V28`: PCA-transformed features
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(df)} transactions")
            
            # Show sample
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            if st.button("🔍 Analyze All Transactions"):
                with st.spinner("Analyzing transactions..."):
                    # Prepare data
                    X = df[['Amount'] + [f'V{i}' for i in range(1, 29)]].values
                    
                    # Scale amount
                    X[:, 0] = scaler.transform(X[:, [0]]).flatten()
                    
                    # Get predictions
                    results = []
                    for i, transaction in enumerate(X):
                        is_fraud, fraud_prob, confidence, scores = predict_fraud(
                            transaction, iso_forest, autoencoder, ensemble_config
                        )
                        results.append({
                            'Transaction_ID': i + 1,
                            'Fraud_Detected': is_fraud,
                            'Fraud_Probability': fraud_prob,
                            'Confidence': confidence
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Transactions", len(results_df))
                    col2.metric("Fraudulent", results_df['Fraud_Detected'].sum())
                    col3.metric("Fraud Rate", f"{results_df['Fraud_Detected'].mean()*100:.2f}%")
                    
                    # Results table
                    st.subheader("Detection Results")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.header("Model Insights & Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Comparison")
        
        comparison_data = {
            'Model': ['Isolation Forest', 'Autoencoder', 'Ensemble'],
            'ROC-AUC': [0.852, 0.916, 0.947],
            'Precision': [0.576, 0.267, 0.089],
            'Recall': [0.717, 0.320, 0.427]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True)
        
        # ROC-AUC bar chart
        fig = go.Figure(data=[
            go.Bar(x=comparison_df['Model'], y=comparison_df['ROC-AUC'], 
                   marker_color=['lightblue', 'lightgreen', 'gold'])
        ])
        fig.update_layout(
            title="ROC-AUC Comparison",
            yaxis_title="ROC-AUC Score",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("How It Works")
        
        st.markdown("""
        **1. Isolation Forest**
        - Tree-based anomaly detection
        - Isolates outliers using random splits
        - Fast and efficient
        
        **2. Autoencoder**
        - Neural network (32 → 10 → 32 architecture)
        - Learns normal transaction patterns
        - High reconstruction error = anomaly
        
        **3. Ensemble**
        - Combines both models (30% IF + 70% AE)
        - Leverages strengths of each approach
        - More robust than individual models
        """)
        
        st.info("💡 **Why unsupervised?** In production, fraud labels aren't available in real-time. This system learns normal patterns and flags deviations.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
Built with Streamlit | Models: Isolation Forest + Autoencoder | Dataset: 284K+ transactions
</div>
""", unsafe_allow_html=True)
