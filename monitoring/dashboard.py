# monitoring/dashboard.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.evaluate import evaluate_model

st.title("Football Predictor Monitoring Dashboard")

# Load historical metrics
try:
    metrics_history = pd.read_csv('monitoring/metrics_history.csv')
    st.line_chart(metrics_history.set_index('date'))
except FileNotFoundError:
    st.warning("No historical metrics found")

# Run new evaluation
if st.button("Run Evaluation on Latest Data"):
    with st.spinner("Evaluating model..."):
        metrics = evaluate_model('data/raw/latest_fixtures.csv')
        st.success("Evaluation completed!")
        st.json(metrics)
        
        # Save to history
        new_row = pd.DataFrame({
            'date': [pd.Timestamp.now()],
            'accuracy': [metrics['accuracy']],
            'f1_macro': [metrics['f1_macro']],
            'home_xg_mae': [metrics['home_xg_mae']],
            'away_xg_mae': [metrics['away_xg_mae']]
        })
        
        try:
            history = pd.read_csv('monitoring/metrics_history.csv')
            updated = pd.concat([history, new_row])
        except FileNotFoundError:
            updated = new_row
            
        updated.to_csv('monitoring/metrics_history.csv', index=False)
        st.experimental_rerun()

# Check for data drift
if st.button("Check for Data Drift"):
    with st.spinner("Analyzing data distributions..."):
        drift_results = detect_drift()
        drift_detected = any([result['drift_detected'] for result in drift_results.values()])
        
        if drift_detected:
            st.error("Data drift detected! Consider retraining the model.")
            st.json(drift_results)
        else:
            st.success("No significant data drift detected")