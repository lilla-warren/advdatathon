# 🏥 HCT DATATHON 2025 - ULTRA FAST VERSION
# Minimal Streamlit app for maximum speed
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Absolute minimal configuration
st.set_page_config(page_title="HCT Fast", layout="centered")

# Generate sample data directly (no file upload for now)
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 300  # Small dataset
    data = {
        'age': np.random.normal(45, 15, n),
        'bmi': np.random.normal(25, 5, n),
        'bp': np.random.normal(120, 15, n),
    }
    # Simple binary target
    risk = data['age'] * 0.1 + data['bmi'] * 0.3 + np.random.normal(0, 3, n)
    data['risk_high'] = (risk > risk.mean()).astype(int)
    return pd.DataFrame(data)

def main():
    st.title("🏥 HCT Datathon - Fast Demo")
    
    # Load data once
    if 'df' not in st.session_state:
        st.session_state.df = generate_data()
    
    df = st.session_state.df
    
    # Simple interface
    st.write(f"Data: {len(df)} samples, {len(df.columns)} features")
    
    # Show data
    if st.checkbox("Show data preview"):
        st.dataframe(df.head())
    
    # Quick model
    if st.button("🚀 Train Quick Model"):
        with st.spinner("Training..."):
            # Simple preprocessing
            X = df[['age', 'bmi', 'bp']]
            y = df['risk_high']
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)  # Very small
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success(f"✅ Model trained! Accuracy: {accuracy:.3f}")
            
            # Show feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.write("**Feature Importance:**")
            st.dataframe(importance)

if __name__ == "__main__":
    main()
