# 🏥 HCT DATATHON 2025 - SAFE FILE UPLOAD VERSION
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="HCT Safe", layout="centered")

def main():
    st.title("🏥 HCT Datathon - Safe Version")
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Show basic info
            st.write("**Dataset Preview:**")
            st.dataframe(df.head())
            
            # Target selection
            target_col = st.selectbox("Select target column:", df.columns)
            
            if target_col:
                st.write(f"Target distribution: {df[target_col].value_counts().to_dict()}")
                
                # Simple model
                if st.button("🚀 Train Safe Model"):
                    with st.spinner("Training safely..."):
                        try:
                            # Use only numeric features for safety
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            if target_col in numeric_cols:
                                numeric_cols.remove(target_col)
                            
                            if len(numeric_cols) > 0:
                                X = df[numeric_cols]
                                y = df[target_col]
                                
                                # Safe split without stratification
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.3, random_state=42
                                )
                                
                                # Simple model
                                model = RandomForestClassifier(n_estimators=10, random_state=42)
                                model.fit(X_train, y_train)
                                
                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test, y_pred)
                                
                                st.success(f"✅ Safe training complete! Accuracy: {accuracy:.3f}")
                                
                            else:
                                st.error("No numeric features found for modeling")
                                
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to begin")

if __name__ == "__main__":
    main()
