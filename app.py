# 🏥 HCT DATATHON 2025 - CUSTOM DATASET UPLOAD VERSION
# Supports both sample data and custom dataset upload
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

# Configuration
st.set_page_config(page_title="HCT Datathon 2025", layout="wide")

# Generate sample data for demo
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n = 500
    data = {
        'age': np.random.normal(45, 15, n).astype(int),
        'bmi': np.random.normal(25, 5, n),
        'blood_pressure': np.random.normal(120, 15, n),
        'cholesterol': np.random.normal(200, 40, n),
        'glucose': np.random.normal(100, 20, n),
        'exercise_hours': np.random.exponential(3, n),
        'smoking': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'family_history': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    }
    # Realistic risk calculation
    risk = (data['age'] * 0.1 + data['bmi'] * 0.3 + data['blood_pressure'] * 0.05 + 
            data['cholesterol'] * 0.1 + data['smoking'] * 15 + data['family_history'] * 10 -
            data['exercise_hours'] * 2 + np.random.normal(0, 8, n))
    data['health_risk'] = (risk > np.percentile(risk, 60)).astype(int)
    return pd.DataFrame(data)

def safe_get_dummies(df, columns=None):
    """Safe version of get_dummies that ensures unique column names"""
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    if len(columns) == 0:
        return df
    
    # Get dummies and ensure unique column names
    dummies = pd.get_dummies(df[columns], drop_first=True)
    
    # Check for duplicate column names
    if len(dummies.columns) != len(set(dummies.columns)):
        # If duplicates exist, create unique names
        new_columns = []
        for col in dummies.columns:
            if col in df.columns:
                new_columns.append(f"{col}_dummy")
            else:
                new_columns.append(col)
        dummies.columns = new_columns
    
    # Drop original categorical columns and join with dummies
    result = df.drop(columns=columns).join(dummies)
    
    return result

def main():
    st.title("🏥 HCT Datathon 2025 - Healthcare Analytics")
    
    # Sidebar for data management
    st.sidebar.title("📁 Data Management")
    
    data_source = st.sidebar.radio("Choose data source:", 
                                  ["Use Sample Data", "Upload Your Dataset"])
    
    if data_source == "Use Sample Data":
        if st.session_state.df is None or st.sidebar.button("Generate Sample Data"):
            st.session_state.df = generate_sample_data()
            st.session_state.target_col = 'health_risk'
            st.sidebar.success("Sample data loaded!")
    
    else:  # Upload Your Dataset
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"Dataset loaded! Shape: {st.session_state.df.shape}")
                
                # Let user select target column
                if st.session_state.df is not None:
                    st.session_state.target_col = st.sidebar.selectbox(
                        "Select target variable:", 
                        st.session_state.df.columns
                    )
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
    
    # Check if we have data to work with
    if st.session_state.df is None or st.session_state.target_col is None:
        st.info("👋 Welcome! Please load your data using the sidebar to begin analysis.")
        st.markdown("""
        ### 🚀 Getting Started:
        1. **Choose data source** in the sidebar (sample data or upload your own CSV)
        2. **If uploading**, select your target variable
        3. **Navigate** through the different analysis sections
        
        ### 📊 Sample Data Includes:
        - Age, BMI, Blood Pressure, Cholesterol, Glucose levels
        - Exercise hours, Smoking status, Family history
        - Binary health risk prediction target
        
        ### 🎯 Analysis Features:
        - **Data Overview**: Explore and visualize your data
        - **Model Training**: Build predictive models
        - **Results & Insights**: Evaluate model performance
        - **Recommendations**: Get actionable insights
        """)
        return
    
    df = st.session_state.df
    target_col = st.session_state.target_col
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", 
                           ["📊 Data Overview", "🤖 Model Training", "📈 Results & Insights", "💡 Recommendations"])
    
    # Page 1: Data Overview
    if page == "📊 Data Overview":
        st.header("📊 Data Overview & Exploration")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            if df[target_col].dtype in [np.int64, np.float64]:
                high_risk = df[target_col].sum() if df[target_col].nunique() == 2 else "N/A"
                st.metric("High Risk Cases", high_risk)
            else:
                st.metric("Classes", df[target_col].nunique())
        with col4:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        # Data preview
        with st.expander("📋 Dataset Preview"):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Target Variable:** {target_col}")
        
        # Basic statistics
        with st.expander("📈 Basic Statistics"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Data quality info
        with st.expander("🔍 Data Quality Info"):
            st.write("**Data Types:**")
            st.write(df.dtypes)
            st.write("**Missing Values per Column:**")
            st.write(df.isnull().sum())
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Target Distribution")
            if df[target_col].dtype in [np.int64, np.float64] and df[target_col].nunique() <= 10:
                # For numeric targets with few unique values (like binary classification)
                target_counts = df[target_col].value_counts()
                fig = px.pie(values=target_counts.values, names=target_counts.index,
                            title=f"Target Distribution - {target_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # For continuous targets or many classes
                fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Distribution")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                feature_to_plot = st.selectbox("Select feature to plot:", numeric_cols)
                fig = px.histogram(df, x=feature_to_plot, title=f"Distribution of {feature_to_plot}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, title="Correlation Heatmap", 
                           color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric features for correlation heatmap")
        
        # Simple scatter plot
        st.subheader("Feature Relationships")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis feature:", numeric_cols, key="x_feature")
            with col2:
                y_feature = st.selectbox("Y-axis feature:", numeric_cols, key="y_feature")
            
            if x_feature and y_feature and x_feature != y_feature:
                color_feature = target_col if target_col in numeric_cols else None
                fig = px.scatter(df, x=x_feature, y=y_feature, color=color_feature,
                               title=f"{x_feature} vs {y_feature}",
                               color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig, use_container_width=True)
    
    # Page 2: Model Training
    elif page == "🤖 Model Training":
        st.header("🤖 Predictive Model Training")
        
        st.info(f"**Target Variable:** {target_col}")
        
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Select Model:",
                ["Random Forest", "Logistic Regression", "Both"]
            )
            
            test_size = st.slider("Test Size %", 10, 40, 30)
            
            # Handle categorical target
            if df[target_col].dtype == 'object':
                st.warning("Target variable is categorical. It will be encoded for modeling.")
        
        with col2:
            # Available features (excluding target)
            available_features = [col for col in df.columns if col != target_col]
            features = st.multiselect(
                "Select Features:",
                available_features,
                default=available_features[:min(4, len(available_features))]  # Default to first 4 features
            )
            
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                if not features:
                    st.error("Please select at least one feature")
                else:
                    st.session_state.model_trained = True
                    st.session_state.selected_features = features
        
        # Train models when button is clicked
        if st.session_state.get('model_trained', False) and st.session_state.get('selected_features'):
            with st.spinner("Training models... This will take a few seconds."):
                try:
                    # Prepare data
                    X = df[st.session_state.selected_features]
                    y = df[target_col]
                    
                    # Encode target if categorical
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                        st.info(f"Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                    else:
                        y_encoded = y
                    
                    # Use safe get_dummies for features
                    X_encoded = safe_get_dummies(X)
                    
                    # Ensure no duplicate column names
                    if len(X_encoded.columns) != len(set(X_encoded.columns)):
                        X_encoded.columns = [f"{col}_{i}" if col in X_encoded.columns[:i] else col 
                                           for i, col in enumerate(X_encoded.columns)]
                    
                    # Split data - with safe stratification
                    unique_classes = len(np.unique(y_encoded))
                    can_stratify = unique_classes > 1 and min(pd.Series(y_encoded).value_counts()) > 1
                    
                    if can_stratify:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_encoded, y_encoded, test_size=test_size/100, random_state=42, stratify=y_encoded
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_encoded, y_encoded, test_size=test_size/100, random_state=42
                        )
                    
                    # Train models
                    models = {}
                    results = []
                    
                    if model_choice in ["Random Forest", "Both"]:
                        rf = RandomForestClassifier(n_estimators=50, random_state=42)
                        rf.fit(X_train, y_train)
                        models['Random Forest'] = rf
                        
                        y_pred_rf = rf.predict(X_test)
                        y_prob_rf = rf.predict_proba(X_test)[:, 1] if hasattr(rf, "predict_proba") else None
                        
                        results.append({
                            "Model": "Random Forest",
                            "Accuracy": accuracy_score(y_test, y_pred_rf),
                            "Precision": precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
                            "Recall": recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
                            "F1-Score": f1_score(y_test, y_pred_rf, average='weighted', zero_division=0),
                            "ROC-AUC": roc_auc_score(y_test, y_prob_rf) if y_prob_rf is not None and len(np.unique(y_test)) > 1 else "N/A"
                        })
                    
                    if model_choice in ["Logistic Regression", "Both"]:
                        lr = LogisticRegression(max_iter=1000, random_state=42)
                        lr.fit(X_train, y_train)
                        models['Logistic Regression'] = lr
                        
                        y_pred_lr = lr.predict(X_test)
                        y_prob_lr = lr.predict_proba(X_test)[:, 1] if hasattr(lr, "predict_proba") else None
                        
                        results.append({
                            "Model": "Logistic Regression",
                            "Accuracy": accuracy_score(y_test, y_pred_lr),
                            "Precision": precision_score(y_test, y_pred_lr, average='weighted', zero_division=0),
                            "Recall": recall_score(y_test, y_pred_lr, average='weighted', zero_division=0),
                            "F1-Score": f1_score(y_test, y_pred_lr, average='weighted', zero_division=0),
                            "ROC-AUC": roc_auc_score(y_test, y_prob_lr) if y_prob_lr is not None and len(np.unique(y_test)) > 1 else "N/A"
                        })
                    
                    # Store results
                    st.session_state.models = models
                    st.session_state.results = pd.DataFrame(results)
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = X_encoded.columns.tolist()
                    
                    st.success("✅ Models trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
    
    # Pages 3 and 4 remain the same as previous version...
    # [The rest of the code for Results & Insights and Recommendations pages would go here]
    # For brevity, I'm showing the key changes for dataset upload

if __name__ == "__main__":
    main()
