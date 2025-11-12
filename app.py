# 🏥 HCT DATATHON 2025 - OPTIMIZED HEALTHCARE ANALYTICS PLATFORM
# Lightweight and fast Streamlit app
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings("ignore")

# Page configuration - SIMPLIFIED
st.set_page_config(
    page_title="HCT Datathon 2025",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for better performance
)

# Custom CSS - MINIMAL
st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .section-header { font-size: 1.5rem; color: #2e86ab; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()

@st.cache_data
def load_data(uploaded_file):
    """Cache data loading"""
    return pd.read_csv(uploaded_file)

@st.cache_data
def generate_sample_data():
    """Generate sample data for demo"""
    np.random.seed(42)
    n_samples = 500  # Reduced for performance
    
    data = {
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'bmi': np.random.normal(25, 5, n_samples),
        'blood_pressure': np.random.normal(120, 15, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
        'glucose': np.random.normal(100, 20, n_samples),
        'exercise_hours': np.random.exponential(3, n_samples),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    # Simple target creation
    risk_score = (
        data['age'] * 0.1 + data['bmi'] * 0.3 + data['smoking'] * 10 + 
        np.random.normal(0, 5, n_samples)
    )
    data['health_risk'] = (risk_score > risk_score.mean()).astype(int)
    
    return pd.DataFrame(data)

def main():
    # Simple header
    st.markdown('<div class="main-header">🏥 HCT Datathon 2025 - Healthcare Analytics</div>', unsafe_allow_html=True)
    
    # Sidebar - SIMPLIFIED
    with st.sidebar:
        st.header("📁 Data Management")
        
        use_sample = st.checkbox("Use Sample Data", value=True)
        if not use_sample:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                st.session_state.df = load_data(uploaded_file)
        else:
            if st.button("Generate Sample Data") or st.session_state.df is None:
                st.session_state.df = generate_sample_data()
        
        if st.session_state.df is not None:
            st.session_state.target_col = st.selectbox("Select Target Variable:", st.session_state.df.columns)
            
            analysis_type = st.radio(
                "Select Analysis:",
                ["📊 Data Overview", "🤖 Quick Modeling", "📈 Results"]
            )
    
    # Main content
    if st.session_state.df is None:
        st.info("👈 Please load data from the sidebar to begin analysis")
        return
    
    df = st.session_state.df
    
    if analysis_type == "📊 Data Overview":
        st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
        
        # Basic info in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Quick stats
        st.subheader("Quick Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Simple visualizations
        st.subheader("Basic Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.target_col:
                # Target distribution
                fig = px.pie(df, names=st.session_state.target_col, 
                           title=f"Target Distribution - {st.session_state.target_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap (numeric only)
            numeric_df = df.select_dtypes(include=np.number)
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                fig = px.imshow(corr, title="Correlation Heatmap", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "🤖 Quick Modeling":
        st.markdown('<div class="section-header">🤖 Quick Predictive Modeling</div>', unsafe_allow_html=True)
        
        if not st.session_state.target_col:
            st.warning("Please select a target variable first")
            return
        
        # Simple model configuration
        st.subheader("Model Configuration")
        
        model_choice = st.selectbox(
            "Select Model:",
            ["Logistic Regression", "Random Forest", "Both"]
        )
        
        test_size = st.slider("Test Size %", 10, 40, 30)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Prepare data
                X = df.drop(columns=[st.session_state.target_col])
                y = df[st.session_state.target_col]
                
                # Handle categorical data simply
                X = pd.get_dummies(X, drop_first=True)
                
                # Encode target if needed
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                results = []
                
                if model_choice in ["Logistic Regression", "Both"]:
                    lr = LogisticRegression(max_iter=1000, random_state=42)
                    lr.fit(X_train_scaled, y_train)
                    st.session_state.models['Logistic Regression'] = lr
                    
                    y_pred = lr.predict(X_test_scaled)
                    y_prob = lr.predict_proba(X_test_scaled)[:, 1]
                    
                    results.append({
                        "Model": "Logistic Regression",
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred),
                        "Recall": recall_score(y_test, y_pred),
                        "F1-Score": f1_score(y_test, y_pred),
                        "ROC-AUC": roc_auc_score(y_test, y_prob)
                    })
                
                if model_choice in ["Random Forest", "Both"]:
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced for speed
                    rf.fit(X_train_scaled, y_train)
                    st.session_state.models['Random Forest'] = rf
                    
                    y_pred = rf.predict(X_test_scaled)
                    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
                    
                    results.append({
                        "Model": "Random Forest",
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred),
                        "Recall": recall_score(y_test, y_pred),
                        "F1-Score": f1_score(y_test, y_pred),
                        "ROC-AUC": roc_auc_score(y_test, y_prob)
                    })
                
                st.session_state.results = pd.DataFrame(results)
                st.success("✅ Model training completed!")
    
    elif analysis_type == "📈 Results":
        st.markdown('<div class="section-header">📈 Model Results</div>', unsafe_allow_html=True)
        
        if st.session_state.results.empty:
            st.info("Please train models first in the 'Quick Modeling' section")
            return
        
        # Display results
        st.subheader("Performance Metrics")
        st.dataframe(st.session_state.results.style.format({
            'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 
            'F1-Score': '{:.3f}', 'ROC-AUC': '{:.3f}'
        }), use_container_width=True)
        
        # Simple visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curve for first model
            if 'Logistic Regression' in st.session_state.models:
                model = st.session_state.models['Logistic Regression']
                X = df.drop(columns=[st.session_state.target_col])
                X = pd.get_dummies(X, drop_first=True)
                y = df[st.session_state.target_col]
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                X_test_scaled = scaler.fit_transform(X_test)
                
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = roc_auc_score(y_test, y_prob)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {auc_score:.3f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
                fig.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance for Random Forest
            if 'Random Forest' in st.session_state.models:
                model = st.session_state.models['Random Forest']
                X = df.drop(columns=[st.session_state.target_col])
                X = pd.get_dummies(X, drop_first=True)
                
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                fig = px.bar(importance_df, x='importance', y='feature', 
                           title='Top 10 Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
        
        # Simple insights
        st.subheader("💡 Key Insights")
        
        best_model = st.session_state.results.loc[st.session_state.results['F1-Score'].idxmax(), 'Model']
        best_score = st.session_state.results.loc[st.session_state.results['F1-Score'].idxmax(), 'F1-Score']
        
        st.write(f"**Best Model**: {best_model} (F1-Score: {best_score:.3f})")
        
        if 'Random Forest' in st.session_state.models:
            st.write("**Top Features**:")
            model = st.session_state.models['Random Forest']
            X = df.drop(columns=[st.session_state.target_col])
            X = pd.get_dummies(X, drop_first=True)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(3)
            
            for _, row in importance_df.iterrows():
                st.write(f"- {row['feature']} (importance: {row['importance']:.3f})")

if __name__ == "__main__":
    main()
