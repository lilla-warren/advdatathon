# 🏥 HCT DATATHON 2025 - ENHANCED FAST VERSION
# Added more features while keeping it fast
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

# Configuration
st.set_page_config(page_title="HCT Datathon 2025", layout="wide")

# Generate sample data
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500  # Slightly larger but still fast
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
    data['health_risk'] = (risk > np.percentile(risk, 60)).astype(int)  # Top 40% as high risk
    return pd.DataFrame(data)

def main():
    st.title("🏥 HCT Datathon 2025 - Healthcare Analytics")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", 
                           ["📊 Data Overview", "🤖 Model Training", "📈 Results & Insights", "💡 Recommendations"])
    
    # Load data
    if 'df' not in st.session_state:
        st.session_state.df = generate_data()
    
    df = st.session_state.df
    
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
            st.metric("High Risk Cases", df['health_risk'].sum())
        with col4:
            st.metric("Low Risk Cases", len(df) - df['health_risk'].sum())
        
        # Data preview
        with st.expander("📋 Dataset Preview"):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Shape:** {df.shape}")
        
        # Basic statistics
        with st.expander("📈 Basic Statistics"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Target Distribution")
            risk_counts = df['health_risk'].value_counts()
            fig = px.pie(values=risk_counts.values, names=['Low Risk', 'High Risk'],
                        title="Health Risk Distribution", color=['green', 'red'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution by Risk")
            fig = px.box(df, x='health_risk', y='age', color='health_risk',
                        title="Age Distribution by Risk Level", 
                        labels={'health_risk': 'Risk Level', 'age': 'Age'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, title="Correlation Heatmap", 
                       color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    
    # Page 2: Model Training
    elif page == "🤖 Model Training":
        st.header("🤖 Predictive Model Training")
        
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Select Model:",
                ["Random Forest", "Logistic Regression", "Both"]
            )
            
            test_size = st.slider("Test Size %", 10, 40, 30)
        
        with col2:
            features = st.multiselect(
                "Select Features:",
                [col for col in df.columns if col != 'health_risk'],
                default=['age', 'bmi', 'blood_pressure', 'smoking']
            )
            
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                st.session_state.model_trained = True
                st.session_state.selected_features = features
        
        # Train models when button is clicked
        if st.session_state.get('model_trained', False) and st.session_state.get('selected_features'):
            with st.spinner("Training models... This will take a few seconds."):
                try:
                    # Prepare data
                    X = df[st.session_state.selected_features]
                    y = df['health_risk']
                    
                    # Handle categorical features if any
                    X_encoded = pd.get_dummies(X, drop_first=True) if X.select_dtypes(include=['object']).any().any() else X
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=test_size/100, random_state=42, stratify=y
                    )
                    
                    # Train models
                    models = {}
                    results = []
                    
                    if model_choice in ["Random Forest", "Both"]:
                        rf = RandomForestClassifier(n_estimators=50, random_state=42)
                        rf.fit(X_train, y_train)
                        models['Random Forest'] = rf
                        
                        y_pred_rf = rf.predict(X_test)
                        y_prob_rf = rf.predict_proba(X_test)[:, 1]
                        
                        results.append({
                            "Model": "Random Forest",
                            "Accuracy": accuracy_score(y_test, y_pred_rf),
                            "Precision": precision_score(y_test, y_pred_rf),
                            "Recall": recall_score(y_test, y_pred_rf),
                            "F1-Score": f1_score(y_test, y_pred_rf),
                            "ROC-AUC": roc_auc_score(y_test, y_prob_rf)
                        })
                    
                    if model_choice in ["Logistic Regression", "Both"]:
                        lr = LogisticRegression(max_iter=1000, random_state=42)
                        lr.fit(X_train, y_train)
                        models['Logistic Regression'] = lr
                        
                        y_pred_lr = lr.predict(X_test)
                        y_prob_lr = lr.predict_proba(X_test)[:, 1]
                        
                        results.append({
                            "Model": "Logistic Regression",
                            "Accuracy": accuracy_score(y_test, y_pred_lr),
                            "Precision": precision_score(y_test, y_pred_lr),
                            "Recall": recall_score(y_test, y_pred_lr),
                            "F1-Score": f1_score(y_test, y_pred_lr),
                            "ROC-AUC": roc_auc_score(y_test, y_prob_lr)
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
    
    # Page 3: Results & Insights
    elif page == "📈 Results & Insights":
        st.header("📈 Model Results & Performance")
        
        if not st.session_state.get('models') or st.session_state.get('results', pd.DataFrame()).empty:
            st.info("👈 Please train models first in the 'Model Training' section")
            return
        
        results_df = st.session_state.results
        models = st.session_state.models
        
        # Performance metrics
        st.subheader("📊 Performance Comparison")
        styled_results = results_df.style.format({
            'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 
            'F1-Score': '{:.3f}', 'ROC-AUC': '{:.3f}'
        }).highlight_max(subset=['Accuracy', 'F1-Score', 'ROC-AUC'], color='lightgreen')
        
        st.dataframe(styled_results, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curves")
            fig = go.Figure()
            
            for model_name, model in models.items():
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(st.session_state.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(st.session_state.y_test, y_prob)
                    auc_score = roc_auc_score(st.session_state.y_test, y_prob)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, 
                        name=f'{model_name} (AUC = {auc_score:.3f})',
                        line=dict(width=2)
                    ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title='ROC Curves',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Confusion Matrix")
            selected_model = st.selectbox("Select model for confusion matrix:", list(models.keys()))
            
            if selected_model in models:
                model = models[selected_model]
                y_pred = model.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                
                fig = px.imshow(cm, text_auto=True,
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['Low Risk', 'High Risk'],
                              y=['Low Risk', 'High Risk'],
                              title=f'Confusion Matrix - {selected_model}',
                              color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if 'Random Forest' in models:
            st.subheader("🔍 Feature Importance")
            rf_model = models['Random Forest']
            
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df.head(10), x='Importance', y='Feature',
                       title='Top 10 Most Important Features',
                       color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    # Page 4: Recommendations
    elif page == "💡 Recommendations":
        st.header("💡 Clinical Recommendations & Insights")
        
        if not st.session_state.get('models'):
            st.info("👈 Please train models first to get recommendations")
            return
        
        # Get best model
        results_df = st.session_state.results
        best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        best_model = st.session_state.models[best_model_name]
        
        st.success(f"**Best Performing Model:** {best_model_name} (F1-Score: {results_df.loc[results_df['F1-Score'].idxmax(), 'F1-Score']:.3f})")
        
        # Feature-based recommendations
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            top_features = importance_df.head(3)
            
            st.subheader("🎯 Top Intervention Targets")
            
            for idx, (_, row) in enumerate(top_features.iterrows(), 1):
                feature = row['Feature']
                importance = row['Importance']
                
                with st.expander(f"{idx}. {feature} (Impact: {importance:.3f})"):
                    if 'bmi' in feature.lower():
                        st.markdown("""
                        **Weight Management Strategy:**
                        - Implement structured diet programs
                        - Encourage 150+ minutes of weekly exercise
                        - Regular BMI monitoring
                        - Nutritional counseling sessions
                        """)
                    elif 'blood' in feature.lower() or 'pressure' in feature.lower():
                        st.markdown("""
                        **Blood Pressure Control:**
                        - Regular BP monitoring (weekly)
                        - Sodium intake reduction
                        - Stress management techniques
                        - Medication adherence support
                        """)
                    elif 'smoking' in feature.lower():
                        st.markdown("""
                        **Smoking Cessation Program:**
                        - Nicotine replacement therapy
                        - Behavioral counseling
                        - Support group referrals
                        - Regular follow-ups
                        """)
                    elif 'age' in feature.lower():
                        st.markdown("""
                        **Age-Appropriate Screening:**
                        - Enhanced monitoring for age-related risks
                        - Regular health check-ups
                        - Preventive care emphasis
                        - Lifestyle modification support
                        """)
                    else:
                        st.markdown("""
                        **General Health Intervention:**
                        - Regular health screenings
                        - Lifestyle modification programs
                        - Patient education
                        - Continuous monitoring
                        """)
        
        # Risk stratification insight
        st.subheader("📊 Population Health Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_risk_pct = (df['health_risk'].sum() / len(df)) * 100
            st.metric("High Risk Population", f"{high_risk_pct:.1f}%")
        
        with col2:
            avg_age = df['age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        
        with col3:
            smoking_rate = (df['smoking'].sum() / len(df)) * 100
            st.metric("Smoking Prevalence", f"{smoking_rate:.1f}%")
        
        # Actionable summary
        st.subheader("🚀 Recommended Action Plan")
        st.markdown("""
        1. **Priority Screening** for high-risk individuals identified by the model
        2. **Targeted Interventions** based on top predictive features
        3. **Preventive Care** programs for moderate-risk population
        4. **Continuous Monitoring** with regular model updates
        5. **Stakeholder Education** on risk factors and prevention
        """)

if __name__ == "__main__":
    main()
