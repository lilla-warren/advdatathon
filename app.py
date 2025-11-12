# app.py - CLINICAL INTELLIGENCE PLATFORM (Optimized for Streamlit Cloud)
# Author: HCT Datathon 2025 Team
# Description: Advanced Healthcare Analytics with ML, Explainability & Ethics

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import shap
import joblib
import io
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Clinical Intelligence Platform",
    page_icon="🏥",
    layout="wide",
)

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------
st.markdown("""
<style>
    .main {background-color: #f9fafc;}
    .section-header {
        font-size: 22px; font-weight: 600; color: #2c3e50;
        margin-top: 1.5em; border-bottom: 2px solid #3498db; padding-bottom: 5px;
    }
    .stDataFrame {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# App State Holder
# ----------------------------------------------------------
class AnalyticsState:
    def __init__(self):
        self.models = {}
        self.results = pd.DataFrame()
        self.target_col = None

analytics = AnalyticsState()

# ----------------------------------------------------------
# Sidebar Setup
# ----------------------------------------------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# ----------------------------------------------------------
# Main Function
# ----------------------------------------------------------
def main():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data uploaded successfully!")
        st.write(f"**Dataset Shape:** {df.shape}")
        st.dataframe(df.head(), use_container_width=True)

        # Select target column
        analytics.target_col = st.selectbox("🎯 Select Target Variable", df.columns)

        # Select analysis type
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Module",
            [
                "📊 Data Exploration",
                "🤖 Model Training & Evaluation",
                "🧮 Model Explainability",
                "💡 Prescriptive Insights",
                "⚖️ Ethics & Governance"
            ]
        )

        # ----------------------------------------------------------
        # 1️⃣ Data Exploration
        # ----------------------------------------------------------
        if analysis_type == "📊 Data Exploration":
            st.markdown('<div class="section-header">📊 Data Exploration & Visualization</div>', unsafe_allow_html=True)
            st.write("Basic statistics and feature distributions.")
            st.write(df.describe())

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select numeric column", numeric_cols)
                fig = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------
        # 2️⃣ Model Training & Evaluation
        # ----------------------------------------------------------
        elif analysis_type == "🤖 Model Training & Evaluation":
            st.markdown('<div class="section-header">🤖 Predictive Modeling</div>', unsafe_allow_html=True)
            st.write("Train multiple models and compare their performance.")

            target = analytics.target_col
            X = df.drop(columns=[target])
            y = df[target]

            categorical_cols = X.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                X = pd.get_dummies(X, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models_config = {
                "Logistic Regression": {"model": LogisticRegression(max_iter=200), "params": None},
                "Random Forest": {"model": RandomForestClassifier(random_state=42), "params": {'n_estimators': [100, 200]}},
                "Gradient Boosting": {"model": GradientBoostingClassifier(random_state=42), "params": None},
                "Support Vector Machine": {"model": SVC(probability=True), "params": {'C': [0.1, 1, 10]}}
            }

            selected_models = st.multiselect("Select models to train:", list(models_config.keys()),
                                             default=["Logistic Regression", "Random Forest"])
            use_grid_search = st.checkbox("Enable Hyperparameter Tuning (Grid Search)")
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

            if st.button("🚀 Train Models"):
                results = []
                for model_name in selected_models:
                    st.write(f"Training {model_name}...")
                    cfg = models_config[model_name]
                    model = cfg["model"]

                    if use_grid_search and cfg["params"]:
                        grid = GridSearchCV(model, cfg["params"], cv=cv_folds, scoring='f1_macro')
                        grid.fit(X_train_scaled, y_train)
                        best_model = grid.best_estimator_
                        st.success(f"Best params for {model_name}: {grid.best_params_}")
                    else:
                        best_model = model.fit(X_train_scaled, y_train)

                    analytics.models[model_name] = best_model
                    y_pred = best_model.predict(X_test_scaled)
                    y_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None

                    cv_score = cross_val_score(best_model, X_train_scaled, y_train, cv=cv_folds, scoring='f1_macro')
                    metrics = {
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average='weighted'),
                        "Recall": recall_score(y_test, y_pred, average='weighted'),
                        "F1-Score": f1_score(y_test, y_pred, average='weighted'),
                        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
                        "CV F1-Mean": cv_score.mean(),
                        "CV F1-Std": cv_score.std()
                    }
                    results.append(metrics)

                analytics.results = pd.DataFrame(results)
                st.success("✅ Models trained successfully!")

                st.dataframe(analytics.results.style.highlight_max(['Accuracy', 'F1-Score'], color='lightgreen'))

        # ----------------------------------------------------------
        # 3️⃣ Model Explainability
        # ----------------------------------------------------------
        elif analysis_type == "🧮 Model Explainability":
            st.markdown('<div class="section-header">🧮 Model Explainability</div>', unsafe_allow_html=True)

            if not analytics.models:
                st.warning("⚠️ Train models first.")
                return

            selected_model = st.selectbox("Select model for SHAP explainability", list(analytics.models.keys()))
            model = analytics.models[selected_model]

            X = df.drop(columns=[analytics.target_col])
            X = pd.get_dummies(X, drop_first=True)
            sample = X.sample(min(100, len(X)), random_state=42)

            try:
                explainer = shap.Explainer(model, sample)
                shap_values = explainer(sample)

                st.subheader("📈 SHAP Summary")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values.values, sample, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")

        # ----------------------------------------------------------
        # 4️⃣ Prescriptive Insights
        # ----------------------------------------------------------
        elif analysis_type == "💡 Prescriptive Insights":
            st.markdown('<div class="section-header">💡 Prescriptive Insights</div>', unsafe_allow_html=True)

            if not analytics.models or analytics.results.empty:
                st.warning("⚠️ Train models first.")
                return

            best_model_name = analytics.results.loc[analytics.results['F1-Score'].idxmax(), 'Model']
            best_model = analytics.models[best_model_name]

            st.success(f"Best model: {best_model_name}")

            if hasattr(best_model, "feature_importances_"):
                X = df.drop(columns=[analytics.target_col])
                feature_imp = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": best_model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(feature_imp.set_index("Feature"))

        # ----------------------------------------------------------
        # 5️⃣ Ethics & Governance
        # ----------------------------------------------------------
        elif analysis_type == "⚖️ Ethics & Governance":
            st.markdown('<div class="section-header">⚖️ Ethics & Governance</div>', unsafe_allow_html=True)
            st.markdown("""
            - Transparent ML explainability  
            - Bias detection and fairness  
            - Patient data anonymization  
            - Continuous validation and monitoring
            """)

    else:
        st.markdown("""
        ## 🏥 Welcome to the Clinical Intelligence Platform
        Upload your dataset using the sidebar to begin exploring insights!
        """)

# ----------------------------------------------------------
# Run App
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
