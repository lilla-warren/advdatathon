# 🏥 HCT DATATHON 2025 - ADVANCED HEALTHCARE ANALYTICS PLATFORM
# Streamlit app for descriptive, diagnostic, predictive, and ethical AI analytics
# Author: (Your Name)
# Institution: Higher Colleges of Technology
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import shap
import joblib
import io
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🏥 HCT Datathon 2025 - Healthcare Analytics",
    page_icon="💉",
    layout="wide"
)

# ----------------------------------------------------------
# 1️⃣ Header and Description
# ----------------------------------------------------------
st.title("🏥 HCT Datathon 2025 Challenge — Clinical Intelligence Platform")
st.markdown("""
### Goal
The HCT Datathon 2025 Challenge promotes innovation and ethical AI in healthcare analytics.  
Participants transform **data into knowledge**, enabling **informed decision-making** and advancing **responsible AI** practice across HCT.

**Objectives:**
1. Apply ML & data analytics to solve a healthcare prediction problem.  
2. Derive descriptive, diagnostic & predictive insights.  
3. Build interpretable, responsible AI models.  
4. Emphasize transparency, fairness, and ethical reflection.
---
""")

# ----------------------------------------------------------
# 2️⃣ File Upload Section
# ----------------------------------------------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data successfully uploaded!")
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# ----------------------------------------------------------
# 3️⃣ Descriptive Analytics
# ----------------------------------------------------------
st.header("📊 Descriptive Analytics")

st.write("**Dataset Shape:**", df.shape)
st.write("**Column Overview:**", list(df.columns))

# Summary stats
st.subheader("Summary Statistics")
st.dataframe(df.describe())

# Distribution visualizations
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if num_cols:
    st.subheader("Variable Distributions")
    selected_num = st.selectbox("Select a numerical feature:", num_cols)
    fig = px.histogram(df, x=selected_num, title=f"Distribution of {selected_num}", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

# Class balance visualization
st.subheader("Class Distribution")
target_col = st.selectbox("Select target variable (classification):", df.columns)
if target_col:
    fig = px.pie(df, names=target_col, title=f"Class Distribution for {target_col}")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# 4️⃣ Diagnostic Analytics
# ----------------------------------------------------------
st.header("🧠 Diagnostic Analytics")

st.subheader("Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# Pairwise plots for exploration
st.subheader("Pairwise Relationships")
if len(num_cols) > 1:
    selected_features = st.multiselect("Select up to 3 numeric features:", num_cols, default=num_cols[:3])
    if len(selected_features) >= 2:
        fig = px.scatter_matrix(df, dimensions=selected_features, color=target_col)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# 5️⃣ Predictive Analytics
# ----------------------------------------------------------
st.header("🤖 Predictive Analytics")

st.markdown("""
**Goal:** Build classification models (Logistic Regression, Decision Tree, Random Forest)  
using a 70:30 train-test split and evaluate with Accuracy, Precision, Recall, F1, and ROC-AUC.
""")

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target if categorical
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    }
    results.append(metrics)

results_df = pd.DataFrame(results)
st.subheader("Model Evaluation Metrics")
st.dataframe(results_df)

# Visualization: ROC curve for Random Forest
best_model = models["Random Forest"]
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
fig.update_layout(title="ROC Curve - Random Forest", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig, use_container_width=True)

# Confusion Matrix
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
st.pyplot(fig)

# ----------------------------------------------------------
# 6️⃣ Explainability & Transparency (SHAP)
# ----------------------------------------------------------
st.header("🧩 Explainability & Transparency")

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

st.subheader("SHAP Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(feat_imp)

# ----------------------------------------------------------
# 7️⃣ Prescriptive Analytics & Ethics Reflection
# ----------------------------------------------------------
st.header("💡 Prescriptive Insights & Ethics")

st.markdown("""
**Key Insights:**
- Identify top contributing features for prediction.
- Suggest early interventions based on data patterns.
- Support responsible decision-making with explainable results.

**Ethical Reflection:**
- Ensure fairness: Evaluate if model treats all groups equitably.
- Protect privacy: Use anonymized and de-identified data.
- Maintain accountability: Document model assumptions and limitations.
""")

# ----------------------------------------------------------
# 8️⃣ Downloadables
# ----------------------------------------------------------
st.header("📤 Download Results")

buffer = io.BytesIO()
results_df.to_csv(buffer, index=False)
st.download_button("Download Model Metrics (CSV)", data=buffer.getvalue(), file_name="model_metrics.csv", mime="text/csv")

joblib.dump(best_model, "trained_model.joblib")
with open("trained_model.joblib", "rb") as f:
    st.download_button("Download Trained Model", data=f, file_name="trained_model.joblib")
