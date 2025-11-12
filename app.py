import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import joblib

st.title("🏥 Healthcare Analytics Platform")

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.write(df.head())

    target_col = st.selectbox("Select the target column", df.columns)
    
    # -----------------------------
    # 2️⃣ Preprocess Data
    # -----------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    min_class_count = y_train.value_counts().min()
    cv_folds = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # -----------------------------
    # 3️⃣ Models
    # -----------------------------
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"C": [0.1, 1, 10]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
        }
    }
    
    selected_models = st.multiselect("Select models to train", list(models_config.keys()), default=list(models_config.keys()))
    use_grid_search = st.checkbox("Use Grid Search for hyperparameter tuning", value=True)
    
    analytics = type("Analytics", (), {})()
    analytics.models = {}
    results = []

    # -----------------------------
    # 4️⃣ Train Models
    # -----------------------------
    for model_name in selected_models:
        st.write(f"Training {model_name}...")
        model_config = models_config[model_name]
        model = model_config["model"]
        
        if use_grid_search and model_config["params"]:
            gs = GridSearchCV(model, model_config["params"], cv=cv, scoring="f1_macro")
            gs.fit(X_train_scaled, y_train)
            best_model = gs.best_estimator_
            st.success(f"Best params for {model_name}: {gs.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train_scaled, y_train)
        
        analytics.models[model_name] = best_model
        
        # Predictions
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        # Metrics
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring="f1_macro")
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
            "CV F1-Mean": cv_scores.mean(),
            "CV F1-Std": cv_scores.std()
        }
        results.append(metrics)
    
    analytics.results = pd.DataFrame(results)

    # -----------------------------
    # 5️⃣ Display Results
    # -----------------------------
    st.subheader("📊 Model Performance Metrics")
    st.dataframe(analytics.results)

    # ROC Curves
    st.subheader("📈 ROC Curves")
    fig = go.Figure()
    for model_name in selected_models:
        model = analytics.models[model_name]
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{model_name} (AUC={auc_score:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig)

    # -----------------------------
    # 6️⃣ Confusion Matrix
    # -----------------------------
    st.subheader("🧮 Confusion Matrix")
    selected_model_cm = st.selectbox("Select model for confusion matrix", selected_models)
    cm_model = analytics.models[selected_model_cm]
    y_pred_cm = cm_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_cm)
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues")
    st.plotly_chart(fig_cm)

    # -----------------------------
    # 7️⃣ SHAP Explainability
    # -----------------------------
    st.subheader("🔍 SHAP Feature Importance")
    shap_model_name = st.selectbox("Select model for SHAP", selected_models)
    shap_model = analytics.models[shap_model_name]

    try:
        X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
        X_sample_encoded = pd.get_dummies(X_sample, drop_first=True)
        explainer = shap.Explainer(shap_model, X_sample_encoded)
        shap_values = explainer(X_sample_encoded)

        fig, ax = plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_values.values, X_sample_encoded, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")

    # -----------------------------
    # 8️⃣ Prescriptive Insights
    # -----------------------------
    st.subheader("💡 Prescriptive Insights")
    best_model_name = analytics.results.loc[analytics.results["F1-Score"].idxmax(), "Model"]
    best_model = analytics.models[best_model_name]
    
    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        feat_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        st.write(f"Top Features in {best_model_name}:")
        st.dataframe(feat_importance.head(10))

        # Recommendations
        st.subheader("📝 Recommendations based on top features")
        for idx, row in feat_importance.head(5).iterrows():
            feature = row["feature"]
            st.markdown(f"**{feature}**:")
            if "bmi" in feature.lower():
                st.markdown("- Encourage weight management programs and exercise")
            elif "glucose" in feature.lower():
                st.markdown("- Monitor blood glucose and dietary adjustments")
            elif "blood" in feature.lower() or "pressure" in feature.lower():
                st.markdown("- Regular blood pressure monitoring and lifestyle advice")
            else:
                st.markdown("- Regular health checkups and lifestyle improvement")

    # -----------------------------
    # 9️⃣ Download Results
    # -----------------------------
    st.subheader("📤 Download Metrics & Model")
    csv_buffer = io.BytesIO()
    analytics.results.to_csv(csv_buffer, index=False)
    st.download_button("Download Metrics CSV", data=csv_buffer.getvalue(), file_name="metrics.csv", mime="text/csv")

    model_buffer = io.BytesIO()
    joblib.dump(best_model, model_buffer)
    st.download_button("Download Best Model", data=model_buffer.getvalue(), file_name=f"best_model_{best_model_name}.joblib", mime="application/octet-stream")
