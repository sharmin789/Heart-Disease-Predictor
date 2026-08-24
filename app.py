import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM DARK THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1220;
    color: white;
}


section[data-testid="stSidebar"] {
    background-color: #111a2b;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    color: #aab4c8;
    font-size: 16px;
    margin-bottom: 25px;
}

.metric-card {
    background-color: #111a2b;
    border: 1px solid #26344d;
    border-radius: 14px;
    padding: 18px;
    min-height: 105px;
}

.metric-title {
    color: #9ca8bc;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.result-card {
    background-color: #151d2d;
    border-radius: 14px;
    padding: 25px;
    margin-top: 10px;
}

.prediction-description {
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.6;
    margin-top: 15px;
}

.high-risk {
    color: #ff5b5b;
    font-size: 27px;
    font-weight: 800;
}

.low-risk {
    color: #55d98b;
    font-size: 27px;
    font-weight: 800;
}

.probability {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

div.stButton > button {
    width: 100%;
    background-color: #ef4444;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_data.csv")


model = load_model()
df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ❤️ Patient Input Features")

st.sidebar.caption(
    "Enter patient information to generate a machine-learning prediction."
)


# Age
age = st.sidebar.slider(
    "Age",
    min_value=20,
    max_value=90,
    value=52
)


# Sex
sex_label = st.sidebar.selectbox(
    "Sex",
    ["Male", "Female"]
)

sex = 1 if sex_label == "Male" else 0


# Chest Pain
cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

cp_label = st.sidebar.selectbox(
    "Chest Pain Type",
    list(cp_options.keys())
)

cp = cp_options[cp_label]


# Blood Pressure
trestbps = st.sidebar.slider(
    "Resting Blood Pressure",
    min_value=80,
    max_value=220,
    value=140
)


# Cholesterol
chol = st.sidebar.slider(
    "Cholesterol (mg/dl)",
    min_value=100,
    max_value=600,
    value=240
)


# Fasting Blood Sugar
fbs_label = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"]
)

fbs = 1 if fbs_label == "Yes" else 0


# Resting ECG
restecg_options = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

restecg_label = st.sidebar.selectbox(
    "Resting ECG",
    list(restecg_options.keys())
)

restecg = restecg_options[restecg_label]


# Maximum Heart Rate
thalach = st.sidebar.slider(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)


# Exercise Angina
exang_label = st.sidebar.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

exang = 1 if exang_label == "Yes" else 0


# Oldpeak
oldpeak = st.sidebar.slider(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=6.2,
    value=1.0,
    step=0.1
)


# Slope
slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope_label = st.sidebar.selectbox(
    "Slope of ST",
    list(slope_options.keys())
)

slope = slope_options[slope_label]


# Major vessels
ca = st.sidebar.selectbox(
    "Major Vessels (0-3)",
    [0, 1, 2, 3],
    index=1
)


# Thalassemia
thal_options = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}

thal_label = st.sidebar.selectbox(
    "Thalassemia",
    list(thal_options.keys())
)

thal = thal_options[thal_label]


# Prediction button
predict_button = st.sidebar.button(
    "❤️ Predict Heart Disease"
)


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning system for predicting the presence of heart disease based on patient features.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DATASET STATISTICS
# =========================================================

total_patients = len(df)

number_features = 13

number_classes = 2

disease_count = int(df["target"].sum())

no_disease_count = int(
    len(df) - disease_count
)

disease_percentage = (
    disease_count / len(df) * 100
)

no_disease_percentage = (
    no_disease_count / len(df) * 100
)


m1, m2, m3, m4, m5 = st.columns(5)


with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Patients</div>
            <div class="metric-value">{total_patients}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Features</div>
            <div class="metric-value">{number_features}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Classes</div>
            <div class="metric-value">{number_classes}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Disease</div>
            <div class="metric-value">{disease_percentage:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">No Disease</div>
            <div class="metric-value">{no_disease_percentage:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# DATA VISUALIZATION
# =========================================================

st.markdown(
    '<div class="section-title">📊 Dataset Visualization</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


# Target distribution
with col1:

    target_counts = pd.DataFrame({
        "Status": [
            "Disease",
            "No Disease"
        ],
        "Patients": [
            disease_count,
            no_disease_count
        ]
    })

    fig = px.pie(
        target_counts,
        names="Status",
        values="Patients",
        hole=0.55,
        title="Heart Disease Distribution"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# Age distribution
with col2:

    fig = px.histogram(
        df,
        x="age",
        nbins=15,
        title="Age Distribution"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# Cholesterol distribution
with col3:

    fig = px.histogram(
        df,
        x="chol",
        nbins=20,
        title="Cholesterol Distribution"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# CORRELATION + AGE/HEART RATE
# =========================================================

col1, col2 = st.columns(2)


# Correlation heatmap
with col1:

    st.subheader("🔥 Feature Correlation")

    correlation = df.corr(
        numeric_only=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        annot=True,
        fmt=".1f",
        ax=ax
    )

    ax.set_title(
        "Feature Correlation Heatmap"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# Age vs heart rate
with col2:

    st.subheader(
        "❤️ Age vs Maximum Heart Rate"
    )

    plot_df = df.copy()

    plot_df["Status"] = plot_df[
        "target"
    ].map({
        0: "No Disease",
        1: "Disease"
    })

    fig = px.scatter(
        plot_df,
        x="age",
        y="thalach",
        color="Status",
        hover_data=[
            "chol",
            "trestbps"
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PATIENT PREDICTION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 Patient Prediction</div>',
    unsafe_allow_html=True
)


patient_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})


if predict_button:

    prediction = model.predict(
        patient_data
    )[0]

    probability = model.predict_proba(
        patient_data
    )[0][1]


    col1, col2 = st.columns(2)


    with col1:

        if prediction == 1:

            st.markdown(
                """
                <div class="result-card">
                    <div class="high-risk">
                        ⚠️ HIGHER PREDICTED RISK
                    </div>

                    <div class="prediction-description">
    The machine-learning model predicts
    the presence of heart disease for this input.
</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-card">
                    <div class="low-risk">
                        ✓ LOWER PREDICTED RISK
                    </div>

                    <div class="prediction-description">
    The machine-learning model predicts
    the absence of heart disease for this input.
</div>
                </div>
                """,
                unsafe_allow_html=True
            )


    with col2:

        st.markdown(
            "### Predicted Probability"
        )

        st.markdown(
            f'<div class="probability">'
            f'{probability:.2%}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.progress(
            float(probability)
        )

        st.caption(
            "Educational machine-learning estimate, "
            "not a medical diagnosis."
        )


# =========================================================
# MODEL EVALUATION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Model Evaluation</div>',
    unsafe_allow_html=True
)


X = df.drop(
    "target",
    axis=1
)

y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_probability
)


e1, e2, e3, e4, e5 = st.columns(5)


with e1:
    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

with e2:
    st.metric(
        "Precision",
        f"{precision:.2%}"
    )

with e3:
    st.metric(
        "Recall",
        f"{recall:.2%}"
    )

with e4:
    st.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

with e5:
    st.metric(
        "ROC AUC",
        f"{auc:.2%}"
    )


# =========================================================
# CONFUSION MATRIX + ROC CURVE
# =========================================================

col1, col2 = st.columns(2)


# Confusion Matrix
with col1:

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=[
            "No Disease",
            "Disease"
        ],
        yticklabels=[
            "No Disease",
            "Disease"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ROC Curve
with col2:

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=3,
        label=f"AUC = {auc:.2f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Course Project • Heart Disease Prediction • "
    "Random Forest Classifier"
)

st.caption(
    "Educational machine-learning demonstration — "
    "not a medical diagnostic system."
)