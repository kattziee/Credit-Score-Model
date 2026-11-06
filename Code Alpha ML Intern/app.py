
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "Credit Score"
    / "model_outputs"
    / "random_forest_credit_model.joblib"
)

# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1120, #111827);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 15px;
    color: #94a3b8;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #e2e8f0;
}

div[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    border: 1px solid #93c5fd;
    color: white;
}

div[data-testid="stForm"] {
    background: #172033;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #334155;
}

div[data-testid="stProgressBar"] > div > div {
    background-color: #3b82f6;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.sidebar.title("💳 Credit Risk Analyzer")
st.sidebar.caption("Machine Learning Prediction System")

st.sidebar.divider()

st.sidebar.markdown("### About Project")
st.sidebar.write(
    "An intelligent credit default prediction "
    "system powered by Machine Learning."
)

st.sidebar.markdown("### Model Information")
st.sidebar.info("Random Forest Classifier")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model()
    features = list(model.feature_names_in_)
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# =====================================================
# 4. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">Credit Risk Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Credit Default Prediction & Financial Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# 5. INPUT FORM
# =====================================================

st.markdown(
    '<div class="section-title">Customer Financial Information</div>',
    unsafe_allow_html=True
)

with st.form("credit_prediction_form"):

    # PERSONAL INFORMATION
    st.markdown("### 👤 Personal Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        limit_bal = st.number_input(
            "Credit Limit",
            min_value=0,
            value=50000,
            step=5000
        )

    with c2:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with c3:
        sex = st.selectbox("Sex", [1, 2])

    c4, c5 = st.columns(2)

    with c4:
        education = st.selectbox(
            "Education",
            [1, 2, 3, 4, 5, 6, 0]
        )

    with c5:
        marriage = st.selectbox(
            "Marital Status",
            [1, 2, 3, 0]
        )

    st.divider()

    # PAYMENT HISTORY
    st.markdown("### 📅 Payment History")

    pay = {}

    payment_fields = [
        "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6"
    ]

    payment_cols = st.columns(3)

    for i, field in enumerate(payment_fields):
        with payment_cols[i % 3]:
            pay[field] = st.selectbox(
                field,
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                index=2
            )

    st.divider()

    # BILL AMOUNTS
    st.markdown("### 💰 Monthly Bill Amounts")

    bills = {}
    bill_cols = st.columns(3)

    for i in range(1, 7):
        with bill_cols[(i - 1) % 3]:
            bills[f"BILL_AMT{i}"] = st.number_input(
                f"Bill Amount {i}",
                value=0,
                step=1000
            )

    st.divider()

    # PAYMENT AMOUNTS
    st.markdown("### 💵 Monthly Payment Amounts")

    payments = {}
    payment_amount_cols = st.columns(3)

    for i in range(1, 7):
        with payment_amount_cols[(i - 1) % 3]:
            payments[f"PAY_AMT{i}"] = st.number_input(
                f"Payment Amount {i}",
                min_value=0,
                value=0,
                step=1000
            )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Analyze Credit Risk",
        use_container_width=True
    )

# =====================================================
# 6. PREDICTION
# =====================================================

if submitted:

    values = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay,
        **bills,
        **payments
    }

    # FEATURE ENGINEERING
    values["TOTAL_PAYMENT_DELAY"] = sum(
        max(0, pay[field]) for field in payment_fields
    )

    values["TOTAL_BILL_AMOUNT"] = sum(bills.values())

    values["TOTAL_PAYMENT_AMOUNT"] = sum(
        payments.values()
    )

    values["AVERAGE_BILL_AMOUNT"] = (
        sum(bills.values()) / 6
    )

    # CHECK MODEL FEATURES
    missing_features = [
        feature for feature in features
        if feature not in values
    ]

    if missing_features:
        st.error(
            "The following model features are missing: "
            + ", ".join(missing_features)
        )
        st.stop()

    # CREATE INPUT DATAFRAME IN TRAINING ORDER
    input_df = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features
    )

    # PREDICT
    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]

    default_probability = probabilities[
        list(model.classes_).index(1)
    ]

    non_default_probability = 1 - default_probability

    # =================================================
    # 7. RESULTS DASHBOARD
    # =================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Prediction Results</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Default Probability",
            f"{default_probability * 100:.2f}%"
        )

    with m2:
        st.metric(
            "Non-Default Probability",
            f"{non_default_probability * 100:.2f}%"
        )

    with m3:
        st.metric(
            "Predicted Class",
            "Default" if prediction == 1 else "No Default"
        )

    # =================================================
    # 8. RISK ASSESSMENT
    # =================================================

    st.markdown("### Credit Risk Assessment")

    st.progress(float(default_probability))

    st.caption(
        f"Estimated default probability: "
        f"{default_probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error(
            "⚠️ Higher predicted risk of credit default"
        )
    else:
        st.success(
            "✓ Model predicts no credit default"
        )

    if abs(default_probability - 0.5) <= 0.05:
        st.warning(
            "This prediction is close to the model's "
            "50% classification threshold."
        )

    st.caption(
        "The predicted class uses the model's classification "
        "threshold. Probability is an estimate, not a guarantee."
    )

    # =================================================
    # 9. CUSTOMER SUMMARY
    # =================================================

    st.divider()

    st.markdown("### 📋 Customer Financial Summary")

    summary_df = pd.DataFrame({
        "Financial Feature": features,
        "Customer Value": [
            values[feature] for feature in features
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # DOWNLOAD SUMMARY
    csv_data = summary_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Customer Report",
        data=csv_data,
        file_name="credit_risk_prediction_report.csv",
        mime="text/csv"
    )

    # =================================================
    # 10. FINAL NOTE
    # =================================================

    st.divider()

    st.info(
        "This project is developed for educational and "
        "machine learning demonstration purposes. "
        "It is not an official credit approval system."
    )

# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8;">
        <p><b>Credit Risk Analyzer</b></p>
        <p>Machine Learning Internship Project | CodeAlpha</p>
        <p>Built with Python, Scikit-learn, Pandas and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 11. MODEL PERFORMANCE & EVALUATION
# =====================================================

OUTPUT_DIR = BASE_DIR / "Credit Score" / "model_outputs"

st.divider()
st.markdown("## 📊 Machine Learning Model Evaluation")

# MODEL COMPARISON
comparison_files = list(
    OUTPUT_DIR.glob("*comparison*.csv")
)

if comparison_files:
    comparison_df = pd.read_csv(comparison_files[0])

    st.markdown("### Model Performance Comparison")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # Performance charts
    metric_columns = [
        col for col in
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        if col in comparison_df.columns
    ]

    if metric_columns and "Model" in comparison_df.columns:
        st.markdown("### Performance Metrics")

        chart_df = comparison_df.set_index("Model")[metric_columns]

        st.bar_chart(chart_df)

    # Download comparison report
    st.download_button(
        "⬇️ Download Model Comparison",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_report.csv",
        mime="text/csv"
    )
else:
    st.warning("Model comparison CSV not found.")

# ROC CURVE
st.markdown("### ROC Curve Comparison")

roc_files = list(OUTPUT_DIR.glob("*roc*.png"))

if roc_files:
    for roc_file in roc_files:
        st.image(
            str(roc_file),
            caption="ROC Curve Comparison",
            use_container_width=True
        )
else:
    st.info("ROC curve image not found.")

# CONFUSION MATRICES
st.markdown("### Confusion Matrices")

confusion_files = list(
    OUTPUT_DIR.glob("*confusion*.png")
)

if confusion_files:
    cols = st.columns(2)

    for i, image_file in enumerate(confusion_files):
        with cols[i % 2]:
            st.image(
                str(image_file),
                caption=image_file.stem.replace("_", " "),
                use_container_width=True
            )
else:
    st.info("Confusion matrix images not found.")

# FEATURE IMPORTANCE
st.markdown("### Random Forest Feature Importance")

importance_images = list(
    OUTPUT_DIR.glob("*importance*.png")
)

importance_csv = list(
    OUTPUT_DIR.glob("*importance*.csv")
)

if importance_images:
    for image_file in importance_images:
        st.image(
            str(image_file),
            caption="Random Forest Feature Importance",
            use_container_width=True
        )

if importance_csv:
    importance_df = pd.read_csv(importance_csv[0])

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Feature Importance",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name="feature_importance.csv",
        mime="text/csv"
    )

if not importance_images and not importance_csv:
    st.info("Feature importance files not found.")
