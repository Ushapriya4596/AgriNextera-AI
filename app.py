
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AgriNextera AI",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("AgriNextera_AI_Model.pkl")

model = load_model()

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #f4f8f3;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #174d32;
}
.subtitle {
    color: #557361;
    font-size: 17px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce9df;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🌱 AgriNextera AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Smart Precision Irrigation & Crop Water Intelligence System</div>',
    unsafe_allow_html=True
)

st.write("")
st.success("Turning field data into intelligent agricultural insights.")

# Overview metrics
c1, c2, c3 = st.columns(3)

c1.metric("Dataset Records", "150")
c2.metric("Model", "Random Forest")
c3.metric("Holdout Accuracy", "96.67%")

st.caption(
    "Accuracy is from one small 30-record test set; "
    "it does not guarantee real-world performance."
)

st.divider()

predict_tab, insights_tab, about_tab = st.tabs([
    "🌾 Field Predictor",
    "📊 Model Insights",
    "ℹ️ About"
])

# Predictor
with predict_tab:
    st.header("Field Condition Analysis")
    st.write("Enter your field measurements below.")

    with st.form("field_form"):
        col1, col2 = st.columns(2)

        with col1:
            crop_type = st.selectbox("Crop Type Code", [1, 2])
            crop_days = st.number_input(
                "Crop Age (Days)", 0, 365, 30
            )
            soil_moisture = st.number_input(
                "Soil Moisture", min_value=0.0, value=300.0
            )

        with col2:
            soil_temperature = st.number_input(
                "Soil Temperature", value=25.0
            )
            temperature = st.number_input(
                "Air Temperature", value=30.0
            )
            humidity = st.number_input(
                "Humidity (%)", 0.0, 100.0, 60.0
            )

        submitted = st.form_submit_button(
            "🌱 Analyse Field Conditions",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([{
            "CropType": crop_type,
            "CropDays": crop_days,
            "Soil Moisture": soil_moisture,
            "Soil Temperature": soil_temperature,
            "Temperature": temperature,
            "Humidity": humidity
        }])

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("🤖 Model Prediction")

        st.info(f"Predicted class: **Class {prediction}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            probability_data = pd.DataFrame({
                "Class": [
                    f"Class {c}" for c in model.classes_
                ],
                "Model Probability": probabilities
            })

            st.write("Prediction probability by class")
            st.bar_chart(
                probability_data.set_index("Class")
            )

        st.warning(
            "Class meanings have not yet been verified. "
            "Do not use this experimental prediction alone "
            "to make irrigation decisions."
        )

# Model insights
with insights_tab:
    st.header("Understanding the AI Model")

    st.write(
        "The model uses field measurements to identify patterns "
        "associated with the dataset's irrigation labels."
    )

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": [
                "Crop Type",
                "Crop Days",
                "Soil Moisture",
                "Soil Temperature",
                "Air Temperature",
                "Humidity"
            ],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(
            importance.set_index("Feature")
        )

        st.caption(
            "Feature importance shows how the model uses features. "
            "It does not prove cause and effect."
        )

    st.subheader("Model Evaluation")
    st.write("Holdout test accuracy: **96.67%**")
    st.write("Test records: **30**")
    st.write("Correct predictions: **29 out of 30**")
    st.write("Cross-validation average accuracy: **94%**")

# About
with about_tab:
    st.header("About AgriNextera AI")

    st.write("""
    AgriNextera AI is an experimental smart-agriculture
    prototype designed to explore machine learning for
    irrigation-related dataset classification.

    **Technology Stack**
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest
    - Streamlit

    **Dataset:** 150 records.

    **Current limitation:** The dataset is small, and the
    meanings of its class labels and crop codes need verification.
    This prototype is not a substitute for expert agricultural advice.
    """)

st.divider()
st.caption("AgriNextera AI | Where Agriculture Meets Intelligence 🌱")
