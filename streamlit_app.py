import streamlit as st
import pandas as pd
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*Inconsistent.*')

# Add notebook folder to Python path
sys.path.insert(0, str(Path(__file__).parent / 'notebook'))
from absenteeism_module import absenteeism_model, CustomScaler

# Page configuration
st.set_page_config(
    page_title="Absenteeism Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("🏢 Employee Absenteeism Predictor")
st.markdown("Predict if an employee is likely to have excessive absenteeism")

# Initialize the model (cache it to avoid reloading)
@st.cache_resource
def load_model():
    import os
    notebook_dir = Path(__file__).parent / 'notebook'
    original_dir = os.getcwd()
    os.chdir(notebook_dir)
    
    try:
        model = absenteeism_model('model', 'scaler')
    finally:
        os.chdir(original_dir)
    
    return model

# Load the model
model = load_model()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Employee Information")
    
    # Enhanced Reason for Absence with detailed descriptions
    st.markdown("**Reason for Absence:**")
    reason_options = {
        "None": {
            "value": 0,
            "description": "No absence",
            "details": "Employee is not absent"
        },
        "Group 1: Disease Related": {
            "value": 1,
            "description": "Related to diseases (codes 1-14)",
            "details": """
            - Includes: Infectious/parasitic diseases, neoplasms, metabolic disorders, mental illnesses,
            diseases of the nervous system, circulatory diseases, respiratory diseases, 
            digestive diseases, genitourinary diseases, skin conditions, and musculoskeletal disorders
            """
        },
        "Group 2: Pregnancy & Maternal": {
            "value": 2,
            "description": "Related to pregnancy and childbirth",
            "details": """
            - Includes: Pregnancy-related conditions, childbirth, and postpartum complications
            """
        },
        "Group 3: Poisoning & Other": {
            "value": 3,
            "description": "Poisoning or conditions not classified",
            "details": """
            - Includes: Poisoning, side effects of drugs, burns, and other injuries
            """
        },
        "Group 4: Light Reasons (22-28)": {
            "value": 4,
            "description": "Light/miscellaneous reasons for absence",
            "details": """
            - Includes: Medical consultations, dental procedures, physiotherapy, follow-up appointments,
            and other light medical reasons
            """
        }
    }
    
    reason = st.selectbox("Select Reason for Absence", list(reason_options.keys()))
    
    # Display detailed information with expander
    with st.expander("📖 Read More"):
        st.info(reason_options[reason]["details"])
    
    # Extract reason types
    reason_val = reason_options[reason]["value"]
    reason_1 = 1 if reason_val == 1 else 0
    reason_2 = 1 if reason_val == 2 else 0
    reason_3 = 1 if reason_val == 3 else 0
    reason_4 = 1 if reason_val == 4 else 0
    
    # Date input to extract month (instead of slider)
    st.markdown("**Select Date:**")
    selected_date = st.date_input("Pick a date", value=datetime.now())
    month_value = selected_date.month
    
    age = st.number_input("Age (years)", min_value=18, max_value=80, value=30)
    
    children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
    
    pets = st.slider("Number of Pets", min_value=0, max_value=10, value=0)

with col2:
    st.subheader("💼 Work & Health Information")
    
    transportation_expense = st.slider("Transportation Expense (USD)", min_value=0, max_value=500, value=100)
    
    body_mass_index = st.slider("Body Mass Index (BMI)", min_value=15.0, max_value=45.0, value=25.0, step=0.1)
    
    education_mapping = {
        "High School": 0,
        "Graduate": 1,
        "Post-graduate": 1,
        "Master or Doctor": 1
    }
    education_level = st.selectbox("Education Level", list(education_mapping.keys()))
    education = education_mapping[education_level]

# Make prediction
if st.button("🔮 Predict Absenteeism Risk", width='stretch'):
    # Create a single row dataframe with the input values
    input_data = pd.DataFrame({
        'reason_1': [reason_1],
        'reason_2': [reason_2],
        'reason_3': [reason_3],
        'reason_4': [reason_4],
        'month_value': [month_value],
        'transportation_expense': [transportation_expense],
        'age': [age],
        'body_mass_index': [body_mass_index],
        'education': [education],
        'children': [children],
        'pets': [pets]
    })
    
    # Scale the data
    scaled_data = model.scaler.transform(input_data).values
    model.data = scaled_data
    
    # Get prediction
    probability = model.reg.predict_proba(scaled_data)[0][1]
    prediction = model.reg.predict(scaled_data)[0]
    
    # Display results
    st.divider()
    st.subheader("📈 Prediction Results")
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction == 0:
            st.success("✅ **Low Absenteeism Risk**")
        else:
            st.warning("⚠️ **High Absenteeism Risk**")
    
    with col_result2:
        st.metric("Excessive Absenteeism Probability", f"{probability:.2%}")
    
    # Additional details
    st.divider()
    st.subheader("📊 Details")
    
    details_col1, details_col2, details_col3 = st.columns(3)
    with details_col1:
        st.metric("Risk Score", f"{probability:.4f}")
    with details_col2:
        st.metric("Prediction Class", "Excessive" if prediction == 1 else "Normal")
    with details_col3:
        st.metric("Confidence", f"{max(1-probability, probability):.2%}")
    
    # Show input summary
    st.divider()
    st.subheader("📝 Input Summary")
    st.dataframe(input_data, width='stretch')

# Footer
st.divider()
st.markdown("""
---
**How to use this predictor:**
1. Select the reason for absence from the dropdown (click "Read More" for details)
2. Pick a date to extract the month for prediction
3. Fill in the employee information on the left
4. Enter work and health details on the right
5. Click the "Predict Absenteeism Risk" button to get the prediction
6. View the results and probability score

*This model predicts the likelihood of excessive employee absenteeism based on various factors.*
""")

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    color: gray;
    background-color: white;
}
</style>

<div class="footer">
    <p>
        <b>Pichsereyvathanak KHY</b> |
        <a href="mailto:pichsereyvathanak.khy@gmail.com">Email</a> |
        <a href="https://github.com/PichSereyVathanak" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/pichsereyvathanak-khy">Linkedin</a>
    </p>
</div>
""", unsafe_allow_html=True)
