import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
try:
    with open('model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Make sure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# --- Web App Interface ---

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# Title of the app
st.title('Diabetes Risk Predictor 🩺')

st.write("""
Enter your medical details to predict your risk of diabetes.
This app uses a machine learning model to make a prediction based on the Pima Indians Diabetes Dataset.
""")

# --- User Input Fields in the Sidebar ---

st.sidebar.header('Patient Data')

def user_input_features():
    pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=120)
    blood_pressure = st.sidebar.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=80)
    skin_thickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input('Insulin Level (μU/mL)', min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input('BMI (Body Mass Index)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    
    # --- THIS IS THE NEWLY ADDED FEATURE ---
    diabetes_pedigree = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.47, format="%.3f")
    
    age = st.sidebar.number_input('Age (years)', min_value=1, max_value=120, value=35)
    
    # Create a dictionary for the data in the correct order
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree, # Added feature
        'Age': age
    }
    
    features = np.array(list(data.values())).reshape(1, -1)
    return features

# Get user input
input_features = user_input_features()

# --- Prediction and Display ---

st.subheader('Prediction Result')

# Create a button to make a prediction
if st.button('Predict My Risk'):
    # Scale the user input
    input_scaled = loaded_scaler.transform(input_features)
    
    # Make a prediction
    prediction = loaded_model.predict(input_scaled)
    prediction_proba = loaded_model.predict_proba(input_scaled)
    
    # Display the result
    st.markdown("---")
    if prediction[0] == 1:
        st.error('**Result: High Risk of Diabetes**')
    else:
        st.success('**Result: Low Risk of Diabetes**')
        
    st.subheader('Prediction Probability')
    st.write(f"The model predicts a **{prediction_proba[0][1]*100:.2f}%** probability of having diabetes.")