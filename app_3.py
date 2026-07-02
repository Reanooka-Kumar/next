import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# App Title and Description
st.title("🎓 Student Placement Prediction Engine")
st.markdown("""
This app predicts whether a student will get **Placed** or **Not Placed** based on academic performance, projects, skills, and certifications.
""")

# ---------------------------------------------------------
# 1. Load Data and Train Model (Cached for Performance)
# ---------------------------------------------------------
@st.cache_resource
def train_placement_model():
    # Load dataset
    df = pd.read_csv("Placement_Prediction_data.csv")
    
    # Drop identifier columns safely
    df.drop(["Unnamed: 0", "StudentId"], axis=1, inplace=True, errors='ignore')
    
    # Map categorical text features to binary indicators
    df['Internship'] = df['Internship'].map({'Yes': 1, 'No': 0})
    df['Hackathon'] = df['Hackathon'].map({'Yes': 1, 'No': 0})
    df['PlacementStatus'] = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})
    
    # Define features and target variable
    X = df.drop('PlacementStatus', axis=1)
    y = df['PlacementStatus']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train robust Random Forest Classifier
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns.tolist()

# Load the model assets
try:
    model, scaler, feature_names = train_placement_model()
except FileNotFoundError:
    st.error("'Placement_Prediction_data.csv' not found. Please make sure the CSV file is in the same directory as this script.")
    st.stop()

# ---------------------------------------------------------
# 2. User Input Form Interface
# ---------------------------------------------------------
st.header("Enter Student Parameters")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        cgpa = st.slider("CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        tenth_perc = st.slider("10th Standard Percentage", min_value=35, max_value=100, value=75, step=1)
        twelfth_perc = st.slider("12th Standard Percentage", min_value=35, max_value=100, value=75, step=1)
        backlogs = st.number_input("Active Backlogs", min_value=0, max_value=20, value=0, step=1)
        comm_rating = st.slider("Communication Skills (1 - 5)", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
        skills_rating = st.slider("Technical Skills Rating (1 - 10)", min_value=1, max_value=10, value=6, step=1)

    with col2:
        major_projects = st.number_input("Number of Major Projects", min_value=0, max_value=10, value=1, step=1)
        mini_projects = st.number_input("Number of Mini Projects", min_value=0, max_value=10, value=1, step=1)
        workshops = st.number_input("Workshops / Certifications", min_value=0, max_value=20, value=1, step=1)
        
        st.write("") # Spacing element
        st.write("")
        internship_opt = st.radio("Completed an Internship?", options=["Yes", "No"], index=1, horizontal=True)
        hackathon_opt = st.radio("Participated in Hackathons?", options=["Yes", "No"], index=1, horizontal=True)
        
    # Submit button
    submit_button = st.form_submit_button(label="Predict Placement Status")

# ---------------------------------------------------------
# 3. Model Prediction and Result Visuals
# ---------------------------------------------------------
if submit_button:
    # Convert radio choices back to model binary integers
    internship = 1 if internship_opt == "Yes" else 0
    hackathon = 1 if hackathon_opt == "Yes" else 0
    
    # Construct structured input sample in exact training sequence
    input_sample = np.array([[
        cgpa, 
        major_projects, 
        workshops, 
        mini_projects, 
        skills_rating, 
        comm_rating, 
        internship, 
        hackathon, 
        twelfth_perc, 
        tenth_perc, 
        backlogs
    ]])
    
    # Scale input features
    input_sample_scaled = scaler.transform(input_sample)
    
    # Generate predictions
    prediction = model.predict(input_sample_scaled)[0]
    probability = model.predict_proba(input_sample_scaled)[0][1]
    
    st.write("---")
    st.header("Prediction Output Results")
    
    # Display results with metrics and alert callouts
    if prediction == 1:
        st.success("Status: PLACED")
        st.balloons()
    else:
        st.error("Status: NOT PLACED")
        
    # Interactive probability display indicator
    st.metric(label="Placement Probability Chance", value=f"{probability * 100:.2f}%")
    st.progress(float(probability))