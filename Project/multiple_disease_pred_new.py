# -*- coding: utf-8 -*-
"""
Created on Tue May 20 21:02:08 2025

@author: yasha
"""

import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from risk_analyzer import train_risk_models, analyze_glucose_insulin, analyze_heart_parameters

# Loading the models
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))

# Load risk analysis models
diabetes_risk_model, heart_risk_model = train_risk_models()

# Sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Multiple Disease Prediction:',
        ['Diabetes Prediction', 'Heart Disease Prediction']
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose level (mg/dL)', help='Normal range: 70-140 mg/dL')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value (mm Hg)', help='Normal range: 90-120 mm Hg')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin level (µU/mL)', help='Normal range: 2-25 µU/mL')
    with col3:
        BMI = st.text_input('BMI value (kg/m²)', help='Normal range: 18.5-24.9 kg/m²')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the person')

    # Code for prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to float
            inputs = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
            diab_prediction = diabetes_model.predict([inputs])
            
            if diab_prediction[0] == 1:
                diagnosis = 'The person is Diabetic'
            else:
                diagnosis = 'The person is not Diabetic'
            
            # Display primary diagnosis
            st.success(diagnosis)
            
            # Risk Analysis
            st.subheader("Detailed Risk Analysis")
            
            # Get risk insights
            risk_insights = analyze_glucose_insulin(float(Glucose), float(Insulin))
            
            # Calculate overall risk probability
            risk_prob = diabetes_risk_model.predict_proba([[float(Glucose), float(Insulin)]])[0][1]
            risk_percentage = risk_prob * 100
            
            # Display overall risk
            st.write("Overall Risk Score:", f"{risk_percentage:.1f}%")
            
            # Create three columns for risk visualization
            col1, col2, col3 = st.columns(3)
            
            # Display risk meter
            if risk_percentage >= 70:
                col2.error(f"🔴 High Risk: {risk_percentage:.1f}%")
            elif risk_percentage >= 30:
                col2.warning(f"🟡 Moderate Risk: {risk_percentage:.1f}%")
            else:
                col2.success(f"🟢 Low Risk: {risk_percentage:.1f}%")
            
            # Display detailed insights
            st.subheader("Risk Factor Breakdown")
            for factor, level, message in risk_insights:
                if level == "HIGH":
                    st.error(f"🔴 {factor}: {message}")
                elif level == "MODERATE":
                    st.warning(f"🟡 {factor}: {message}")
                else:
                    st.success(f"🟢 {factor}: {message}")
        
        except ValueError:
            st.error('Please enter valid numeric values for all fields')

# Heart Disease prediction page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (0: Female, 1: Male)')
    with col3:
        cp = st.text_input('Chest Pain types (0-3)', help='0: No pain, 1: Mild, 2: Moderate, 3: Severe')
    
    with col1:
        trestbps = st.text_input('Resting Blood Pressure (mm Hg)', help='Normal range: 90-120 mm Hg')
    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dl)', help='Normal range: < 200 mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1: Yes, 0: No)')
    
    with col1:
        restecg = st.text_input('Resting ECG Results (0-2)', help='0: Normal, 1: ST-T abnormality, 2: LVH')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1: Yes, 0: No)')
    
    with col1:
        oldpeak = st.text_input('ST Depression by Exercise')
    with col2:
        slope = st.text_input('Slope of Peak Exercise ST')
    with col3:
        ca = st.text_input('Number of Major Vessels (0-3)')
    
    with col1:
        thal = st.text_input('Thalassemia', help='0: Normal, 1: Fixed defect, 2: Reversible defect')

    if st.button('Heart Disease Test Result'):
        try:
            # Convert inputs to float
            inputs = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            heart_prediction = heart_disease_model.predict([inputs])
            
            if heart_prediction[0] == 1:
                diagnosis = 'The person has heart disease'
            else:
                diagnosis = 'The person does not have heart disease'
            
            # Display primary diagnosis
            st.success(diagnosis)
            
            # Risk Analysis
            st.subheader("Detailed Risk Analysis")
            
            # Get risk insights
            risk_insights = analyze_heart_parameters(float(cp), float(exang))
            
            # Calculate overall risk probability
            risk_prob = heart_risk_model.predict_proba([[float(cp), float(exang)]])[0][1]
            risk_percentage = risk_prob * 100
            
            # Display overall risk
            st.write("Overall Risk Score:", f"{risk_percentage:.1f}%")
            
            # Create three columns for risk visualization
            col1, col2, col3 = st.columns(3)
            
            # Display risk meter
            if risk_percentage >= 70:
                col2.error(f"🔴 High Risk: {risk_percentage:.1f}%")
            elif risk_percentage >= 30:
                col2.warning(f"🟡 Moderate Risk: {risk_percentage:.1f}%")
            else:
                col2.success(f"🟢 Low Risk: {risk_percentage:.1f}%")
            
            # Display detailed insights
            st.subheader("Risk Factor Breakdown")
            for factor, level, message in risk_insights:
                if level == "HIGH":
                    st.error(f"🔴 {factor}: {message}")
                elif level == "MODERATE":
                    st.warning(f"🟡 {factor}: {message}")
                else:
                    st.success(f"🟢 {factor}: {message}")
        
        except ValueError:
            st.error('Please enter valid numeric values for all fields')
