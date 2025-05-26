# -*- coding: utf-8 -*-
"""
Created on Tue May 20 21:02:08 2025

@author: yasha
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# Loading the models

diabetes_model = pickle.load(open('C:/Users/yasha/OneDrive/Desktop/Multiple Disease Prediction System/saved models/diabetes_model.sav','rb'))
                                  
heart_disease_model = pickle.load(open('C:/Users/yasha/OneDrive/Desktop/Multiple Disease Prediction System/saved models/heart_disease_model.sav','rb'))                                  
                                       


#Sidebar for navigation

with st.sidebar:
    
    
    selected = st.sidebar.selectbox(
    'Multiple Disease Prediction:',
    ['Diabetes Prediction', 'Heart Disease Prediction']
    
)
                                   
# Diabetes Prediction Page

if(selected == 'Diabetes Prediction'):
    
    #page title
    st.title('Diabetes Prediction using ML')
    
    # Getting input data from user
    # Columns for input fields
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose level')
        
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
        
    with col2:
        Insulin = st.text_input('Insulin level')
        
    with col3:
        BMI = st.text_input('BMI value')
        
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the person')
    
    
    # Code for prediction
    diab_dignosis = ''
    
    # Creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        
        if(diab_prediction[0]==1):
            diab_dignosis = 'The person is Diabetic'
            
        else:
            diab_dignosis = 'The person is not Diabetic'
            
    st.success(diab_dignosis)
    
    
#Heart Disease prediction page

if(selected == 'Heart Disease Prediction'):
    #page title
    st.title('Heart Disease Prediction using ML')
    
    # Getting input data from user
    # Columns for input fields
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
    
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestrol in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca= st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal:0=normal;1=fixed defect;2=reversable defect')
        
        
    # Code for prediction
    heart_dignosis = ''
    
    # Creating a button for prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = diabetes_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        
        if(diab_prediction[0]==1):
            heart_dignosis = 'The person has heart disease'
            
        else:
            heart_dignosis = 'The person does not have heart disease'
            
    st.success(heart_dignosis)
        
    
