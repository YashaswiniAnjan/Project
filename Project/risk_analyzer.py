import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_risk_models():
    # Load datasets
    diabetes_data = pd.read_csv('diabetes.csv')
    heart_data = pd.read_csv('heart.csv')
    
    # Train Diabetes Risk Model (using Glucose and Insulin)
    X_diabetes = diabetes_data[['Glucose', 'Insulin']]
    y_diabetes = diabetes_data['Outcome']
    diabetes_risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_risk_model.fit(X_diabetes, y_diabetes)
    
    # Train Heart Risk Model (using cp and exang)
    X_heart = heart_data[['cp', 'exang']]
    y_heart = heart_data['target']
    heart_risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
    heart_risk_model.fit(X_heart, y_heart)
    
    return diabetes_risk_model, heart_risk_model

def analyze_glucose_insulin(glucose, insulin):
    insights = []
    
    # Glucose Analysis
    if glucose > 140:
        risk_level = "HIGH"
        insights.append(("Glucose", risk_level, "Glucose levels are significantly elevated. This indicates high risk."))
    elif glucose > 100:
        risk_level = "MODERATE"
        insights.append(("Glucose", risk_level, "Glucose levels are moderately elevated. Monitor and consult doctor."))
    else:
        risk_level = "LOW"
        insights.append(("Glucose", risk_level, "Glucose levels are within normal range."))

    # Insulin Analysis
    if insulin > 166:
        risk_level = "HIGH"
        insights.append(("Insulin", risk_level, "Insulin levels are significantly elevated. This indicates high risk."))
    elif insulin > 100:
        risk_level = "MODERATE"
        insights.append(("Insulin", risk_level, "Insulin levels are moderately elevated. Monitor and consult doctor."))
    else:
        risk_level = "LOW"
        insights.append(("Insulin", risk_level, "Insulin levels are within normal range."))
        
    return insights

def analyze_heart_parameters(cp, exang):
    insights = []
    
    # Chest Pain Analysis
    if cp == 3:
        risk_level = "HIGH"
        insights.append(("Chest Pain", risk_level, "Severe chest pain type detected. Immediate medical attention recommended."))
    elif cp > 0:
        risk_level = "MODERATE"
        insights.append(("Chest Pain", risk_level, "Presence of chest pain. Consult with healthcare provider."))
    else:
        risk_level = "LOW"
        insights.append(("Chest Pain", risk_level, "No significant chest pain reported."))

    # Exercise Angina Analysis
    if exang == 1:
        risk_level = "HIGH"
        insights.append(("Exercise Angina", risk_level, "Exercise-induced angina present. This is a significant risk factor."))
    else:
        risk_level = "LOW"
        insights.append(("Exercise Angina", risk_level, "No exercise-induced angina reported."))
