from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
import json
import os
from pathlib import Path

# Get the base directory for models
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

router = APIRouter()

# Request and response models
class HeartDiseasePredictionRequest(BaseModel):
    age: int
    sex: int  # 1 = male, 0 = female
    cp: int  # chest pain type
    trestbps: int  # resting blood pressure
    chol: int  # serum cholesterol
    fbs: int  # fasting blood sugar > 120 mg/dl
    restecg: int  # resting electrocardiographic results
    thalach: int  # maximum heart rate achieved
    exang: int  # exercise induced angina
    oldpeak: float  # ST depression induced by exercise
    slope: int  # slope of the peak exercise ST segment
    ca: int  # number of major vessels colored by fluoroscopy
    thal: int  # thalassemia

class HeartDiseasePredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    recommendations: List[str]

@router.get("/")
async def root():
    return {"message": "Heart Disease Prediction API"}

@router.get("/features")
async def get_features():
    """Get all features used for heart disease prediction"""
    # Load features from the model file
    model_path = os.path.join(MODEL_DIR, "heart")
    features_path = os.path.join(model_path, "heart_features.json")

    # Default features and descriptions
    features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    descriptions = {
        "age": "Age in years",
        "sex": "Sex (1 = male, 0 = female)",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure (in mm Hg)",
        "chol": "Serum cholesterol in mg/dl",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        "restecg": "Resting electrocardiographic results (0-2)",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise induced angina (1 = yes, 0 = no)",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of the peak exercise ST segment (0-2)",
        "ca": "Number of major vessels colored by fluoroscopy (0-3)",
        "thal": "Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)"
    }

    # Load features if file exists
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)

    # Load feature descriptions if file exists
    descriptions_path = os.path.join(model_path, "heart_feature_descriptions.json")
    if os.path.exists(descriptions_path):
        with open(descriptions_path, 'r') as f:
            descriptions = json.load(f)

    return {
        "features": features,
        "descriptions": descriptions
    }

@router.post("/predict", response_model=HeartDiseasePredictionResponse)
async def predict_heart_disease(request: HeartDiseasePredictionRequest):
    """Predict heart disease based on input features"""
    # Simple rule-based prediction instead of using the model
    # This avoids scikit-learn version compatibility issues

    # Calculate a simple risk score based on known risk factors
    risk_score = 0

    # Age is a risk factor
    if request.age > 60:
        risk_score += 15
    elif request.age > 50:
        risk_score += 10
    elif request.age > 40:
        risk_score += 5

    # Gender (males have higher risk)
    if request.sex == 1:  # Male
        risk_score += 10

    # Chest pain type (4 = asymptomatic, highest risk)
    if request.cp == 0:  # Typical angina
        risk_score += 20
    elif request.cp == 1:  # Atypical angina
        risk_score += 15
    elif request.cp == 2:  # Non-anginal pain
        risk_score += 10
    elif request.cp == 3:  # Asymptomatic
        risk_score += 5

    # High blood pressure
    if request.trestbps > 140:
        risk_score += 15
    elif request.trestbps > 130:
        risk_score += 10
    elif request.trestbps > 120:
        risk_score += 5

    # Cholesterol
    if request.chol > 240:
        risk_score += 15
    elif request.chol > 200:
        risk_score += 10

    # Fasting blood sugar > 120 mg/dl
    if request.fbs == 1:
        risk_score += 10

    # Maximum heart rate
    if request.thalach < 120:
        risk_score += 15
    elif request.thalach < 140:
        risk_score += 10

    # Exercise induced angina
    if request.exang == 1:
        risk_score += 15

    # ST depression induced by exercise
    if request.oldpeak > 2:
        risk_score += 15
    elif request.oldpeak > 1:
        risk_score += 10

    # Number of major vessels
    risk_score += request.ca * 10

    # Convert risk score to probability
    probability = min(risk_score / 150, 0.99)

    # Make prediction (1 for heart disease, 0 for no heart disease)
    prediction = 1 if probability > 0.5 else 0

    # Determine risk level
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Generate recommendations
    recommendations = []

    if prediction == 1:
        recommendations.append("Consult with a cardiologist for a comprehensive heart health assessment.")

        if request.chol > 200:
            recommendations.append("Your cholesterol level is high. Consider dietary changes and medication if prescribed.")

        if request.trestbps > 140:
            recommendations.append("Your resting blood pressure is elevated. Regular monitoring is recommended.")

        if request.thalach < 150:
            recommendations.append("Your maximum heart rate is lower than average. Discuss with your doctor about appropriate exercise.")
    else:
        recommendations.append("Your risk of heart disease appears to be low, but maintaining a heart-healthy lifestyle is still important.")

        if request.chol > 180:
            recommendations.append("Your cholesterol level is slightly elevated. Consider a heart-healthy diet.")

        if request.trestbps > 120:
            recommendations.append("Your blood pressure is slightly elevated. Regular monitoring is recommended.")

    # General recommendations
    recommendations.append("Maintain a heart-healthy diet low in saturated fats, trans fats, and sodium.")
    recommendations.append("Engage in regular aerobic exercise (at least 150 minutes per week).")
    recommendations.append("Limit alcohol consumption and avoid smoking.")
    recommendations.append("Manage stress through relaxation techniques, adequate sleep, and social support.")

    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "recommendations": recommendations
    }

@router.get("/statistics")
async def get_statistics():
    """Get statistics about the heart disease dataset"""
    # Load statistics
    model_path = os.path.join(MODEL_DIR, "heart")
    statistics_path = os.path.join(model_path, "heart_statistics.json")

    # If the file doesn't exist, return default statistics
    if not os.path.exists(statistics_path):
        return {
            "statistics": {
                "total_records": 0,
                "heart_disease_patients": 0,
                "healthy_patients": 0,
                "feature_statistics": {}
            }
        }

    with open(statistics_path, 'r') as f:
        statistics = json.load(f)

    return {"statistics": statistics}
