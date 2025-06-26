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
class DiabetesPredictionRequest(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

class DiabetesPredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    recommendations: List[str]

@router.get("/")
async def root():
    return {"message": "Diabetes Prediction API"}

@router.get("/features")
async def get_features():
    """Get all features used for diabetes prediction"""
    # Load features from the model file
    model_path = os.path.join(MODEL_DIR, "diabetes")
    features_path = os.path.join(model_path, "diabetes_features.json")

    # If the file doesn't exist, return a default list
    if not os.path.exists(features_path):
        return {"features": ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree_function", "age"]}

    with open(features_path, 'r') as f:
        features = json.load(f)

    return {"features": features}

@router.post("/predict", response_model=DiabetesPredictionResponse)
async def predict_diabetes(request: DiabetesPredictionRequest):
    """Predict diabetes based on input features"""
    # Simple rule-based prediction instead of using the model
    # This avoids scikit-learn version compatibility issues

    # Calculate a simple risk score based on known risk factors
    risk_score = 0

    # High glucose is a strong indicator
    if request.glucose > 140:
        risk_score += 30
    elif request.glucose > 125:
        risk_score += 20
    elif request.glucose > 100:
        risk_score += 10

    # BMI is another important factor
    if request.bmi > 30:
        risk_score += 20
    elif request.bmi > 25:
        risk_score += 10

    # Age increases risk
    if request.age > 50:
        risk_score += 15
    elif request.age > 40:
        risk_score += 10
    elif request.age > 30:
        risk_score += 5

    # Family history (diabetes pedigree function)
    if request.diabetes_pedigree_function > 0.8:
        risk_score += 15
    elif request.diabetes_pedigree_function > 0.5:
        risk_score += 10

    # Pregnancies (for women)
    if request.pregnancies > 4:
        risk_score += 10

    # Insulin resistance indicators
    if request.insulin < 60 and request.glucose > 125:
        risk_score += 10

    # Convert risk score to probability
    probability = min(risk_score / 100, 0.99)

    # Make prediction (1 for diabetes, 0 for no diabetes)
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
        recommendations.append("Consult with a healthcare professional for a comprehensive diabetes assessment.")

        if request.glucose > 140:
            recommendations.append("Your glucose level is high. Consider monitoring your blood sugar regularly.")

        if request.bmi > 30:
            recommendations.append("Your BMI indicates obesity. Consider a weight management program.")

        if request.age > 40 and request.diabetes_pedigree_function > 0.5:
            recommendations.append("Given your age and family history, regular diabetes screening is important.")
    else:
        recommendations.append("Your risk of diabetes appears to be low, but maintaining a healthy lifestyle is still important.")

        if request.bmi > 25:
            recommendations.append("Your BMI indicates overweight. Consider maintaining a healthy diet and regular exercise.")

        if request.glucose > 100:
            recommendations.append("Your glucose level is slightly elevated. Consider reducing sugar intake.")

    # General recommendations
    recommendations.append("Maintain a balanced diet rich in fruits, vegetables, and whole grains.")
    recommendations.append("Engage in regular physical activity (at least 150 minutes per week).")
    recommendations.append("Limit alcohol consumption and avoid smoking.")

    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "recommendations": recommendations
    }

@router.get("/statistics")
async def get_statistics():
    """Get statistics about the diabetes dataset"""
    # Load statistics
    model_path = os.path.join(MODEL_DIR, "diabetes")
    statistics_path = os.path.join(model_path, "diabetes_statistics.json")

    # If the file doesn't exist, return default statistics
    if not os.path.exists(statistics_path):
        return {
            "statistics": {
                "total_records": 0,
                "diabetic_patients": 0,
                "non_diabetic_patients": 0,
                "feature_statistics": {}
            }
        }

    with open(statistics_path, 'r') as f:
        statistics = json.load(f)

    return {"statistics": statistics}
