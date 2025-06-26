from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from services.heart.finetuned_heart_service import FinetunedHeartService

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
    feature_importance: Optional[Dict[str, Any]] = None

# Dependency to get the heart service
def get_finetuned_heart_service():
    return FinetunedHeartService()

@router.get("/")
async def root():
    return {"message": "Fine-tuned Heart Disease Prediction API"}

@router.get("/features")
async def get_features(service: FinetunedHeartService = Depends(get_finetuned_heart_service)):
    """Get all features used for heart disease prediction"""
    return {
        "features": service.get_features(),
        "selected_features": service.get_selected_features()
    }

@router.post("/predict", response_model=HeartDiseasePredictionResponse)
async def predict_heart_disease(
    request: HeartDiseasePredictionRequest,
    service: FinetunedHeartService = Depends(get_finetuned_heart_service)
):
    """Predict heart disease based on input features using fine-tuned model"""
    # Convert request to dictionary
    data = request.dict()
    
    # Make prediction
    result = service.predict_heart_disease(data)
    return result

@router.get("/statistics")
async def get_statistics(service: FinetunedHeartService = Depends(get_finetuned_heart_service)):
    """Get statistics about the heart disease dataset"""
    return {"statistics": service.get_statistics()}

@router.get("/metrics")
async def get_model_metrics(service: FinetunedHeartService = Depends(get_finetuned_heart_service)):
    """Get model performance metrics"""
    return {"metrics": service.get_model_metrics()}
