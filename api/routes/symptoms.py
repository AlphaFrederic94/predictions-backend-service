from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
from pydantic import BaseModel

from services.symptoms.advanced_symptom_service import AdvancedSymptomService

router = APIRouter()

# Dependency to get the service instance
def get_symptom_service():
    return AdvancedSymptomService()

# Request and response models
class SymptomPredictionRequest(BaseModel):
    symptoms: List[str]

class SymptomPredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    description: str
    precautions: List[str]
    symptom_severity: Dict[str, int]

@router.get("/")
async def root():
    return {"message": "Symptom Prediction API"}

@router.get("/all")
async def get_all_symptoms(service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Get all available symptoms"""
    return {"symptoms": service.get_all_symptoms()}

@router.post("/predict", response_model=SymptomPredictionResponse)
async def predict_disease(request: SymptomPredictionRequest, service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Predict disease based on symptoms"""
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Please provide at least one symptom")

    # Use the service to predict the disease
    result = service.predict_disease(request.symptoms)

    return result

@router.get("/diseases")
async def get_all_diseases(service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Get all diseases with their descriptions"""
    return {"diseases": service.get_all_diseases()}
