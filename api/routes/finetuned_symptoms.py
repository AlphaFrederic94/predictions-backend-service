from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from services.symptoms.finetuned_symptom_service import FinetunedSymptomService

router = APIRouter()

# Request and response models
class SymptomPredictionRequest(BaseModel):
    symptoms: List[str]

class SymptomPredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    description: str
    precautions: List[str]
    symptom_severity: Dict[str, int]

class TopPredictionsRequest(BaseModel):
    symptoms: List[str]
    top_n: Optional[int] = 3

class TopPredictionsResponse(BaseModel):
    top_predictions: List[Dict[str, Any]]
    symptom_severity: Dict[str, int]

# Dependency to get the symptom service
def get_finetuned_symptom_service():
    return FinetunedSymptomService()

@router.get("/")
async def root():
    return {"message": "Fine-tuned Symptom Prediction API"}

@router.get("/all")
async def get_all_symptoms(service: FinetunedSymptomService = Depends(get_finetuned_symptom_service)):
    """Get all available symptoms"""
    return {"symptoms": service.get_all_symptoms()}

@router.post("/predict", response_model=SymptomPredictionResponse)
async def predict_disease(
    request: SymptomPredictionRequest,
    service: FinetunedSymptomService = Depends(get_finetuned_symptom_service)
):
    """Predict disease based on symptoms using fine-tuned model"""
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Please provide at least one symptom")
    
    result = service.predict_disease(request.symptoms)
    return result

@router.post("/predict/top", response_model=TopPredictionsResponse)
async def predict_top_diseases(
    request: TopPredictionsRequest,
    service: FinetunedSymptomService = Depends(get_finetuned_symptom_service)
):
    """Predict top diseases based on symptoms using fine-tuned model"""
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Please provide at least one symptom")
    
    result = service.predict_top_diseases(request.symptoms, request.top_n)
    return result

@router.get("/diseases")
async def get_all_diseases(service: FinetunedSymptomService = Depends(get_finetuned_symptom_service)):
    """Get all diseases with their descriptions"""
    return {"diseases": service.get_all_diseases()}

@router.get("/disease/{disease_name}")
async def get_disease_info(
    disease_name: str,
    service: FinetunedSymptomService = Depends(get_finetuned_symptom_service)
):
    """Get information about a specific disease"""
    disease_info = service.get_disease_info(disease_name)
    if "error" in disease_info:
        raise HTTPException(status_code=404, detail=disease_info["error"])
    return {"disease_info": disease_info}

@router.get("/disease/{disease_name}/symptoms")
async def get_disease_symptoms(
    disease_name: str,
    service: FinetunedSymptomService = Depends(get_finetuned_symptom_service)
):
    """Get symptoms associated with a specific disease"""
    symptoms = service.get_disease_symptoms(disease_name)
    if not symptoms:
        raise HTTPException(status_code=404, detail=f"Disease '{disease_name}' not found or has no associated symptoms")
    return {
        "disease": disease_name,
        "symptoms": symptoms
    }

@router.get("/metrics")
async def get_model_metrics(service: FinetunedSymptomService = Depends(get_finetuned_symptom_service)):
    """Get model performance metrics"""
    return {"metrics": service.get_model_metrics()}
