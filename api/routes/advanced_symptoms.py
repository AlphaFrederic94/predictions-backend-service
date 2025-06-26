from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from services.symptoms.advanced_symptom_service import AdvancedSymptomService

router = APIRouter()

# Request and response models
class SymptomPredictionRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptoms to predict disease from")

class SymptomPredictionResponse(BaseModel):
    predicted_disease: str = Field(..., description="Predicted disease name")
    confidence: float = Field(..., description="Confidence score (0-100)")
    description: str = Field(..., description="Disease description")
    precautions: List[str] = Field(..., description="Recommended precautions")
    symptom_severity: Dict[str, int] = Field(..., description="Severity of each symptom")
    matching_symptoms: List[str] = Field(..., description="Symptoms that match the disease")
    total_disease_symptoms: int = Field(..., description="Total number of symptoms for this disease")

class TopPredictionsRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptoms to predict diseases from")
    top_n: Optional[int] = Field(3, description="Number of top predictions to return")

class DiseaseInfo(BaseModel):
    disease: str = Field(..., description="Disease name")
    confidence: float = Field(..., description="Confidence score (0-100)")
    description: str = Field(..., description="Disease description")
    precautions: List[str] = Field(..., description="Recommended precautions")
    matching_symptoms: List[str] = Field(..., description="Symptoms that match the disease")
    total_disease_symptoms: int = Field(..., description="Total number of symptoms for this disease")

class TopPredictionsResponse(BaseModel):
    top_predictions: List[DiseaseInfo] = Field(..., description="List of top disease predictions")
    symptom_severity: Dict[str, int] = Field(..., description="Severity of each symptom")

class SimilarSymptomsRequest(BaseModel):
    partial_symptom: str = Field(..., description="Partial symptom name to search for")
    limit: Optional[int] = Field(10, description="Maximum number of results to return")

class SimilarSymptomsResponse(BaseModel):
    matches: List[str] = Field(..., description="List of matching symptoms")

# Dependency to get the service instance
def get_symptom_service():
    return AdvancedSymptomService()

@router.get("/")
async def root():
    return {"message": "Advanced Symptom Prediction API"}

@router.get("/symptoms")
async def get_all_symptoms(service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Get all available symptoms"""
    return {"symptoms": service.get_all_symptoms()}

@router.post("/predict", response_model=SymptomPredictionResponse)
async def predict_disease(request: SymptomPredictionRequest, service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Predict disease based on symptoms using advanced model"""
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Please provide at least one symptom")

    # Use the service to predict the disease
    result = service.predict_disease(request.symptoms)

    return result

@router.post("/predict/top", response_model=TopPredictionsResponse)
async def predict_top_diseases(request: TopPredictionsRequest, service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Predict top diseases based on symptoms using advanced model"""
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Please provide at least one symptom")

    # Use the service to predict top diseases
    result = service.predict_top_diseases(request.symptoms, request.top_n)

    return result

@router.get("/diseases")
async def get_all_diseases(service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Get all diseases with their descriptions and symptoms"""
    return {"diseases": service.get_all_diseases()}

@router.get("/disease/{disease_name}")
async def get_disease_info(disease_name: str, service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Get information about a specific disease"""
    result = service.get_disease_info(disease_name)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return {"disease_info": result}

@router.post("/symptoms/similar", response_model=SimilarSymptomsResponse)
async def get_similar_symptoms(request: SimilarSymptomsRequest, service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Get symptoms that match a partial string"""
    if not request.partial_symptom:
        raise HTTPException(status_code=400, detail="Please provide a partial symptom name")

    # Use the service to find similar symptoms
    matches = service.get_similar_symptoms(request.partial_symptom, request.limit)

    return {"matches": matches}

@router.get("/metrics")
async def get_model_metrics(service: AdvancedSymptomService = Depends(get_symptom_service)):
    """Get model performance metrics"""
    return {"metrics": service.get_model_metrics()}
