from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routes
from api.routes import symptoms, diabetes, heart
from api.routes import finetuned_symptoms, finetuned_heart
from api.routes import advanced_symptoms
from api.routes import chatbot
from common.exceptions.base_exception import BaseAPIException

# Create FastAPI app
app = FastAPI(
    title="Medical Prediction API",
    description="API for predicting various medical conditions",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(BaseAPIException)
async def base_exception_handler(request: Request, exc: BaseAPIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Include routers
# Standard routes
app.include_router(symptoms.router, prefix="/api/symptoms", tags=["Symptoms"])
app.include_router(diabetes.router, prefix="/api/diabetes", tags=["Diabetes"])
app.include_router(heart.router, prefix="/api/heart", tags=["Heart Disease"])

# Fine-tuned routes
app.include_router(finetuned_symptoms.router, prefix="/api/finetuned/symptoms", tags=["Fine-tuned Symptoms"])
app.include_router(finetuned_heart.router, prefix="/api/finetuned/heart", tags=["Fine-tuned Heart Disease"])

# Advanced routes
app.include_router(advanced_symptoms.router, prefix="/api/advanced/symptoms", tags=["Advanced Symptoms"])

# Chatbot routes
app.include_router(chatbot.router, prefix="/api/chatbot", tags=["Medical Chatbot"])

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Medical Prediction API",
        "documentation": "/docs",
        "standard_endpoints": {
            "symptoms": "/api/symptoms",
            "diabetes": "/api/diabetes",
            "heart": "/api/heart"
        },
        "finetuned_endpoints": {
            "symptoms": "/api/finetuned/symptoms",
            "heart": "/api/finetuned/heart"
        },
        "advanced_endpoints": {
            "symptoms": "/api/advanced/symptoms"
        },
        "chatbot_endpoints": {
            "chat": "/api/chatbot/chat",
            "status": "/api/chatbot/status"
        }
    }
