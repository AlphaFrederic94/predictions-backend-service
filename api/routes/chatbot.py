from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import both chatbot services
from services.chatbot import medical_chatbot_service

# Try to import the Docker-based service
try:
    from services.chatbot.docker_chatbot_service import docker_chatbot_service
    DOCKER_SERVICE_AVAILABLE = True
    print("Docker-based chatbot service is available")
except ImportError:
    DOCKER_SERVICE_AVAILABLE = False
    print("Docker-based chatbot service is not available, using fallback service")

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    system_message: Optional[str] = None
    max_tokens: Optional[int] = 256

class ChatResponse(BaseModel):
    response: str
    model: str

@router.post("/chat", response_model=ChatResponse)
async def chat_with_medical_bot(request: ChatRequest):
    """
    Chat with the medical AI assistant.

    This endpoint allows users to send messages to a medical AI assistant
    and receive responses related to medical topics.

    - **message**: The user's message or question
    - **system_message**: Optional system message to guide the AI's behavior
    - **max_tokens**: Maximum number of tokens in the response (default: 256)
    """
    try:
        # Use Docker-based service if available, otherwise use fallback
        if DOCKER_SERVICE_AVAILABLE and docker_chatbot_service.available:
            print(f"Using Docker-based chatbot service for query: {request.message[:50]}...")
            result = docker_chatbot_service.generate_response(
                user_message=request.message,
                system_message=request.system_message,
                max_tokens=request.max_tokens
            )
        else:
            print(f"Using fallback chatbot service for query: {request.message[:50]}...")
            result = medical_chatbot_service.generate_response(
                user_message=request.message,
                system_message=request.system_message,
                max_tokens=request.max_tokens
            )

        if "error" in result:
            # Return a fallback response instead of raising an exception
            return {
                "response": "I'm sorry, but the medical AI model is currently unavailable. This could be due to access restrictions or server issues. Please try again later or contact support if the issue persists.",
                "model": result.get("model", "unknown")
            }

        return result
    except Exception as e:
        # Return a fallback response instead of raising an exception
        print(f"Error in chat endpoint: {str(e)}")
        return {
            "response": "I apologize, but I encountered an error while processing your request. Please try again with a different question or contact support if the issue persists.",
            "model": "fallback-error-handler"
        }

@router.get("/status")
async def get_model_status():
    """
    Get the status of the medical chatbot model.

    Returns information about whether the model is initialized and ready to use.
    """
    # Initialize the fallback model if it's not already initialized
    if not medical_chatbot_service.initialized:
        medical_chatbot_service.initialize()

    # Initialize the Docker-based model if available
    if DOCKER_SERVICE_AVAILABLE and not docker_chatbot_service.initialized:
        docker_chatbot_service.initialize()

    # Prepare the response
    response = {
        "fallback_service": {
            "initialized": medical_chatbot_service.initialized,
            "available": medical_chatbot_service.available,
            "model_id": medical_chatbot_service.model_id
        }
    }

    # Add Docker service info if available
    if DOCKER_SERVICE_AVAILABLE:
        response["docker_service"] = {
            "initialized": docker_chatbot_service.initialized,
            "available": docker_chatbot_service.available,
            "model_id": docker_chatbot_service.model_id
        }

        # Set the active service
        response["active_service"] = "docker_service" if docker_chatbot_service.available else "fallback_service"
    else:
        response["active_service"] = "fallback_service"

    return response
