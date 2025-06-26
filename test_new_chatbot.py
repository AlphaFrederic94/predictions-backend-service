import os
import requests
import json
from services.chatbot import medical_chatbot_service

# Set the API URL for when the API is running
API_URL = "http://localhost:8000/api/chatbot"

def test_chatbot_service_directly():
    """Test the chatbot service directly without going through the API."""
    print("\nTesting chatbot service directly...")

    # Initialize the service
    medical_chatbot_service.initialize()

    # Print the status
    print(f"Initialized: {medical_chatbot_service.initialized}")
    print(f"Available: {medical_chatbot_service.available}")
    print(f"Model ID: {medical_chatbot_service.model_id}")

    # Test generating a response
    result = medical_chatbot_service.generate_response(
        "What are the symptoms of diabetes?",
        max_tokens=256
    )

    print("\nDirect Service Response:")
    if "error" in result:
        print(f"Error: {result['error']}")
        return False
    else:
        print(f"Model: {result['model']}")
        print(f"Response: {result['response']}")
        if "note" in result:
            print(f"Note: {result['note']}")
        return True

def test_chatbot_status():
    """Test the chatbot status endpoint (when API is running)."""
    try:
        response = requests.get(f"{API_URL}/status")
        print("Status Response:", json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("Could not connect to the API. Is it running?")
        return False

def test_chatbot_chat():
    """Test the chatbot chat endpoint (when API is running)."""
    data = {
        "message": "What are the symptoms of diabetes?",
        "max_tokens": 256
    }

    try:
        response = requests.post(f"{API_URL}/chat", json=data)

        if response.status_code == 200:
            result = response.json()
            print("\nChat Response:")
            print(f"Model: {result['model']}")
            print(f"Response: {result['response']}")
            if "note" in result:
                print(f"Note: {result['note']}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.ConnectionError:
        print("Could not connect to the API. Is it running?")
        return False

if __name__ == "__main__":
    print("Testing Medical Chatbot...")

    # Test the service directly
    print("\n1. Testing chatbot service directly...")
    service_ok = test_chatbot_service_directly()
    print(f"Direct service test {'passed' if service_ok else 'failed'}")

    # Test API endpoints if needed
    print("\n2. Testing API endpoints...")

    status_ok = test_chatbot_status()
    print(f"Status endpoint test {'passed' if status_ok else 'failed'}")

    chat_ok = test_chatbot_chat()
    print(f"Chat endpoint test {'passed' if chat_ok else 'failed'}")
