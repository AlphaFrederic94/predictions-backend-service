# Medical Chatbot Service

This directory contains the implementation of the Medical Chatbot service for the CareAI application.

## Overview

The Medical Chatbot service provides a template-based approach to answering medical questions. It categorizes questions into different types (symptoms, prevention, treatment, general) and provides appropriate responses based on the category and the specific health topic detected in the question.

## Files

- `__init__.py`: Exports the singleton instance of the MedicalChatbotService
- `medical_chatbot_service.py`: Contains the implementation of the MedicalChatbotService class

## Implementation Details

The chatbot uses the following approach:

1. Determine if a question is medical-related using a comprehensive list of medical keywords
2. Categorize the question into one of four categories: symptoms, prevention, treatment, or general
3. Extract the main health topic from the question (e.g., diabetes, heart disease, headache)
4. Generate a response based on the category and topic, using predefined templates for common topics
5. Return the response along with the model ID

## Usage

The chatbot service is designed to be used as a singleton instance:

```python
from services.chatbot import medical_chatbot_service

# Initialize the service (if not already initialized)
medical_chatbot_service.initialize()

# Generate a response
response = medical_chatbot_service.generate_response(
    user_message="What are the symptoms of diabetes?",
    max_tokens=256
)

print(response["response"])
```

## API Integration

The chatbot service is integrated with the FastAPI application through the `/api/chatbot/chat` endpoint in `api/routes/chatbot.py`.

## Testing

You can test the chatbot service using the provided test scripts:

```bash
# Test the chatbot service directly
python test_new_chatbot.py

# Test the core functionality without importing the module
python simple_test.py
```

## Future Improvements

- Add more specialized responses for additional health topics
- Implement conversation history to allow for multi-turn conversations
- Add support for more languages
- Integrate with a real-time medical knowledge base for more accurate responses
