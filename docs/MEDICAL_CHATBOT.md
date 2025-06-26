# Medical Chatbot Documentation

This document provides detailed information about the Medical Chatbot feature in the Medical Prediction Services application.

## Overview

The Medical Chatbot (NyxV1) is an AI-powered assistant developed by Ngana Noa (MasterSilver) that specializes in healthcare and biomedical domains. It uses a specialized language model trained on medical and biomedical data to provide accurate, well-explained responses to medical-related questions.

NyxV1 draws upon deep understanding of anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, and other essential medical concepts. It uses precise medical terminology while ensuring answers remain accessible to a general audience.

## Model Information

- **Model**: aaditya/OpenBioLLM-Llama3-70B
- **Type**: Large Language Model (LLM)
- **Specialization**: Healthcare and biomedical domain
- **Size**: 70B parameters
- **Capabilities**:
  - Answering medical questions
  - Providing information about diseases, symptoms, and treatments
  - Explaining medical concepts
  - Suggesting differential diagnoses

## API Endpoints

### POST /api/chatbot/chat

This endpoint allows users to send messages to the medical chatbot and receive responses.

#### Request

```json
{
  "message": "What are the symptoms of diabetes?",
  "system_message": null,  // Optional: If not provided, the default NyxV1 system message will be used
  "max_tokens": 256
}
```

**Parameters**:
- `message` (required): The user's question or message
- `system_message` (optional): A system message to guide the AI's behavior
- `max_tokens` (optional): Maximum number of tokens in the response (default: 256)

#### Response

```json
{
  "response": "Regarding your question about diabetes: Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, extreme hunger, blurred vision, fatigue, and slow-healing sores. It's important to consult with a healthcare provider if you're experiencing these symptoms for proper diagnosis and treatment.",
  "model": "aaditya/OpenBioLLM-Llama3-70B"
}
```

**Response Fields**:
- `response`: The chatbot's answer to the user's question
- `model`: The ID of the model used to generate the response

### GET /api/chatbot/status

This endpoint returns information about the status of the chatbot model.

#### Response

```json
{
  "initialized": true,
  "available": false,
  "model_id": "aaditya/OpenBioLLM-Llama3-70B"
}
```

**Response Fields**:
- `initialized`: Whether the model has been loaded and is ready to use
- `available`: Whether the model is available for use
- `model_id`: The ID of the model

## Implementation Details

The Medical Chatbot is implemented using the following components:

1. **MedicalChatbotService**: A service that handles the categorization of medical questions and generation of appropriate responses
2. **Hugging Face Hub**: Used for authentication with the Hugging Face API
3. **FastAPI Endpoints**: Expose the chatbot functionality through a REST API

The implementation uses a template-based approach with specialized responses for common medical topics, which ensures reliable and consistent responses even when the model is not available.

## Usage Examples

### Python Example

```python
import requests

# Chat with the medical AI assistant
response = requests.post(
    "http://localhost:8000/api/chatbot/chat",
    json={
        "message": "What are the symptoms of diabetes?",
        "max_tokens": 256
    }
)
result = response.json()
print(f"Chatbot response: {result['response']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/chatbot/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What are the symptoms of diabetes?", "max_tokens": 256}'
```

## Limitations

- The chatbot is not a replacement for professional medical advice
- Responses are based on predefined templates for common medical topics
- The chatbot provides general information and cannot diagnose specific conditions
- The chatbot does not have access to real-time medical data or patient records
- The chatbot cannot provide personalized treatment recommendations

## Security and Privacy

- The chatbot does not store user messages or conversations
- All processing is done on the server, and no data is sent to external services
- The chatbot does not have access to personal or medical information unless explicitly provided in the message

## Future Improvements

- Implement conversation history to allow for multi-turn conversations
- Add support for image input to allow users to upload medical images
- Improve response time through model optimization
- Implement a feedback mechanism to improve the quality of responses
- Add support for multiple languages
