# Models Backend Integration Guide

This document explains how to integrate the Models Backend with the CareAI frontend application.

## Overview

The Models Backend provides several AI-powered prediction services:

1. **Symptoms Prediction**: Predict diseases based on symptoms
2. **Medical Chatbot**: AI-powered medical assistant for answering healthcare questions
3. **Heart Disease Prediction**: Predict heart disease risk based on health metrics
4. **Diabetes Prediction**: Predict diabetes risk based on health metrics

## Setup and Running

### Prerequisites

- Python 3.12 or higher
- Node.js 14 or higher
- npm or yarn

### Starting the Backend

1. Navigate to the project root directory
2. Run the start script:
   ```bash
   ./start_backend.sh
   ```
   This will:
   - Activate the virtual environment
   - Start the FastAPI server on port 8000

### Starting the Frontend

1. In a separate terminal, navigate to the project root directory
2. Start the React development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```
   This will start the frontend on port 3000

## API Endpoints

### Symptoms Prediction

- `GET /api/advanced/symptoms/symptoms`: Get all available symptoms
- `POST /api/advanced/symptoms/symptoms/similar`: Get symptoms that match a partial string
- `POST /api/advanced/symptoms/predict`: Predict disease based on symptoms
- `POST /api/advanced/symptoms/predict/top`: Predict top diseases based on symptoms
- `GET /api/advanced/symptoms/diseases`: Get all diseases with descriptions and symptoms

### Medical Chatbot

- `POST /api/chatbot/chat`: Chat with the medical AI assistant
- `GET /api/chatbot/status`: Get the status of the medical chatbot model

## Frontend Integration

The frontend integrates with the backend through API calls. The main integration points are:

1. **Symptoms Prediction Page**: `/predictions/symptoms`
   - Fetches available symptoms
   - Searches for symptoms as the user types
   - Sends selected symptoms for prediction
   - Displays prediction results

2. **Chatbot Component**: Available on all pages
   - Connects to the chatbot API
   - Sends user messages and displays responses
   - Shows typing indicator during API calls
   - Displays model status information

## Troubleshooting

If you encounter issues with the integration:

1. **Backend Connection Issues**:
   - Ensure the backend server is running on port 8000
   - Check for CORS errors in the browser console
   - Verify the API endpoints are correct

2. **Model Loading Issues**:
   - The backend uses fallback mechanisms if models fail to load
   - Check the backend console for error messages
   - Ensure the required model files are present in the appropriate directories

3. **Frontend Display Issues**:
   - Clear browser cache and reload
   - Check for JavaScript errors in the console
   - Verify that the API responses match the expected format

## Security Considerations

- The backend uses CORS to allow requests from the frontend
- In production, configure CORS to only allow specific origins
- API keys and sensitive information should be stored in environment variables
