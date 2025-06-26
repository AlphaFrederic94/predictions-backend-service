# Medical Prediction Services

A comprehensive backend system for predicting various medical conditions using machine learning models. The system includes multiple prediction services with a layered architecture and RESTful API endpoints.

## Features

- **Disease Prediction**: Predict diseases based on symptoms with high accuracy
- **Multiple Models**: Standard, fine-tuned, and advanced prediction models
- **Comprehensive API**: RESTful endpoints for all prediction services
- **Detailed Results**: Get confidence scores, descriptions, precautions, and more
- **Top Predictions**: Get multiple possible diagnoses for ambiguous symptoms
- **Similar Symptoms**: Find symptoms based on partial text input
- **Model Metrics**: Access performance metrics for transparency
- **Medical Chatbot**: AI-powered medical assistant for answering healthcare questions

## Architecture

The system follows a layered architecture:

- **Data Layer**: Handles data loading and preprocessing
- **Service Layer**: Contains business logic and model inference
- **API Layer**: Exposes RESTful endpoints for client interaction
- **Model Layer**: Manages trained machine learning models

## Setup and Installation

### Option 1: Standard Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-prediction-services.git
   cd medical-prediction-services
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the models (optional, pre-trained models are included):
   ```bash
   python advanced_symptom_model.py  # Train the advanced symptom prediction model
   ```

5. Start the API server:
   ```bash
   uvicorn api.main:app --reload
   ```

### Option 2: Docker Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-prediction-services.git
   cd medical-prediction-services
   ```

2. Build and start the Docker container:
   ```bash
   # Using docker compose (recommended)
   docker compose up -d

   # Or using Docker directly
   docker build -t medical-prediction-api .
   docker run -d -p 8000:8000 --name medical-api medical-prediction-api
   ```

3. To stop the container:
   ```bash
   # Using docker compose
   docker compose down

   # Or using Docker directly
   docker stop medical-api
   docker rm medical-api
   ```

4. Access the API:
   - API Root: http://localhost:8000/
   - API Documentation: http://localhost:8000/docs
   - API Endpoints: See the API Endpoints section below

5. Docker Compose Features:
   - Automatic restart on failure
   - Volume mapping for persistent data storage
   - Environment variable configuration

The API will be available at http://localhost:8000

## API Endpoints

### Standard Endpoints

#### GET /api/symptoms/all
Returns a list of all possible symptoms

#### POST /api/symptoms/predict
Predicts disease based on provided symptoms

**Request Body:**
```json
{
  "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]
}
```

**Response:**
```json
{
  "predicted_disease": "Fungal infection",
  "confidence": 95.5,
  "description": "A fungal infection, also called mycosis, is a skin disease caused by a fungus.",
  "precautions": ["keep the affected area clean and dry", "use antifungal medications", "maintain good hygiene", "avoid sharing personal items"],
  "symptom_severity": {
    "itching": 1,
    "skin_rash": 3,
    "nodal_skin_eruptions": 4
  }
}
```

#### GET /api/symptoms/diseases
Returns a list of all diseases with their descriptions and precautions

### Advanced Endpoints

#### GET /api/advanced/symptoms/symptoms
Returns a list of all possible symptoms for the advanced model

#### POST /api/advanced/symptoms/predict
Predicts disease using the advanced model

**Request Body:**
```json
{
  "symptoms": ["fatigue", "high_fever", "vomiting", "headache", "nausea"]
}
```

**Response:**
```json
{
  "predicted_disease": "Malaria",
  "confidence": 87.5,
  "description": "An infectious disease caused by protozoan parasites from the Plasmodium family that can be transmitted by the bite of the Anopheles mosquito or by a contaminated needle or transfusion.",
  "precautions": ["Consult nearest hospital", "avoid oily food", "avoid non veg food", "keep mosquitos out"],
  "symptom_severity": {
    "fatigue": 4,
    "high_fever": 7,
    "vomiting": 5,
    "headache": 3,
    "nausea": 5
  },
  "matching_symptoms": ["fatigue", "high_fever", "vomiting", "headache", "nausea"],
  "total_disease_symptoms": 8
}
```

#### POST /api/advanced/symptoms/predict/top
Predicts top N diseases based on symptoms

**Request Body:**
```json
{
  "symptoms": ["fatigue", "headache", "high_fever", "nausea"],
  "top_n": 3
}
```

**Response:**
```json
{
  "top_predictions": [
    {
      "disease": "Malaria",
      "confidence": 37.50,
      "description": "An infectious disease caused by protozoan parasites...",
      "precautions": ["Consult nearest hospital", "avoid oily food", "avoid non veg food", "keep mosquitos out"],
      "matching_symptoms": ["headache", "high_fever", "nausea"],
      "total_disease_symptoms": 8
    },
    {
      "disease": "Typhoid",
      "confidence": 36.36,
      "description": "Typhoid fever is a bacterial infection...",
      "precautions": ["eat high calorie vegitables", "antiboitic therapy", "consult doctor", "medication"],
      "matching_symptoms": ["fatigue", "headache", "high_fever", "nausea"],
      "total_disease_symptoms": 11
    },
    {
      "disease": "Bronchial Asthma",
      "confidence": 33.33,
      "description": "Bronchial asthma is a medical condition...",
      "precautions": ["switch to loose cloths", "take deep breaths", "get away from trigger", "seek help"],
      "matching_symptoms": ["fatigue", "high_fever"],
      "total_disease_symptoms": 6
    }
  ],
  "symptom_severity": {
    "fatigue": 4,
    "headache": 3,
    "high_fever": 7,
    "nausea": 5
  }
}
```

#### POST /api/advanced/symptoms/symptoms/similar
Finds symptoms that match a partial string

**Request Body:**
```json
{
  "partial_symptom": "head",
  "limit": 5
}
```

**Response:**
```json
{
  "matches": ["headache", "blackheads"]
}
```

#### GET /api/advanced/symptoms/metrics
Returns model performance metrics

#### GET /api/advanced/symptoms/diseases
Returns all diseases with descriptions and symptoms

### Chatbot Endpoints

#### POST /api/chatbot/chat
Chat with the medical AI assistant

**Request Body:**
```json
{
  "message": "What are the differential diagnoses for a patient presenting with shortness of breath and chest pain?",
  "system_message": "You are an expert trained on healthcare and biomedical domain!",
  "max_tokens": 256
}
```

**Response:**
```json
{
  "response": "When a patient presents with shortness of breath and chest pain, several differential diagnoses should be considered, ranging from life-threatening to benign conditions. Here are the key differential diagnoses to consider:\n\n1. Acute Coronary Syndrome (ACS):\n   - Includes unstable angina and myocardial infarction (heart attack)\n   - Characterized by chest pressure, pain radiating to arm/jaw, diaphoresis, nausea\n   - Risk factors include age, smoking, diabetes, hypertension, hyperlipidemia\n\n2. Pulmonary Embolism (PE):\n   - Blood clot in the lungs\n   - Often presents with sudden onset of shortness of breath, pleuritic chest pain\n   - Risk factors include immobility, recent surgery, cancer, pregnancy\n\n3. Pneumothorax:\n   - Collapsed lung\n   - Sharp, sudden chest pain with respiratory distress\n   - More common in tall, thin individuals or those with underlying lung disease\n\n4. Pneumonia:\n   - Lung infection\n   - Presents with cough, fever, shortness of breath, and sometimes pleuritic chest pain\n\n5. Aortic Dissection:\n   - Tear in the wall of the aorta\n   - Severe, tearing chest pain that radiates to the back\n   - Risk factors include hypertension, connective tissue disorders\n\n6. Pericarditis:\n   - Inflammation of the pericardium (heart sac)\n   - Sharp chest pain that improves with sitting forward\n   - Often follows viral illness\n\n7. Anxiety/Panic Attack:\n   - Psychological cause of chest pain and shortness of breath\n   - Associated with feeling of impending doom, tingling, palpitations\n\n8. Gastroesophageal Reflux Disease (GERD):\n   - Acid reflux causing chest discomfort\n   - Often worse after meals or when lying down\n\n9. Musculoskeletal Pain:\n   - Pain from chest wall, ribs, or muscles\n   - Usually reproducible with movement or palpation\n\n10. Pleuritis:\n    - Inflammation of the pleura (lung lining)\n    - Sharp pain with breathing\n\nImmediate evaluation is necessary to rule out life-threatening causes like ACS, PE, and aortic dissection.",
  "model": "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"
}
```

#### GET /api/chatbot/status
Get the status of the medical chatbot model

## Testing the API

You can test the API using the provided test scripts:

```bash
# Test all API endpoints
python test_api.py

# Test disease prediction with various symptom combinations
python test_disease_prediction.py

# Test the medical chatbot
python test_chatbot.py
```

## Example Usage

```python
import requests

# Get list of symptoms
response = requests.get("http://localhost:8000/api/advanced/symptoms/symptoms")
all_symptoms = response.json()["symptoms"]
print(f"Available symptoms: {all_symptoms[:10]}...")

# Make a prediction
symptoms = ["fatigue", "high_fever", "vomiting", "headache", "nausea"]
response = requests.post(
    "http://localhost:8000/api/advanced/symptoms/predict",
    json={"symptoms": symptoms}
)
result = response.json()
print(f"Predicted disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']}%")
print(f"Description: {result['description']}")
print(f"Precautions: {result['precautions']}")
print(f"Matching symptoms: {result['matching_symptoms']}")

# Get top 3 predictions
response = requests.post(
    "http://localhost:8000/api/advanced/symptoms/predict/top",
    json={"symptoms": symptoms, "top_n": 3}
)
top_results = response.json()["top_predictions"]
for i, prediction in enumerate(top_results, 1):
    print(f"{i}. {prediction['disease']} ({prediction['confidence']}%)")

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

## Documentation

For more detailed information, see the following documentation:

- [Advanced Symptom Prediction Documentation](docs/ADVANCED_SYMPTOM_PREDICTION.md)
- [Docker Setup Documentation](docs/DOCKER.md)
- [Medical Chatbot Documentation](docs/MEDICAL_CHATBOT.md)

## Performance

The advanced symptom prediction model achieves the following performance metrics:

- **Accuracy**: 97.62%
- **Precision**: 98.81%
- **Recall**: 97.62%
- **F1 Score**: 97.62%
- **Cross-validation F1 Score**: 100%

## Docker Configuration

### Dockerfile

The application is containerized using Docker. The Dockerfile includes:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

For easier management, a docker-compose.yml file is provided:

```yaml
version: '3'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - ENVIRONMENT=production
    restart: unless-stopped
```

### Benefits of Docker Setup

- **Consistency**: Ensures the application runs the same way in every environment
- **Isolation**: Keeps the application and its dependencies isolated from the host system
- **Scalability**: Makes it easier to scale the application horizontally
- **Deployment**: Simplifies deployment to various environments
- **Resource Management**: Allows for better control of resource allocation

## Future Improvements

1. **User Interface**: Develop a frontend application to interact with the API
2. **Symptom Suggestions**: Implement a system to suggest additional symptoms to check
3. **Explanation System**: Provide more detailed explanations of predictions
4. **Severity Assessment**: Add functionality to assess condition severity
5. **Multilingual Support**: Add support for symptoms in multiple languages
6. **Personalization**: Consider patient demographics and medical history
7. **Containerization**: Improve Docker setup with multi-stage builds and optimized images
8. **CI/CD Pipeline**: Set up automated testing and deployment workflows
