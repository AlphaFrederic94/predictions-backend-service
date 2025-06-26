# Medical Prediction Services: Usage Guide

This guide provides detailed instructions on how to set up, run, and use the Medical Prediction Services backend system.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Running the Server](#running-the-server)
3. [API Endpoints Overview](#api-endpoints-overview)
4. [Using the Symptom Prediction API](#using-the-symptom-prediction-api)
5. [Testing the System](#testing-the-system)
6. [Troubleshooting](#troubleshooting)
7. [Development and Extension](#development-and-extension)

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Installation Steps

1. Clone or download the repository:
   ```bash
   git clone https://github.com/yourusername/medical-prediction-services.git
   cd medical-prediction-services
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Train the models:
   ```bash
   # Train the advanced symptom prediction model
   python ml/training/advanced_symptom_model.py
   ```
   Note: Pre-trained models are included in the repository, so this step is optional.

## Running the Server

1. Start the API server:
   ```bash
   uvicorn api.main:app --reload
   ```

2. The server will start and be available at:
   ```
   http://localhost:8000
   ```

3. Access the interactive API documentation:
   ```
   http://localhost:8000/docs
   ```

## API Endpoints Overview

The system provides several API endpoints organized by prediction type:

### Standard Symptom Prediction

- `GET /api/symptoms/all` - Get all available symptoms
- `POST /api/symptoms/predict` - Predict disease based on symptoms
- `GET /api/symptoms/diseases` - Get all diseases with descriptions

### Advanced Symptom Prediction

- `GET /api/advanced/symptoms/symptoms` - Get all available symptoms
- `POST /api/advanced/symptoms/predict` - Predict disease using advanced model
- `POST /api/advanced/symptoms/predict/top` - Get top N disease predictions
- `POST /api/advanced/symptoms/symptoms/similar` - Find similar symptoms
- `GET /api/advanced/symptoms/metrics` - Get model performance metrics
- `GET /api/advanced/symptoms/diseases` - Get all diseases with descriptions
- `GET /api/advanced/symptoms/disease/{disease_name}` - Get specific disease info

## Using the Symptom Prediction API

### Basic Disease Prediction

To predict a disease based on symptoms:

```python
import requests

# Define symptoms
symptoms = ["fatigue", "high_fever", "vomiting", "headache", "nausea"]

# Make prediction request
response = requests.post(
    "http://localhost:8000/api/advanced/symptoms/predict",
    json={"symptoms": symptoms}
)

# Process result
if response.status_code == 200:
    result = response.json()
    print(f"Predicted disease: {result['predicted_disease']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Description: {result['description']}")
    print(f"Precautions:")
    for precaution in result['precautions']:
        print(f"- {precaution}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Getting Top Disease Predictions

For ambiguous symptoms, you can get multiple possible diagnoses:

```python
import requests

# Define symptoms
symptoms = ["fatigue", "headache", "high_fever", "nausea"]

# Make top predictions request
response = requests.post(
    "http://localhost:8000/api/advanced/symptoms/predict/top",
    json={"symptoms": symptoms, "top_n": 3}
)

# Process result
if response.status_code == 200:
    result = response.json()
    print("Top predictions:")
    for i, prediction in enumerate(result["top_predictions"], 1):
        print(f"{i}. {prediction['disease']} ({prediction['confidence']}%)")
        print(f"   Matching symptoms: {', '.join(prediction['matching_symptoms'])}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Finding Similar Symptoms

To help users find symptoms based on partial text:

```python
import requests

# Define partial symptom
partial = "head"

# Make similar symptoms request
response = requests.post(
    "http://localhost:8000/api/advanced/symptoms/symptoms/similar",
    json={"partial_symptom": partial, "limit": 5}
)

# Process result
if response.status_code == 200:
    result = response.json()
    print(f"Symptoms matching '{partial}':")
    for symptom in result["matches"]:
        print(f"- {symptom}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Testing the System

### Automated Testing

The system includes test scripts to verify functionality:

1. Test all API endpoints:
   ```bash
   python test_api.py
   ```

2. Test disease prediction with various symptom combinations:
   ```bash
   python test_disease_prediction.py
   ```

### Manual Testing with Swagger UI

You can test the API manually using the Swagger UI:

1. Start the API server: `uvicorn api.main:app --reload`
2. Open a browser and navigate to: `http://localhost:8000/docs`
3. Use the interactive documentation to test each endpoint:
   - Expand an endpoint by clicking on it
   - Click "Try it out"
   - Enter the required parameters
   - Click "Execute"
   - View the response

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check if the port is already in use
   - Verify that all dependencies are installed
   - Check for syntax errors in the code

2. **Model prediction errors**
   - Ensure the model files exist in the correct location
   - Verify that symptoms are spelled correctly
   - Check if the symptoms list is empty

3. **Missing dependencies**
   - Run `pip install -r requirements.txt` again
   - Check for any error messages during installation

### Logs

Check the server logs for detailed error information. When running with `--reload`, errors will be displayed in the terminal.

## Development and Extension

### Adding New Symptoms or Diseases

To add new symptoms or diseases:

1. Update the dataset files in `data/datasets/more_symptoms/`
2. Retrain the model using `python ml/training/advanced_symptom_model.py`

### Creating a New Prediction Service

To create a new prediction service:

1. Create a new service class in `services/`
2. Create new API routes in `api/routes/`
3. Add the router to `api/main.py`
4. Update the documentation

### Improving Model Performance

To improve model performance:

1. Collect more training data
2. Experiment with different algorithms in `ml/training/`
3. Tune hyperparameters
4. Implement more sophisticated feature engineering

## Conclusion

The Medical Prediction Services backend provides a robust API for disease prediction based on symptoms. With its layered architecture and comprehensive API, it can be easily extended and integrated into various healthcare applications.

For more detailed information about the advanced symptom prediction model, see [Advanced Symptom Prediction Documentation](ADVANCED_SYMPTOM_PREDICTION.md).
