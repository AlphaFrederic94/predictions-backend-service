# Advanced Symptom-Based Disease Prediction

This document provides detailed information about the advanced symptom-based disease prediction model implemented in this project.

## Overview

The advanced symptom prediction model uses state-of-the-art machine learning techniques to predict diseases based on patient symptoms. It offers higher accuracy, better handling of edge cases, and more comprehensive output than the standard and fine-tuned models.

## Key Features

- **Random Forest Classifier**: Uses an optimized Random Forest model for high accuracy and robustness
- **Advanced Feature Engineering**: Incorporates symptom severity, co-occurrence patterns, and symptom counts
- **Feature Selection**: Uses Recursive Feature Elimination with Cross-Validation (RFECV) to select the most informative features
- **Hyperparameter Optimization**: Uses RandomizedSearchCV for efficient hyperparameter tuning
- **Comprehensive Evaluation Metrics**: Provides detailed metrics including accuracy, precision, recall, F1 score, and ROC AUC
- **Pipeline Architecture**: Implements a scikit-learn pipeline for reproducible predictions
- **Multiple Prediction Options**: Provides both single disease prediction and top-N predictions for ambiguous cases
- **Symptom Matching**: Shows which symptoms match the predicted disease and their severity

## Model Architecture

The advanced model uses a Random Forest Classifier with optimized architecture:

1. **Core Model**:
   - Random Forest Classifier with optimized hyperparameters
   - Trained on comprehensive symptom dataset with 41 diseases and 132 symptoms

2. **Feature Engineering**:
   - Symptom severity weighting
   - Symptom count feature
   - Severity score feature
   - Co-occurrence features for top symptoms

3. **Feature Selection**:
   - RFECV to select the most informative features

4. **Hyperparameter Optimization**:
   - RandomizedSearchCV for efficient parameter tuning
   - Optimized for F1 score to balance precision and recall

## API Endpoints

### Prediction Endpoints

- **POST /api/advanced/symptoms/predict**
  - Predicts a disease based on provided symptoms
  - Returns the predicted disease, confidence score, description, precautions, and matching symptoms

- **POST /api/advanced/symptoms/predict/top**
  - Predicts top N diseases based on provided symptoms
  - Returns a list of potential diseases with confidence scores and details

### Information Endpoints

- **GET /api/advanced/symptoms/symptoms**
  - Returns a list of all available symptoms

- **GET /api/advanced/symptoms/diseases**
  - Returns a list of all diseases with descriptions and associated symptoms

- **GET /api/advanced/symptoms/disease/{disease_name}**
  - Returns detailed information about a specific disease

- **POST /api/advanced/symptoms/symptoms/similar**
  - Returns symptoms that match a partial string (for autocomplete functionality)

- **GET /api/advanced/symptoms/metrics**
  - Returns model performance metrics

## Usage Examples

### Predicting a Disease

```bash
curl -X POST -H "Content-Type: application/json" -d '{"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]}' http://localhost:8000/api/advanced/symptoms/predict
```

Response:
```json
{
  "predicted_disease": "Fungal infection",
  "confidence": 92.5,
  "description": "A fungal infection, also called mycosis, is a skin disease caused by a fungus.",
  "precautions": ["bath twice", "use detol or neem in bathing water", "keep the infected area dry", "use clean cloths"],
  "symptom_severity": {
    "itching": 1,
    "skin_rash": 3,
    "nodal_skin_eruptions": 4
  },
  "matching_symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
  "total_disease_symptoms": 4
}
```

### Getting Top Predictions

```bash
curl -X POST -H "Content-Type: application/json" -d '{"symptoms": ["headache", "nausea", "vomiting", "high_fever", "sweating"], "top_n": 3}' http://localhost:8000/api/advanced/symptoms/predict/top
```

Response:
```json
{
  "top_predictions": [
    {
      "disease": "Malaria",
      "confidence": 85.2,
      "description": "Malaria is a disease caused by a parasite. The parasite is transmitted to humans through the bites of infected mosquitoes.",
      "precautions": ["consult nearest hospital", "avoid oily food", "avoid non veg food", "keep mosquitos out"],
      "matching_symptoms": ["headache", "nausea", "vomiting", "high_fever", "sweating"],
      "total_disease_symptoms": 7
    },
    {
      "disease": "Typhoid",
      "confidence": 62.8,
      "description": "Typhoid fever is a bacterial infection that can spread throughout the body, affecting many organs.",
      "precautions": ["eat high calorie vegitables", "antiboitic therapy", "consult doctor", "medication"],
      "matching_symptoms": ["headache", "nausea", "vomiting", "high_fever"],
      "total_disease_symptoms": 9
    },
    {
      "disease": "Dengue",
      "confidence": 48.3,
      "description": "Dengue is a mosquito-borne viral infection causing a severe flu-like illness and, sometimes causing a potentially lethal complication called severe dengue.",
      "precautions": ["drink papaya leaf juice", "avoid fatty spicy food", "keep mosquitos away", "keep hydrated"],
      "matching_symptoms": ["headache", "nausea", "vomiting", "high_fever"],
      "total_disease_symptoms": 12
    }
  ],
  "symptom_severity": {
    "headache": 2,
    "nausea": 5,
    "vomiting": 5,
    "high_fever": 7,
    "sweating": 4
  }
}
```

### Finding Similar Symptoms

```bash
curl -X POST -H "Content-Type: application/json" -d '{"partial_symptom": "head", "limit": 5}' http://localhost:8000/api/advanced/symptoms/symptoms/similar
```

Response:
```json
{
  "matches": [
    "headache",
    "blackheads",
    "dizziness_and_headache",
    "mild_headache",
    "severe_headache"
  ]
}
```

## Performance Metrics

The advanced symptom prediction model achieves the following performance metrics:

- **Accuracy**: 97.62%
- **Precision**: 98.81%
- **Recall**: 97.62%
- **F1 Score**: 97.62%
- **Cross-validation F1 Score**: 100%

These metrics represent a significant improvement over the standard model, with near-perfect performance on the test dataset.

## Training the Model

To train the advanced symptom prediction model, run:

```bash
python ml/training/advanced_symptom_model.py
```

This will:
1. Load and preprocess the symptom dataset from data/datasets/more_symptoms
2. Perform feature engineering including symptom severity weighting
3. Create additional features like symptom count and co-occurrence patterns
4. Split the data into training and testing sets
5. Select features using RFECV
6. Optimize hyperparameters using RandomizedSearchCV
7. Train the Random Forest model
8. Evaluate the model on the test set
9. Compare with PCA-based dimensionality reduction
10. Save the model, metrics, and related data

## Implementation Details

The advanced symptom prediction model is implemented in the following files:

- `ml/training/advanced_symptom_model.py`: Training script for the advanced model
- `services/symptoms/advanced_symptom_service.py`: Service class for using the advanced model
- `api/routes/advanced_symptoms.py`: API routes for the advanced model

## Future Improvements

Potential future improvements for the advanced symptom prediction model include:

1. **User Interface**: Develop a frontend application to interact with the API
2. **Symptom Suggestions**: Implement a system to suggest additional symptoms to check based on initial inputs
3. **Explanation System**: Provide more detailed explanations of why certain diseases were predicted
4. **Severity Assessment**: Add functionality to assess the overall severity of the condition
5. **Multilingual Support**: Add support for symptoms in multiple languages
6. **Personalization**: Consider patient demographics and medical history in predictions
7. **Explainable AI Techniques**: Implementing SHAP values or LIME for better model interpretability
8. **Active Learning**: Implementing a feedback loop to continuously improve the model

## How to Test the System

The system includes comprehensive test scripts to verify functionality:

### Basic API Testing

To test all API endpoints and their responses:

```bash
python test_api.py
```

This script tests:
- Getting all symptoms
- Getting all diseases
- Disease prediction
- Similar symptoms search
- Model metrics retrieval

### Disease Prediction Testing

To test disease prediction with various symptom combinations:

```bash
python test_disease_prediction.py
```

This script tests:
- Common disease predictions (Cold, Diabetes, Malaria, Pneumonia, Dengue)
- Comparison between standard and advanced models
- Top-N predictions for ambiguous symptoms
- Detailed output including matching symptoms and confidence scores

### Manual Testing with Swagger UI

You can also test the API manually using the Swagger UI:

1. Start the API server: `uvicorn api.main:app --reload`
2. Open a browser and navigate to: `http://localhost:8000/docs`
3. Use the interactive documentation to test each endpoint

### Example: Testing Disease Prediction

```python
import requests

# Test with symptoms of Malaria
symptoms = ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"]
response = requests.post(
    "http://localhost:8000/api/advanced/symptoms/predict",
    json={"symptoms": symptoms}
)

result = response.json()
print(f"Predicted disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence']}%")
print(f"Matching symptoms: {result['matching_symptoms']}")
print(f"Total disease symptoms: {result['total_disease_symptoms']}")
print(f"Description: {result['description'][:100]}...")
```
