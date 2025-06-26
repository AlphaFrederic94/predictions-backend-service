# Medical Prediction Services: Quick Start Guide

This guide provides quick instructions to get the Medical Prediction Services up and running.

## Installation

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

## Testing the API

1. Test all API endpoints:
   ```bash
   python test_api.py
   ```

2. Test disease prediction with various symptom combinations:
   ```bash
   python test_disease_prediction.py
   ```

## Example Usage

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

## Documentation

For more detailed information:

- [Full Usage Guide](docs/USAGE_GUIDE.md)
- [Advanced Symptom Prediction Documentation](docs/ADVANCED_SYMPTOM_PREDICTION.md)
- [API Documentation](http://localhost:8000/docs) (when server is running)
