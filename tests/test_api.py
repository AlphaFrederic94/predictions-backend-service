import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_get_all_symptoms():
    """Test the get all symptoms endpoint"""
    response = requests.get(f"{BASE_URL}/api/symptoms/all")
    print("Get All Symptoms Response:")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Number of symptoms: {len(data['symptoms'])}")
    print(f"First 5 symptoms: {data['symptoms'][:5]}")
    print("\n")

def test_get_all_symptoms_advanced():
    """Test the get all symptoms endpoint (advanced)"""
    response = requests.get(f"{BASE_URL}/api/advanced/symptoms/symptoms")
    print("Get All Symptoms (Advanced) Response:")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Number of symptoms: {len(data['symptoms'])}")
    print(f"First 5 symptoms: {data['symptoms'][:5]}")
    print("\n")

def test_predict_disease():
    """Test the predict disease endpoint"""
    symptoms = ["fatigue", "high_fever", "vomiting", "headache", "nausea"]
    payload = {"symptoms": symptoms}
    response = requests.post(f"{BASE_URL}/api/symptoms/predict", json=payload)
    print("Predict Disease Response:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("\n")

def test_predict_disease_advanced():
    """Test the predict disease endpoint (advanced)"""
    symptoms = ["fatigue", "high_fever", "vomiting", "headache", "nausea"]
    payload = {"symptoms": symptoms}
    response = requests.post(f"{BASE_URL}/api/advanced/symptoms/predict", json=payload)
    print("Predict Disease (Advanced) Response:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("\n")

def test_get_similar_symptoms():
    """Test the get similar symptoms endpoint"""
    partial_symptom = "head"
    payload = {"partial_symptom": partial_symptom, "limit": 5}
    response = requests.post(f"{BASE_URL}/api/advanced/symptoms/symptoms/similar", json=payload)
    print("Get Similar Symptoms Response:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("\n")

def test_get_model_metrics():
    """Test the get model metrics endpoint"""
    response = requests.get(f"{BASE_URL}/api/advanced/symptoms/metrics")
    print("Get Model Metrics Response:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("\n")

def test_get_all_diseases():
    """Test the get all diseases endpoint"""
    response = requests.get(f"{BASE_URL}/api/advanced/symptoms/diseases")
    print("Get All Diseases Response:")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Number of diseases: {len(data['diseases'])}")
    print(f"First disease: {list(data['diseases'].keys())[0]}")
    print("\n")

def run_all_tests():
    """Run all tests"""
    test_get_all_symptoms()
    test_get_all_symptoms_advanced()
    test_predict_disease()
    test_predict_disease_advanced()
    test_get_similar_symptoms()
    test_get_model_metrics()
    test_get_all_diseases()

if __name__ == "__main__":
    run_all_tests()
