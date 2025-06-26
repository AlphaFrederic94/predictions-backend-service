import requests
import json
from tabulate import tabulate

BASE_URL = "http://127.0.0.1:8000"

def predict_disease(symptoms, use_advanced=True):
    """Predict disease based on symptoms"""
    endpoint = f"{BASE_URL}/api/advanced/symptoms/predict" if use_advanced else f"{BASE_URL}/api/symptoms/predict"
    payload = {"symptoms": symptoms}
    response = requests.post(endpoint, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def predict_top_diseases(symptoms, top_n=3):
    """Predict top N diseases based on symptoms"""
    endpoint = f"{BASE_URL}/api/advanced/symptoms/predict/top"
    payload = {"symptoms": symptoms, "top_n": top_n}
    response = requests.post(endpoint, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def compare_predictions(symptoms):
    """Compare predictions between standard and advanced models"""
    standard_prediction = predict_disease(symptoms, use_advanced=False)
    advanced_prediction = predict_disease(symptoms, use_advanced=True)
    
    print(f"\n=== Prediction for symptoms: {', '.join(symptoms)} ===")
    
    if standard_prediction and advanced_prediction:
        # Create comparison table
        comparison = [
            ["Model", "Predicted Disease", "Confidence"],
            ["Standard", standard_prediction["predicted_disease"], f"{standard_prediction.get('confidence', 'N/A')}%"],
            ["Advanced", advanced_prediction["predicted_disease"], f"{advanced_prediction.get('confidence', 'N/A')}%"]
        ]
        
        print(tabulate(comparison, headers="firstrow", tablefmt="grid"))
        
        # Print additional details from advanced model
        print("\nAdvanced Model Additional Details:")
        print(f"Matching Symptoms: {', '.join(advanced_prediction.get('matching_symptoms', []))}")
        print(f"Total Disease Symptoms: {advanced_prediction.get('total_disease_symptoms', 'N/A')}")
        
        # Print disease description and precautions
        print(f"\nDisease: {advanced_prediction['predicted_disease']}")
        print(f"Description: {advanced_prediction['description']}")
        print("Precautions:")
        for i, precaution in enumerate(advanced_prediction['precautions'], 1):
            print(f"  {i}. {precaution}")
    
    return standard_prediction, advanced_prediction

def test_common_diseases():
    """Test prediction for common diseases"""
    test_cases = [
        # Common cold symptoms
        ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "swelled_lymph_nodes", "malaise", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "loss_of_smell"],
        
        # Diabetes symptoms
        ["fatigue", "obesity", "excessive_hunger", "increased_appetite", "polyuria", "weight_loss"],
        
        # Malaria symptoms
        ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
        
        # Pneumonia symptoms
        ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
        
        # Dengue symptoms
        ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain", "red_spots_over_body"],
    ]
    
    for symptoms in test_cases:
        compare_predictions(symptoms)

def test_top_predictions():
    """Test top N disease predictions"""
    test_cases = [
        # Ambiguous symptoms that could match multiple diseases
        ["fatigue", "headache", "high_fever", "nausea"],
        
        # Symptoms that could be either viral fever or influenza
        ["fatigue", "high_fever", "headache", "chills", "swelled_lymph_nodes", "malaise"],
        
        # Symptoms that could be either migraine or tension headache
        ["headache", "nausea", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck"],
    ]
    
    for symptoms in test_cases:
        print(f"\n=== Top 3 Predictions for symptoms: {', '.join(symptoms)} ===")
        result = predict_top_diseases(symptoms)
        
        if result:
            # Create table for top predictions
            table_data = [["Rank", "Disease", "Confidence", "Matching Symptoms"]]
            
            for i, prediction in enumerate(result["top_predictions"], 1):
                matching = ", ".join(prediction["matching_symptoms"])
                if len(matching) > 50:
                    matching = matching[:47] + "..."
                
                table_data.append([
                    i, 
                    prediction["disease"], 
                    f"{prediction['confidence']:.2f}%",
                    matching
                ])
            
            print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

def main():
    print("=== Testing Disease Prediction ===")
    print("\nTesting common disease predictions...")
    test_common_diseases()
    
    print("\n\nTesting top disease predictions...")
    test_top_predictions()

if __name__ == "__main__":
    main()
