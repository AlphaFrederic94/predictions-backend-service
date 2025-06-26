import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any

class FinetunedSymptomService:
    """Service for symptom-based disease prediction using fine-tuned model"""

    def __init__(self):
        """Initialize the service by loading the model and related data"""
        # Define paths
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_path, "ml", "models", "symptoms")

        # Load fine-tuned model if available, otherwise use standard model
        finetuned_model_file = os.path.join(model_path, "finetuned_model.pkl")
        standard_model_file = os.path.join(model_path, "rf_model.pkl")

        if os.path.exists(finetuned_model_file):
            with open(finetuned_model_file, 'rb') as f:
                self.model = pickle.load(f)
            print("Using fine-tuned symptom prediction model")
        elif os.path.exists(standard_model_file):
            with open(standard_model_file, 'rb') as f:
                self.model = pickle.load(f)
            print("Using standard symptom prediction model as fallback")
        else:
            self.model = None
            print(f"Warning: No model file found at {finetuned_model_file} or {standard_model_file}")

        # Load feature selector if available
        selector_file = os.path.join(model_path, "feature_selector.pkl")
        if os.path.exists(selector_file):
            with open(selector_file, 'rb') as f:
                self.feature_selector = pickle.load(f)
        else:
            self.feature_selector = None
            print(f"Warning: Feature selector file not found at {selector_file}")

        # Load label encoder
        encoder_file = os.path.join(model_path, "label_encoder.pkl")
        if os.path.exists(encoder_file):
            with open(encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            self.label_encoder = None
            print(f"Warning: Label encoder file not found at {encoder_file}")

        # Load symptoms list
        symptoms_file = os.path.join(model_path, "symptoms_list.json")
        if os.path.exists(symptoms_file):
            with open(symptoms_file, 'r') as f:
                self.symptoms_list = json.load(f)
        else:
            self.symptoms_list = []
            print(f"Warning: Symptoms list file not found at {symptoms_file}")

        # Load disease-symptom mapping
        mapping_file = os.path.join(model_path, "disease_symptom_map.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.disease_symptom_map = json.load(f)
        else:
            self.disease_symptom_map = {}
            print(f"Warning: Disease-symptom mapping file not found at {mapping_file}")

        # Load disease data
        disease_data_file = os.path.join(model_path, "disease_data.json")
        if os.path.exists(disease_data_file):
            with open(disease_data_file, 'r') as f:
                self.disease_data = json.load(f)
        else:
            self.disease_data = {}
            print(f"Warning: Disease data file not found at {disease_data_file}")

        # Load symptom severity
        severity_file = os.path.join(model_path, "symptom_severity.json")
        if os.path.exists(severity_file):
            with open(severity_file, 'r') as f:
                self.symptom_severity = json.load(f)
        else:
            self.symptom_severity = {}
            print(f"Warning: Symptom severity file not found at {severity_file}")

        # Load model metrics
        metrics_file = os.path.join(model_path, "model_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.model_metrics = json.load(f)
        else:
            self.model_metrics = {}
            print(f"Warning: Model metrics file not found at {metrics_file}")

    def get_all_symptoms(self) -> List[str]:
        """Get all available symptoms"""
        return self.symptoms_list

    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """Predict disease based on symptoms"""
        if not self.model or not self.label_encoder or not self.symptoms_list:
            return self._fallback_prediction(symptoms)

        # Create feature vector
        feature_vector = np.zeros(len(self.symptoms_list))
        for symptom in symptoms:
            if symptom in self.symptoms_list:
                idx = self.symptoms_list.index(symptom)
                # Apply severity weight if available
                severity = self.symptom_severity.get(symptom, 1)
                feature_vector[idx] = severity

        # Add symptom count feature if it was used in training
        if 'symptom_count' in self.symptoms_list:
            symptom_count_idx = self.symptoms_list.index('symptom_count')
            feature_vector[symptom_count_idx] = len(symptoms)
        # We don't add extra features to match the model's expected feature count

        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)

        # Apply feature selection if available
        if self.feature_selector:
            try:
                feature_vector = self.feature_selector.transform(feature_vector)
            except Exception as e:
                print(f"Warning: Error applying feature selection: {e}")
                # If feature selection fails, continue with the original features

        # Make prediction
        try:
            prediction_idx = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            confidence = probabilities[prediction_idx] * 100

            # Get disease name
            disease = self.label_encoder.inverse_transform([prediction_idx])[0]
        except Exception as e:
            print(f"Warning: Error making prediction: {e}")
            return self._fallback_prediction(symptoms)

        # Get disease information
        disease_info = self.disease_data.get(disease, {})
        description = disease_info.get('description', 'No description available')
        precautions = disease_info.get('precautions', [])

        # Get severity of provided symptoms
        symptoms_with_severity = {s: self.symptom_severity.get(s, 0) for s in symptoms}

        return {
            "predicted_disease": disease,
            "confidence": round(confidence, 2),
            "description": description,
            "precautions": precautions,
            "symptom_severity": symptoms_with_severity
        }

    def predict_top_diseases(self, symptoms: List[str], top_n: int = 3) -> Dict[str, Any]:
        """Predict top N diseases based on symptoms"""
        if not self.model or not self.label_encoder or not self.symptoms_list:
            return self._fallback_top_predictions(symptoms, top_n)

        # Create feature vector
        feature_vector = np.zeros(len(self.symptoms_list))
        for symptom in symptoms:
            if symptom in self.symptoms_list:
                idx = self.symptoms_list.index(symptom)
                # Apply severity weight if available
                severity = self.symptom_severity.get(symptom, 1)
                feature_vector[idx] = severity

        # Add symptom count feature if it was used in training
        if 'symptom_count' in self.symptoms_list:
            symptom_count_idx = self.symptoms_list.index('symptom_count')
            feature_vector[symptom_count_idx] = len(symptoms)
        # We don't add extra features to match the model's expected feature count

        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)

        # Apply feature selection if available
        if self.feature_selector:
            try:
                feature_vector = self.feature_selector.transform(feature_vector)
            except Exception as e:
                print(f"Warning: Error applying feature selection: {e}")
                # If feature selection fails, continue with the original features

        # Get probabilities for all diseases
        try:
            probabilities = self.model.predict_proba(feature_vector)[0]

            # Get top N predictions
            top_indices = np.argsort(probabilities)[::-1][:top_n]
            top_predictions = []

            for idx in top_indices:
                disease = self.label_encoder.inverse_transform([idx])[0]
                confidence = probabilities[idx] * 100
                disease_info = self.disease_data.get(disease, {})

                top_predictions.append({
                    "disease": disease,
                    "confidence": round(confidence, 2),
                    "description": disease_info.get('description', 'No description available'),
                    "precautions": disease_info.get('precautions', [])
                })
        except Exception as e:
            print(f"Warning: Error making top predictions: {e}")
            return self._fallback_top_predictions(symptoms, top_n)

        # Get severity of provided symptoms
        symptoms_with_severity = {s: self.symptom_severity.get(s, 0) for s in symptoms}

        return {
            "top_predictions": top_predictions,
            "symptom_severity": symptoms_with_severity
        }

    def get_all_diseases(self) -> Dict[str, Dict[str, Any]]:
        """Get all diseases with their descriptions"""
        return self.disease_data

    def get_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """Get information about a specific disease"""
        if disease_name in self.disease_data:
            return self.disease_data[disease_name]
        return {"error": f"Disease '{disease_name}' not found"}

    def get_disease_symptoms(self, disease_name: str) -> List[str]:
        """Get symptoms associated with a specific disease"""
        if disease_name in self.disease_symptom_map:
            return self.disease_symptom_map[disease_name]
        return []

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.model_metrics

    def _fallback_prediction(self, symptoms: List[str]) -> Dict[str, Any]:
        """Fallback prediction method when model is not available"""
        # Simple rule-based prediction
        matches = {}
        for disease, disease_symptoms in self.disease_symptom_map.items():
            # Calculate how many symptoms match
            symptom_matches = set(symptoms).intersection(set(disease_symptoms))
            if symptom_matches:
                matches[disease] = len(symptom_matches) / len(disease_symptoms)

        # Get the best match
        if matches:
            predicted_disease = max(matches.items(), key=lambda x: x[1])[0]
            confidence = matches[predicted_disease] * 100
        else:
            predicted_disease = "Unknown"
            confidence = 0

        # Get disease information
        disease_info = self.disease_data.get(predicted_disease, {})
        description = disease_info.get('description', 'No description available')
        precautions = disease_info.get('precautions', [])

        # Get severity of provided symptoms
        symptoms_with_severity = {s: self.symptom_severity.get(s, 0) for s in symptoms}

        return {
            "predicted_disease": predicted_disease,
            "confidence": round(confidence, 2),
            "description": description,
            "precautions": precautions,
            "symptom_severity": symptoms_with_severity,
            "note": "Using fallback prediction method as model is not available"
        }

    def _fallback_top_predictions(self, symptoms: List[str], top_n: int = 3) -> Dict[str, Any]:
        """Fallback method for top predictions when model is not available"""
        # Simple rule-based prediction
        matches = {}
        for disease, disease_symptoms in self.disease_symptom_map.items():
            # Calculate how many symptoms match
            symptom_matches = set(symptoms).intersection(set(disease_symptoms))
            if symptom_matches:
                matches[disease] = len(symptom_matches) / len(disease_symptoms)

        # Sort diseases by match score
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)

        # Get top N matches
        top_predictions = []
        for disease, score in sorted_matches[:top_n]:
            confidence = score * 100
            disease_info = self.disease_data.get(disease, {})

            top_predictions.append({
                "disease": disease,
                "confidence": round(confidence, 2),
                "description": disease_info.get('description', 'No description available'),
                "precautions": disease_info.get('precautions', [])
            })

        # Get severity of provided symptoms
        symptoms_with_severity = {s: self.symptom_severity.get(s, 0) for s in symptoms}

        return {
            "top_predictions": top_predictions,
            "symptom_severity": symptoms_with_severity,
            "note": "Using fallback prediction method as model is not available"
        }
