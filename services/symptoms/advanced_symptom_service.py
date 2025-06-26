import os
import json
import numpy as np
import joblib
from typing import List, Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class AdvancedSymptomService:
    """Advanced service for symptom-based disease prediction using state-of-the-art models"""

    def __init__(self):
        """Initialize the service by loading the model and related data"""
        # Define paths
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(base_path, "ml", "models", "symptoms")

        # Initialize components
        self.pipeline = None
        self.model = None
        self.label_encoder = None
        self.symptoms_list = []
        self.selected_features = []
        self.disease_symptom_map = {}
        self.disease_data = {}
        self.symptom_severity = {}
        self.model_metrics = {}

        # Load all components
        self._load_pipeline()
        self._load_model()
        self._load_label_encoder()
        self._load_symptoms_list()
        self._load_selected_features()
        self._load_disease_symptom_map()
        self._load_disease_data()
        self._load_symptom_severity()
        self._load_model_metrics()

    def _load_pipeline(self):
        """Load the prediction pipeline"""
        pipeline_file = os.path.join(self.model_path, "advanced_symptom_pipeline.pkl")
        standard_pipeline_file = os.path.join(self.model_path, "symptom_pipeline.pkl")

        if os.path.exists(pipeline_file):
            self.pipeline = joblib.load(pipeline_file)
            print("Using advanced symptom prediction pipeline")
        elif os.path.exists(standard_pipeline_file):
            self.pipeline = joblib.load(standard_pipeline_file)
            print("Using standard symptom prediction pipeline as fallback")
        else:
            self.pipeline = None
            print(f"Warning: No pipeline found at {pipeline_file} or {standard_pipeline_file}")

    def _load_model(self):
        """Load the prediction model"""
        model_file = os.path.join(self.model_path, "advanced_symptom_model.pkl")
        standard_model_file = os.path.join(self.model_path, "rf_model.pkl")

        if os.path.exists(model_file):
            self.model = joblib.load(model_file)
            print("Using advanced symptom prediction model")
        elif os.path.exists(standard_model_file):
            self.model = joblib.load(standard_model_file)
            print("Using standard symptom prediction model as fallback")
        else:
            self.model = None
            print(f"Warning: No model found at {model_file} or {standard_model_file}")

    def _load_label_encoder(self):
        """Load the label encoder"""
        encoder_file = os.path.join(self.model_path, "advanced_label_encoder.pkl")
        standard_encoder_file = os.path.join(self.model_path, "label_encoder.pkl")

        if os.path.exists(encoder_file):
            self.label_encoder = joblib.load(encoder_file)
        elif os.path.exists(standard_encoder_file):
            self.label_encoder = joblib.load(standard_encoder_file)
        else:
            self.label_encoder = None
            print(f"Warning: No label encoder found at {encoder_file} or {standard_encoder_file}")

    def _load_symptoms_list(self):
        """Load the symptoms list"""
        symptoms_file = os.path.join(self.model_path, "advanced_symptoms_list.json")
        standard_symptoms_file = os.path.join(self.model_path, "symptoms_list.json")

        if os.path.exists(symptoms_file):
            with open(symptoms_file, 'r') as f:
                self.symptoms_list = json.load(f)
        elif os.path.exists(standard_symptoms_file):
            with open(standard_symptoms_file, 'r') as f:
                self.symptoms_list = json.load(f)
        else:
            self.symptoms_list = []
            print(f"Warning: No symptoms list found at {symptoms_file} or {standard_symptoms_file}")

    def _load_selected_features(self):
        """Load the selected features"""
        features_file = os.path.join(self.model_path, "advanced_selected_features.json")
        standard_features_file = os.path.join(self.model_path, "selected_features.json")

        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                self.selected_features = json.load(f)
        elif os.path.exists(standard_features_file):
            with open(standard_features_file, 'r') as f:
                self.selected_features = json.load(f)
        else:
            self.selected_features = self.symptoms_list
            print(f"Warning: No selected features found at {features_file} or {standard_features_file}")

    def _load_disease_symptom_map(self):
        """Load the disease-symptom mapping"""
        mapping_file = os.path.join(self.model_path, "advanced_disease_symptom_map.json")
        standard_mapping_file = os.path.join(self.model_path, "disease_symptom_map.json")

        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.disease_symptom_map = json.load(f)
        elif os.path.exists(standard_mapping_file):
            with open(standard_mapping_file, 'r') as f:
                self.disease_symptom_map = json.load(f)
        else:
            self.disease_symptom_map = {}
            print(f"Warning: No disease-symptom mapping found at {mapping_file} or {standard_mapping_file}")

    def _load_disease_data(self):
        """Load the disease data"""
        data_file = os.path.join(self.model_path, "advanced_disease_data.json")
        standard_data_file = os.path.join(self.model_path, "disease_data.json")

        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                self.disease_data = json.load(f)
        elif os.path.exists(standard_data_file):
            with open(standard_data_file, 'r') as f:
                self.disease_data = json.load(f)
        else:
            self.disease_data = {}
            print(f"Warning: No disease data found at {data_file} or {standard_data_file}")

    def _load_symptom_severity(self):
        """Load the symptom severity data"""
        severity_file = os.path.join(self.model_path, "advanced_symptom_severity.json")
        standard_severity_file = os.path.join(self.model_path, "symptom_severity.json")

        if os.path.exists(severity_file):
            with open(severity_file, 'r') as f:
                self.symptom_severity = json.load(f)
        elif os.path.exists(standard_severity_file):
            with open(standard_severity_file, 'r') as f:
                self.symptom_severity = json.load(f)
        else:
            self.symptom_severity = {}
            print(f"Warning: No symptom severity data found at {severity_file} or {standard_severity_file}")

    def _load_model_metrics(self):
        """Load the model metrics"""
        metrics_file = os.path.join(self.model_path, "advanced_model_metrics.json")
        standard_metrics_file = os.path.join(self.model_path, "model_metrics.json")

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.model_metrics = json.load(f)
        elif os.path.exists(standard_metrics_file):
            with open(standard_metrics_file, 'r') as f:
                self.model_metrics = json.load(f)
        else:
            self.model_metrics = {}
            print(f"Warning: No model metrics found at {metrics_file} or {standard_metrics_file}")

    def get_all_symptoms(self) -> List[str]:
        """Get all available symptoms"""
        return self.symptoms_list

    def get_similar_symptoms(self, partial_symptom: str, limit: int = 10) -> List[str]:
        """Get symptoms that match a partial string"""
        if not partial_symptom:
            return []

        # Find matching symptoms
        matches = [s for s in self.symptoms_list if partial_symptom.lower() in s.lower()]

        # Limit the number of results
        matches = matches[:limit]

        return matches

    def get_all_diseases(self) -> Dict[str, Any]:
        """Get all diseases with their descriptions and symptoms"""
        result = {}
        for disease, info in self.disease_data.items():
            symptoms = self.disease_symptom_map.get(disease, [])
            result[disease] = {
                "description": info.get('description', 'No description available'),
                "precautions": info.get('precautions', []),
                "symptoms": symptoms
            }

        return result

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.model_metrics

    def _create_feature_vector(self, symptoms: List[str]) -> np.ndarray:
        """Create a feature vector from the symptoms"""
        # Create base feature vector
        feature_vector = np.zeros(len(self.symptoms_list))

        # Fill in symptoms
        for symptom in symptoms:
            if symptom in self.symptoms_list:
                idx = self.symptoms_list.index(symptom)
                # Apply severity weight if available
                severity = self.symptom_severity.get(symptom, 1)
                feature_vector[idx] = severity

        # Add symptom count if it's in the features
        if 'symptom_count' in self.symptoms_list:
            idx = self.symptoms_list.index('symptom_count')
            feature_vector[idx] = len(symptoms)

        # Add severity score if it's in the features
        if 'severity_score' in self.symptoms_list:
            idx = self.symptoms_list.index('severity_score')
            severity_score = sum(self.symptom_severity.get(s, 0) for s in symptoms)
            feature_vector[idx] = severity_score

        # Add co-occurrence features if they're in the features
        for i, symptom1 in enumerate(symptoms):
            for j, symptom2 in enumerate(symptoms[i+1:]):
                feature_name = f"{symptom1}_{symptom2}"
                if feature_name in self.symptoms_list:
                    idx = self.symptoms_list.index(feature_name)
                    feature_vector[idx] = 1

        return feature_vector

    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """Predict disease based on symptoms using the advanced model"""
        if not symptoms:
            return {
                "predicted_disease": "Unknown",
                "confidence": 0.0,
                "description": "No symptoms provided",
                "precautions": [],
                "symptom_severity": {}
            }

        if not self.pipeline or not self.model or not self.label_encoder or not self.symptoms_list:
            return self._fallback_prediction(symptoms)

        try:
            # Create feature vector
            feature_vector = self._create_feature_vector(symptoms)

            # Reshape for prediction
            feature_vector = feature_vector.reshape(1, -1)

            # Make prediction using the pipeline if available
            if self.pipeline and isinstance(self.pipeline, Pipeline):
                prediction_idx = self.pipeline.predict(feature_vector)[0]
                probabilities = self.pipeline.predict_proba(feature_vector)[0]
            else:
                # Use the model directly if pipeline is not available
                prediction_idx = self.model.predict(feature_vector)[0]
                probabilities = self.model.predict_proba(feature_vector)[0]

            # Get the confidence score
            confidence = probabilities[prediction_idx] * 100

            # Get disease name
            disease = self.label_encoder.inverse_transform([prediction_idx])[0]

            # Get disease information
            disease_info = self.disease_data.get(disease, {})
            description = disease_info.get('description', 'No description available')
            precautions = disease_info.get('precautions', [])

            # Get severity of provided symptoms
            symptoms_with_severity = {s: self.symptom_severity.get(s, 0) for s in symptoms}

            # Get top matching symptoms for this disease
            disease_symptoms = self.disease_symptom_map.get(disease, [])
            matching_symptoms = set(symptoms).intersection(set(disease_symptoms))

            # Calculate confidence based on matching symptoms if prediction confidence is low
            if confidence < 50 and disease_symptoms:
                symptom_match_confidence = (len(matching_symptoms) / len(disease_symptoms)) * 100
                # Use the higher of the two confidence scores
                confidence = max(confidence, symptom_match_confidence)

            return {
                "predicted_disease": disease,
                "confidence": round(confidence, 2),
                "description": description,
                "precautions": precautions,
                "symptom_severity": symptoms_with_severity,
                "matching_symptoms": list(matching_symptoms),
                "total_disease_symptoms": len(disease_symptoms)
            }
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._fallback_prediction(symptoms)

    def predict_top_diseases(self, symptoms: List[str], top_n: int = 3) -> Dict[str, Any]:
        """Predict top N diseases based on symptoms"""
        if not symptoms:
            return {
                "top_predictions": [],
                "symptom_severity": {}
            }

        if not self.pipeline or not self.model or not self.label_encoder or not self.symptoms_list:
            return self._fallback_top_predictions(symptoms, top_n)

        try:
            # Create feature vector
            feature_vector = self._create_feature_vector(symptoms)

            # Reshape for prediction
            feature_vector = feature_vector.reshape(1, -1)

            # Get probabilities for all diseases
            if self.pipeline and isinstance(self.pipeline, Pipeline):
                probabilities = self.pipeline.predict_proba(feature_vector)[0]
            else:
                probabilities = self.model.predict_proba(feature_vector)[0]

            # Get top N predictions
            top_indices = np.argsort(probabilities)[::-1][:top_n]
            top_predictions = []

            for idx in top_indices:
                disease = self.label_encoder.inverse_transform([idx])[0]
                confidence = probabilities[idx] * 100

                # Get disease information
                disease_info = self.disease_data.get(disease, {})
                description = disease_info.get('description', 'No description available')
                precautions = disease_info.get('precautions', [])

                # Get matching symptoms for this disease
                disease_symptoms = self.disease_symptom_map.get(disease, [])
                matching_symptoms = set(symptoms).intersection(set(disease_symptoms))

                # Calculate confidence based on matching symptoms if prediction confidence is low
                if confidence < 50 and disease_symptoms:
                    symptom_match_confidence = (len(matching_symptoms) / len(disease_symptoms)) * 100
                    # Use the higher of the two confidence scores
                    confidence = max(confidence, symptom_match_confidence)

                top_predictions.append({
                    "disease": disease,
                    "confidence": round(confidence, 2),
                    "description": description,
                    "precautions": precautions,
                    "matching_symptoms": list(matching_symptoms),
                    "total_disease_symptoms": len(disease_symptoms)
                })

            # Get severity of provided symptoms
            symptoms_with_severity = {s: self.symptom_severity.get(s, 0) for s in symptoms}

            return {
                "top_predictions": top_predictions,
                "symptom_severity": symptoms_with_severity
            }
        except Exception as e:
            print(f"Error in top predictions: {e}")
            return self._fallback_top_predictions(symptoms, top_n)

    def get_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """Get information about a specific disease"""
        if disease_name in self.disease_data:
            disease_info = self.disease_data[disease_name]
            symptoms = self.disease_symptom_map.get(disease_name, [])

            return {
                "disease": disease_name,
                "description": disease_info.get('description', 'No description available'),
                "precautions": disease_info.get('precautions', []),
                "symptoms": symptoms,
                "symptom_severity": {s: self.symptom_severity.get(s, 0) for s in symptoms}
            }

        return {"error": f"Disease '{disease_name}' not found"}

    def get_all_diseases(self) -> Dict[str, Dict[str, Any]]:
        """Get all diseases with their descriptions and symptoms"""
        result = {}

        for disease, info in self.disease_data.items():
            symptoms = self.disease_symptom_map.get(disease, [])
            result[disease] = {
                "description": info.get('description', 'No description available'),
                "precautions": info.get('precautions', []),
                "symptoms": symptoms
            }

        return result

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.model_metrics

    def get_symptom_severity(self, symptom: str) -> int:
        """Get the severity of a specific symptom"""
        return self.symptom_severity.get(symptom, 0)

    def get_similar_symptoms(self, partial_symptom: str, limit: int = 10) -> List[str]:
        """Get symptoms that match a partial string"""
        matches = [s for s in self.symptoms_list if partial_symptom.lower() in s.lower()]
        return matches[:limit]

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
            matching_symptoms = set(symptoms).intersection(set(self.disease_symptom_map.get(predicted_disease, [])))
        else:
            predicted_disease = "Unknown"
            confidence = 0
            matching_symptoms = set()

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
            "matching_symptoms": list(matching_symptoms),
            "total_disease_symptoms": len(self.disease_symptom_map.get(predicted_disease, [])),
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
            matching_symptoms = set(symptoms).intersection(set(self.disease_symptom_map.get(disease, [])))

            top_predictions.append({
                "disease": disease,
                "confidence": round(confidence, 2),
                "description": disease_info.get('description', 'No description available'),
                "precautions": disease_info.get('precautions', []),
                "matching_symptoms": list(matching_symptoms),
                "total_disease_symptoms": len(self.disease_symptom_map.get(disease, []))
            })

        # Get severity of provided symptoms
        symptoms_with_severity = {s: self.symptom_severity.get(s, 0) for s in symptoms}

        return {
            "top_predictions": top_predictions,
            "symptom_severity": symptoms_with_severity,
            "note": "Using fallback prediction method as model is not available"
        }
