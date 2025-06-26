import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any

class FinetunedHeartService:
    """Service for heart disease prediction using fine-tuned model"""

    def __init__(self):
        """Initialize the service by loading the model and related data"""
        # Define paths
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_path, "ml", "models", "heart")

        # Load fine-tuned model if available, otherwise use standard model
        finetuned_model_file = os.path.join(model_path, "finetuned_heart_model.pkl")
        standard_model_file = os.path.join(model_path, "heart_model.pkl")

        if os.path.exists(finetuned_model_file):
            with open(finetuned_model_file, 'rb') as f:
                self.model = pickle.load(f)
            print("Using fine-tuned heart disease prediction model")
        elif os.path.exists(standard_model_file):
            with open(standard_model_file, 'rb') as f:
                self.model = pickle.load(f)
            print("Using standard heart disease prediction model as fallback")
        else:
            self.model = None
            print(f"Warning: No model file found at {finetuned_model_file} or {standard_model_file}")

        # Load scaler
        scaler_file = os.path.join(model_path, "heart_scaler.pkl")
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None
            print(f"Warning: Scaler file not found at {scaler_file}")

        # Load feature selector
        selector_file = os.path.join(model_path, "heart_feature_selector.pkl")
        if os.path.exists(selector_file):
            with open(selector_file, 'rb') as f:
                self.feature_selector = pickle.load(f)
        else:
            self.feature_selector = None
            print(f"Warning: Feature selector file not found at {selector_file}")

        # Load features
        features_file = os.path.join(model_path, "heart_features.json")
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                self.features = json.load(f)
        else:
            self.features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
            print(f"Warning: Features file not found at {features_file}")

        # Load selected features
        selected_features_file = os.path.join(model_path, "heart_selected_features.json")
        if os.path.exists(selected_features_file):
            with open(selected_features_file, 'r') as f:
                self.selected_features = json.load(f)
        else:
            self.selected_features = self.features
            print(f"Warning: Selected features file not found at {selected_features_file}")

        # Load feature descriptions
        descriptions_file = os.path.join(model_path, "heart_feature_descriptions.json")
        if os.path.exists(descriptions_file):
            with open(descriptions_file, 'r') as f:
                self.feature_descriptions = json.load(f)
        else:
            self.feature_descriptions = {}
            print(f"Warning: Feature descriptions file not found at {descriptions_file}")

        # Load statistics
        statistics_file = os.path.join(model_path, "heart_statistics.json")
        if os.path.exists(statistics_file):
            with open(statistics_file, 'r') as f:
                self.statistics = json.load(f)
        else:
            self.statistics = {}
            print(f"Warning: Statistics file not found at {statistics_file}")

        # Load model metrics
        metrics_file = os.path.join(model_path, "heart_model_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.model_metrics = json.load(f)
        else:
            self.model_metrics = {}
            print(f"Warning: Model metrics file not found at {metrics_file}")

    def get_features(self) -> Dict[str, str]:
        """Get all features with descriptions used for heart disease prediction"""
        return {feature: self.feature_descriptions.get(feature, "") for feature in self.features}

    def get_selected_features(self) -> Dict[str, str]:
        """Get selected features with descriptions used for heart disease prediction"""
        return {feature: self.feature_descriptions.get(feature, "") for feature in self.selected_features}

    def predict_heart_disease(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Predict heart disease based on input features"""
        if not self.model:
            return self._fallback_prediction(data)

        try:
            # Prepare input data - use only the original features
            input_data = np.array([[data.get(feature, 0) for feature in self.features]])

            # Scale the input data if scaler is available
            if self.scaler:
                input_data_scaled = self.scaler.transform(input_data)
            else:
                input_data_scaled = input_data

            # Apply feature selection if available
            if self.feature_selector:
                input_data_selected = self.feature_selector.transform(input_data_scaled)
            else:
                input_data_selected = input_data_scaled

            # Make prediction
            prediction = int(self.model.predict(input_data_selected)[0])

            # Get probability - handle different predict_proba outputs
            proba = self.model.predict_proba(input_data_selected)[0]
            if len(proba) > 1:
                probability = float(proba[1])  # Probability of class 1 (heart disease)
            else:
                probability = float(prediction)  # If only one class, use the prediction
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._fallback_prediction(data)

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Generate recommendations
        recommendations = self._generate_recommendations(data, prediction, probability)

        return {
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "risk_level": risk_level,
            "recommendations": recommendations,
            "feature_importance": self._get_feature_importance(data)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the heart disease dataset"""
        return self.statistics

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.model_metrics

    def _generate_recommendations(self, data: Dict[str, float], prediction: int, probability: float) -> List[str]:
        """Generate personalized recommendations based on prediction and input data"""
        recommendations = []

        if prediction == 1:
            recommendations.append("Consult with a cardiologist for a comprehensive heart health assessment.")

            if data.get('chol', 0) > 200:
                recommendations.append("Your cholesterol level is high. Consider dietary changes and medication if prescribed.")

            if data.get('trestbps', 0) > 140:
                recommendations.append("Your resting blood pressure is elevated. Regular monitoring is recommended.")

            if data.get('thalach', 0) < 150:
                recommendations.append("Your maximum heart rate is lower than average. Discuss with your doctor about appropriate exercise.")

            if data.get('cp', 0) == 0:  # Typical angina
                recommendations.append("You have typical angina symptoms. Follow your doctor's advice for managing chest pain.")

            if data.get('exang', 0) == 1:  # Exercise induced angina
                recommendations.append("You experience angina during exercise. Consider a supervised exercise program.")

            if data.get('ca', 0) > 0:
                recommendations.append(f"You have {int(data.get('ca', 0))} major vessels colored by fluoroscopy. Follow up with your cardiologist.")
        else:
            recommendations.append("Your risk of heart disease appears to be low, but maintaining a heart-healthy lifestyle is still important.")

            if data.get('chol', 0) > 180:
                recommendations.append("Your cholesterol level is slightly elevated. Consider a heart-healthy diet.")

            if data.get('trestbps', 0) > 120:
                recommendations.append("Your blood pressure is slightly elevated. Regular monitoring is recommended.")

            if data.get('age', 0) > 50:
                recommendations.append("As you are over 50, regular cardiovascular check-ups are recommended.")

        # General recommendations
        recommendations.append("Maintain a heart-healthy diet low in saturated fats, trans fats, and sodium.")
        recommendations.append("Engage in regular aerobic exercise (at least 150 minutes per week).")
        recommendations.append("Limit alcohol consumption and avoid smoking.")
        recommendations.append("Manage stress through relaxation techniques, adequate sleep, and social support.")

        return recommendations

    def _get_feature_importance(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Get feature importance for the prediction"""
        if not self.model_metrics or 'feature_importance' not in self.model_metrics:
            return {}

        feature_importance = {}
        if 'selected_features' in self.model_metrics.get('feature_importance', {}):
            selected_features = self.model_metrics['feature_importance']['selected_features']
            for feature in selected_features:
                if feature in self.features:
                    feature_importance[feature] = {
                        "value": data.get(feature, 0),
                        "description": self.feature_descriptions.get(feature, ""),
                        "importance": "High"
                    }

        return feature_importance

    def _fallback_prediction(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Fallback prediction method when model is not available"""
        # Simple rule-based prediction
        risk_score = 0

        # Age is a risk factor
        age = data.get('age', 0)
        if age > 60:
            risk_score += 15
        elif age > 50:
            risk_score += 10
        elif age > 40:
            risk_score += 5

        # Gender (males have higher risk)
        if data.get('sex', 0) == 1:  # Male
            risk_score += 10

        # Chest pain type (4 = asymptomatic, highest risk)
        cp = data.get('cp', 0)
        if cp == 0:  # Typical angina
            risk_score += 20
        elif cp == 1:  # Atypical angina
            risk_score += 15
        elif cp == 2:  # Non-anginal pain
            risk_score += 10
        elif cp == 3:  # Asymptomatic
            risk_score += 5

        # High blood pressure
        trestbps = data.get('trestbps', 0)
        if trestbps > 140:
            risk_score += 15
        elif trestbps > 130:
            risk_score += 10
        elif trestbps > 120:
            risk_score += 5

        # Cholesterol
        chol = data.get('chol', 0)
        if chol > 240:
            risk_score += 15
        elif chol > 200:
            risk_score += 10

        # Fasting blood sugar > 120 mg/dl
        if data.get('fbs', 0) == 1:
            risk_score += 10

        # Maximum heart rate
        thalach = data.get('thalach', 0)
        if thalach < 120:
            risk_score += 15
        elif thalach < 140:
            risk_score += 10

        # Exercise induced angina
        if data.get('exang', 0) == 1:
            risk_score += 15

        # ST depression induced by exercise
        oldpeak = data.get('oldpeak', 0)
        if oldpeak > 2:
            risk_score += 15
        elif oldpeak > 1:
            risk_score += 10

        # Number of major vessels
        ca = data.get('ca', 0)
        risk_score += ca * 10

        # Convert risk score to probability
        probability = min(risk_score / 150, 0.99)

        # Make prediction (1 for heart disease, 0 for no heart disease)
        prediction = 1 if probability > 0.5 else 0

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Generate recommendations
        recommendations = self._generate_recommendations(data, prediction, probability)

        return {
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "risk_level": risk_level,
            "recommendations": recommendations,
            "note": "Using fallback prediction method as model is not available"
        }
