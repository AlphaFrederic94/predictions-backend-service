import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def train_symptom_model():
    """Train a Random Forest model for symptom-based disease prediction"""
    print("Training symptom prediction model...")
    
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_path, "data", "datasets", "symptoms")
    model_path = os.path.join(base_path, "ml", "models", "symptoms")
    
    # Ensure model directory exists
    os.makedirs(model_path, exist_ok=True)
    
    # Load datasets
    dataset_path = os.path.join(data_path, "dataset.csv")
    severity_path = os.path.join(data_path, "Symptom-severity.csv")
    description_path = os.path.join(data_path, "symptom_Description.csv")
    precaution_path = os.path.join(data_path, "symptom_precaution.csv")
    
    print(f"Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    severity_df = pd.read_csv(severity_path)
    description_df = pd.read_csv(description_path)
    precaution_df = pd.read_csv(precaution_path)
    
    # Extract all unique symptoms
    all_symptoms = set()
    for col in df.columns:
        if 'Symptom' in col:
            all_symptoms.update(df[col].dropna().unique())
    
    all_symptoms = sorted([s.strip() for s in all_symptoms if isinstance(s, str) and s.strip()])
    print(f"Total unique symptoms: {len(all_symptoms)}")
    
    # Create feature matrix (one-hot encoding for symptoms)
    X = pd.DataFrame(0, index=range(len(df)), columns=all_symptoms)
    y = df['Disease']
    
    # Fill the feature matrix
    for i, row in df.iterrows():
        for col in df.columns:
            if 'Symptom' in col and pd.notna(row[col]):
                symptom = row[col].strip()
                if symptom in all_symptoms:
                    X.loc[i, symptom] = 1
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Train a Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model and label encoder
    print("Saving model and data...")
    with open(os.path.join(model_path, "rf_model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(model_path, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save symptoms list
    with open(os.path.join(model_path, "symptoms_list.json"), 'w') as f:
        json.dump(all_symptoms, f, indent=2)
    
    # Create a mapping of diseases to symptoms
    disease_symptom_map = {}
    for disease in df['Disease'].unique():
        disease_rows = df[df['Disease'] == disease]
        symptoms = set()
        for _, row in disease_rows.iterrows():
            for col in df.columns:
                if 'Symptom' in col and pd.notna(row[col]):
                    symptom = row[col].strip()
                    if symptom:
                        symptoms.add(symptom)
        disease_symptom_map[disease] = sorted(list(symptoms))
    
    # Save disease-symptom mapping
    with open(os.path.join(model_path, "disease_symptom_map.json"), 'w') as f:
        json.dump(disease_symptom_map, f, indent=2)
    
    # Create disease data
    disease_data = {}
    for _, row in description_df.iterrows():
        disease = row['Disease']
        disease_data[disease] = {
            'description': row['Description'],
            'precautions': []
        }
    
    # Add precautions
    for _, row in precaution_df.iterrows():
        disease = row['Disease']
        if disease in disease_data:
            precautions = []
            for i in range(1, 5):
                col = f'Precaution_{i}'
                if col in row and pd.notna(row[col]) and row[col].strip():
                    precautions.append(row[col].strip())
            disease_data[disease]['precautions'] = precautions
    
    # Save disease data
    with open(os.path.join(model_path, "disease_data.json"), 'w') as f:
        json.dump(disease_data, f, indent=2)
    
    # Create severity dictionary
    severity_dict = {}
    for _, row in severity_df.iterrows():
        severity_dict[row['Symptom']] = int(row['weight'])
    
    # Save severity data
    with open(os.path.join(model_path, "symptom_severity.json"), 'w') as f:
        json.dump(severity_dict, f, indent=2)
    
    print("Symptom prediction model training completed!")
    return model_path

if __name__ == "__main__":
    train_symptom_model()
