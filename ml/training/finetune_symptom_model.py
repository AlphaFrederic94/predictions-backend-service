import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import SelectFromModel

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def finetune_symptom_model():
    """Fine-tune the symptom-based disease prediction model with advanced techniques"""
    print("Fine-tuning symptom prediction model...")
    
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
    
    # Data exploration
    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique diseases: {df['Disease'].nunique()}")
    
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
    
    # Add symptom severity as a feature weight
    severity_dict = {}
    for _, row in severity_df.iterrows():
        severity_dict[row['Symptom']] = int(row['weight'])
    
    # Apply severity weights to the feature matrix
    for symptom in all_symptoms:
        if symptom in severity_dict:
            X[symptom] = X[symptom] * severity_dict.get(symptom, 1)
    
    # Feature Engineering: Add symptom count feature
    X['symptom_count'] = X[all_symptoms].sum(axis=1)
    
    # Feature Engineering: Add symptom co-occurrence features for top symptoms
    top_symptoms = ['fatigue', 'high_fever', 'headache', 'vomiting', 'cough']
    for i, symptom1 in enumerate(top_symptoms[:-1]):
        for symptom2 in top_symptoms[i+1:]:
            if symptom1 in X.columns and symptom2 in X.columns:
                X[f"{symptom1}_{symptom2}"] = X[symptom1] & X[symptom2]
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Feature selection to improve model performance
    print("Performing feature selection...")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Select important features
    sfm = SelectFromModel(base_model, threshold='mean')
    sfm.fit(X_train, y_train)
    
    # Get selected feature indices
    selected_features_idx = sfm.get_support(indices=True)
    selected_features = X.columns[selected_features_idx].tolist()
    
    print(f"Selected {len(selected_features)} out of {len(X.columns)} features")
    print(f"Top 10 important features: {selected_features[:10]}")
    
    # Use selected features
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)
    
    # Hyperparameter tuning with GridSearchCV for Random Forest
    print("Performing hyperparameter tuning for Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf_grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=StratifiedKFold(5),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    rf_grid_search.fit(X_train_selected, y_train)
    best_rf = rf_grid_search.best_estimator_
    print(f"Best RF parameters: {rf_grid_search.best_params_}")
    
    # Hyperparameter tuning for Gradient Boosting
    print("Performing hyperparameter tuning for Gradient Boosting...")
    gb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    
    gb_grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=gb_param_grid,
        cv=StratifiedKFold(5),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    gb_grid_search.fit(X_train_selected, y_train)
    best_gb = gb_grid_search.best_estimator_
    print(f"Best GB parameters: {gb_grid_search.best_params_}")
    
    # Hyperparameter tuning for SVM
    print("Performing hyperparameter tuning for SVM...")
    svm_param_grid = {
        'C': [1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced', None]
    }
    
    svm_grid_search = GridSearchCV(
        estimator=SVC(probability=True, random_state=42),
        param_grid=svm_param_grid,
        cv=StratifiedKFold(5),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    svm_grid_search.fit(X_train_selected, y_train)
    best_svm = svm_grid_search.best_estimator_
    print(f"Best SVM parameters: {svm_grid_search.best_params_}")
    
    # Create a voting classifier
    print("Creating voting classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('svm', best_svm)
        ],
        voting='soft'
    )
    
    voting_clf.fit(X_train_selected, y_train)
    
    # Evaluate all models
    models = {
        'Random Forest': best_rf,
        'Gradient Boosting': best_gb,
        'SVM': best_svm,
        'Voting Classifier': voting_clf
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_selected)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Find the best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name}")
    
    # Cross-validation to ensure model robustness
    print("Performing cross-validation...")
    cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=StratifiedKFold(5), scoring='f1_weighted')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f}")
    
    # Save the model and related data
    print("Saving model and data...")
    
    # Save the model
    with open(os.path.join(model_path, "finetuned_model.pkl"), 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save the feature selector
    with open(os.path.join(model_path, "feature_selector.pkl"), 'wb') as f:
        pickle.dump(sfm, f)
    
    # Save the label encoder
    with open(os.path.join(model_path, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save symptoms list
    with open(os.path.join(model_path, "symptoms_list.json"), 'w') as f:
        json.dump(all_symptoms, f, indent=2)
    
    # Save selected features
    with open(os.path.join(model_path, "selected_features.json"), 'w') as f:
        json.dump(selected_features, f, indent=2)
    
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
    
    # Save severity dictionary
    with open(os.path.join(model_path, "symptom_severity.json"), 'w') as f:
        json.dump(severity_dict, f, indent=2)
    
    # Save model performance metrics
    model_metrics = {
        "best_model": best_model_name,
        "accuracy": float(results[best_model_name]['accuracy']),
        "f1_score": float(results[best_model_name]['f1_score']),
        "cross_validation_scores": [float(score) for score in cv_scores],
        "mean_cv_score": float(cv_scores.mean()),
        "feature_selection": {
            "total_features": len(X.columns),
            "selected_features": len(selected_features),
            "top_features": selected_features[:20]
        }
    }
    
    with open(os.path.join(model_path, "model_metrics.json"), 'w') as f:
        json.dump(model_metrics, f, indent=2)
    
    print("Fine-tuned symptom prediction model training completed!")
    print(f"Best model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
    
    return model_path

if __name__ == "__main__":
    finetune_symptom_model()
