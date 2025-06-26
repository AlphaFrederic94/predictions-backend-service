import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def train_diabetes_model():
    """Train a Random Forest model for diabetes prediction"""
    print("Training diabetes prediction model...")
    
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_path, "data", "datasets", "diabetes")
    model_path = os.path.join(base_path, "ml", "models", "diabetes")
    
    # Ensure model directory exists
    os.makedirs(model_path, exist_ok=True)
    
    # Load dataset
    dataset_path = os.path.join(data_path, "diabetes.csv")
    
    print(f"Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Diabetes cases: {df['Outcome'].sum()} ({df['Outcome'].mean() * 100:.2f}%)")
    
    # Prepare the data
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    # Store feature names
    features = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model with hyperparameter tuning
    print("Training Random Forest model with hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Use a smaller subset of the parameter grid for faster execution
    quick_param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    
    # Create the base model
    rf = RandomForestClassifier(random_state=42)
    
    # Create the grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=quick_param_grid,  # Use quick_param_grid for faster execution
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search to the data
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate the model
    y_pred = best_rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importances
    feature_importances = pd.DataFrame(
        best_rf.feature_importances_,
        index=features,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    print("Feature Importances:")
    print(feature_importances)
    
    # Save the model and scaler
    print("Saving model and data...")
    with open(os.path.join(model_path, "diabetes_model.pkl"), 'wb') as f:
        pickle.dump(best_rf, f)
    
    with open(os.path.join(model_path, "diabetes_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(os.path.join(model_path, "diabetes_features.json"), 'w') as f:
        json.dump(features, f, indent=2)
    
    # Save feature importances
    feature_importances.to_csv(os.path.join(model_path, "diabetes_feature_importances.csv"))
    
    # Calculate statistics for the dataset
    statistics = {
        "total_records": len(df),
        "diabetic_patients": int(df["Outcome"].sum()),
        "non_diabetic_patients": int(len(df) - df["Outcome"].sum()),
        "feature_statistics": {}
    }
    
    # Calculate statistics for each feature
    for feature in features:
        statistics["feature_statistics"][feature] = {
            "mean": float(df[feature].mean()),
            "median": float(df[feature].median()),
            "min": float(df[feature].min()),
            "max": float(df[feature].max()),
            "std": float(df[feature].std())
        }
    
    # Save statistics
    with open(os.path.join(model_path, "diabetes_statistics.json"), 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print("Diabetes prediction model training completed!")
    return model_path

if __name__ == "__main__":
    train_diabetes_model()
