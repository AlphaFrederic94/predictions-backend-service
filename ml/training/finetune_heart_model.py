import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.feature_selection import RFECV, SelectFromModel
from imblearn.over_sampling import SMOTE

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def finetune_heart_model():
    """Fine-tune the heart disease prediction model with advanced techniques"""
    print("Fine-tuning heart disease prediction model...")
    
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_path, "data", "datasets", "heart")
    model_path = os.path.join(base_path, "ml", "models", "heart")
    
    # Ensure model directory exists
    os.makedirs(model_path, exist_ok=True)
    
    # Load dataset
    dataset_path = os.path.join(data_path, "heart.csv")
    
    print(f"Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Data exploration
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Heart disease cases: {df['target'].sum()} ({df['target'].mean() * 100:.2f}%)")
    
    # Check for missing values
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Create feature description dictionary
    feature_descriptions = {
        "age": "Age in years",
        "sex": "Sex (1 = male, 0 = female)",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure (in mm Hg)",
        "chol": "Serum cholesterol in mg/dl",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        "restecg": "Resting electrocardiographic results (0-2)",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise induced angina (1 = yes, 0 = no)",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of the peak exercise ST segment (0-2)",
        "ca": "Number of major vessels colored by fluoroscopy (0-3)",
        "thal": "Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)"
    }
    
    # Prepare the data
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Store original feature names
    original_features = X.columns.tolist()
    
    # Feature Engineering: Create polynomial features for key predictors
    print("Creating polynomial features...")
    key_features = ['age', 'trestbps', 'chol', 'thalach']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features = poly.fit_transform(X[key_features])
    
    # Get feature names for polynomial features
    poly_feature_names = []
    for i, feat1 in enumerate(key_features):
        for j in range(i, len(key_features)):
            feat2 = key_features[j]
            if i != j:
                poly_feature_names.append(f"{feat1}_{feat2}")
    
    # Create DataFrame with polynomial features
    poly_df = pd.DataFrame(poly_features[:, len(key_features):], 
                          columns=poly_feature_names)
    
    # Combine original and polynomial features
    X = pd.concat([X, poly_df], axis=1)
    
    # Feature Engineering: Create risk score feature
    X['risk_score'] = (
        (X['age'] > 50).astype(int) * 2 +
        (X['sex'] == 1).astype(int) * 1.5 +
        (X['cp'] == 0).astype(int) * 2 +
        (X['trestbps'] > 140).astype(int) * 1.5 +
        (X['chol'] > 240).astype(int) * 1.5 +
        (X['fbs'] == 1).astype(int) * 1 +
        (X['thalach'] < 150).astype(int) * 1.5 +
        (X['exang'] == 1).astype(int) * 2 +
        (X['oldpeak'] > 1).astype(int) * 1.5 +
        (X['ca'] > 0).astype(int) * 2
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Handle class imbalance with SMOTE
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Original training set shape: {X_train.shape}")
    print(f"Resampled training set shape: {X_train_resampled.shape}")
    print(f"Original class distribution: {pd.Series(y_train).value_counts()}")
    print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts()}")
    
    # Scale the features
    print("Scaling features...")
    scaler = RobustScaler()  # RobustScaler is less affected by outliers
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using RFECV
    print("Performing feature selection with RFECV...")
    rfecv = RFECV(
        estimator=RandomForestClassifier(random_state=42),
        step=1,
        cv=StratifiedKFold(5),
        scoring='f1',
        min_features_to_select=5,
        n_jobs=-1
    )
    rfecv.fit(X_train_scaled, y_train_resampled)
    
    # Get selected features
    all_features = X.columns.tolist()
    selected_features = [all_features[i] for i in range(len(all_features)) if rfecv.support_[i]]
    print(f"Selected features: {selected_features}")
    print(f"Optimal number of features: {rfecv.n_features_}")
    
    # Use selected features
    X_train_selected = X_train_scaled[:, rfecv.support_]
    X_test_selected = X_test_scaled[:, rfecv.support_]
    
    # Hyperparameter tuning for Random Forest
    print("Performing hyperparameter tuning for Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf_grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=StratifiedKFold(5),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    rf_grid_search.fit(X_train_selected, y_train_resampled)
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
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    gb_grid_search.fit(X_train_selected, y_train_resampled)
    best_gb = gb_grid_search.best_estimator_
    print(f"Best GB parameters: {gb_grid_search.best_params_}")
    
    # Hyperparameter tuning for Neural Network
    print("Performing hyperparameter tuning for Neural Network...")
    nn_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    
    nn_grid_search = GridSearchCV(
        estimator=MLPClassifier(random_state=42, max_iter=1000),
        param_grid=nn_param_grid,
        cv=StratifiedKFold(5),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    nn_grid_search.fit(X_train_selected, y_train_resampled)
    best_nn = nn_grid_search.best_estimator_
    print(f"Best NN parameters: {nn_grid_search.best_params_}")
    
    # Create a voting classifier
    print("Creating voting classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('nn', best_nn)
        ],
        voting='soft'
    )
    
    voting_clf.fit(X_train_selected, y_train_resampled)
    
    # Evaluate all models
    models = {
        'Random Forest': best_rf,
        'Gradient Boosting': best_gb,
        'Neural Network': best_nn,
        'Voting Classifier': voting_clf
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_selected)
        y_prob = model.predict_proba(X_test_selected)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
    
    # Find the best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name}")
    
    # Cross-validation of the best model
    print(f"Performing cross-validation for {best_model_name}...")
    cv_scores = cross_val_score(best_model, X_train_selected, y_train_resampled, cv=StratifiedKFold(5), scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f}")
    
    # Save the best model and related data
    print("Saving model and data...")
    
    # Save the model
    with open(os.path.join(model_path, "finetuned_heart_model.pkl"), 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save the scaler
    with open(os.path.join(model_path, "heart_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the feature selector
    with open(os.path.join(model_path, "heart_feature_selector.pkl"), 'wb') as f:
        pickle.dump(rfecv, f)
    
    # Save original feature names
    with open(os.path.join(model_path, "heart_features.json"), 'w') as f:
        json.dump(original_features, f, indent=2)
    
    # Save selected features
    with open(os.path.join(model_path, "heart_selected_features.json"), 'w') as f:
        json.dump(selected_features, f, indent=2)
    
    # Save feature descriptions
    with open(os.path.join(model_path, "heart_feature_descriptions.json"), 'w') as f:
        json.dump(feature_descriptions, f, indent=2)
    
    # Save model performance metrics
    model_metrics = {
        "best_model": best_model_name,
        "model_results": {name: {k: float(v) for k, v in model_results.items()} for name, model_results in results.items()},
        "cross_validation_scores": [float(score) for score in cv_scores],
        "mean_cv_score": float(cv_scores.mean()),
        "feature_importance": {
            "selected_features": selected_features,
            "feature_descriptions": feature_descriptions
        }
    }
    
    with open(os.path.join(model_path, "heart_model_metrics.json"), 'w') as f:
        json.dump(model_metrics, f, indent=2)
    
    # Calculate statistics for the dataset
    statistics = {
        "total_records": len(df),
        "heart_disease_patients": int(df["target"].sum()),
        "healthy_patients": int(len(df) - df["target"].sum()),
        "feature_statistics": {}
    }
    
    # Calculate statistics for each feature
    for feature in original_features:
        statistics["feature_statistics"][feature] = {
            "mean": float(df[feature].mean()),
            "median": float(df[feature].median()),
            "min": float(df[feature].min()),
            "max": float(df[feature].max()),
            "std": float(df[feature].std())
        }
    
    # Save statistics
    with open(os.path.join(model_path, "heart_statistics.json"), 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print("Fine-tuned heart disease prediction model training completed!")
    print(f"Best model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"AUC: {results[best_model_name]['auc']:.4f}")
    
    return model_path

if __name__ == "__main__":
    finetune_heart_model()
