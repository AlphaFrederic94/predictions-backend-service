import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
import joblib

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(base_path)

def train_advanced_symptom_model():
    """Train an advanced symptom prediction model using RandomForestClassifier with RandomizedSearchCV and PCA"""
    print("Training advanced symptom prediction model...")

    # Create model directory if it doesn't exist
    model_path = os.path.join(base_path, "ml", "models", "symptoms")
    os.makedirs(model_path, exist_ok=True)

    # Load the datasets
    train_data_path = os.path.join(base_path, "data", "datasets", "more_symptoms", "Training.csv")
    test_data_path = os.path.join(base_path, "data", "datasets", "more_symptoms", "Testing.csv")

    # Load symptom severity data
    severity_path = os.path.join(base_path, "data", "datasets", "symptoms", "Symptom-severity.csv")
    description_path = os.path.join(base_path, "data", "datasets", "symptoms", "symptom_Description.csv")
    precaution_path = os.path.join(base_path, "data", "datasets", "symptoms", "symptom_precaution.csv")

    print(f"Loading training data from {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    print(f"Loading testing data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # Load severity, description, and precaution data
    print(f"Loading severity data from {severity_path}")
    severity_df = pd.read_csv(severity_path)

    print(f"Loading description data from {description_path}")
    description_df = pd.read_csv(description_path)

    print(f"Loading precaution data from {precaution_path}")
    precaution_df = pd.read_csv(precaution_path)

    # Combine training and testing data for preprocessing
    df = pd.concat([train_df, test_df])

    # Data exploration
    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique diseases: {df['prognosis'].nunique()}")
    print(f"Disease distribution:\n{df['prognosis'].value_counts().head()}")

    # Get all column names except the target column 'prognosis'
    # Remove any unnamed columns
    all_columns = df.columns.tolist()
    all_symptoms = [col for col in all_columns if col != 'prognosis' and not col.startswith('Unnamed:')]
    print(f"Total symptoms in dataset: {len(all_symptoms)}")

    # Create feature matrix (symptoms are already one-hot encoded)
    X = df[all_symptoms]
    y = df['prognosis']

    # Split back into training and testing sets
    X_train = train_df[all_symptoms]
    y_train = train_df['prognosis']
    X_test = test_df[all_symptoms]
    y_test = test_df['prognosis']

    # Create severity dictionary
    severity_dict = {}
    for _, row in severity_df.iterrows():
        severity_dict[row['Symptom']] = int(row['weight'])

    # Feature Engineering: Apply severity weights
    X_weighted = X.copy()
    for symptom in all_symptoms:
        if symptom in severity_dict:
            X_weighted[symptom] = X_weighted[symptom] * severity_dict.get(symptom, 1)

    # Feature Engineering: Add symptom count feature
    X_weighted['symptom_count'] = X_weighted[all_symptoms].sum(axis=1)

    # Feature Engineering: Add symptom severity score
    X_weighted['severity_score'] = 0
    for symptom in all_symptoms:
        if symptom in severity_dict:
            X_weighted['severity_score'] += X_weighted[symptom]

    # Feature Engineering: Add co-occurrence features for top symptoms
    # Find top symptoms by frequency
    top_symptoms = X.sum().sort_values(ascending=False).head(10).index.tolist()
    print(f"Top 10 symptoms by frequency: {top_symptoms}")

    # Add co-occurrence features
    for i, symptom1 in enumerate(top_symptoms[:-1]):
        for symptom2 in top_symptoms[i+1:]:
            feature_name = f"{symptom1}_{symptom2}"
            X_weighted[feature_name] = X_weighted[symptom1] & X_weighted[symptom2]

    # Apply the same transformations to train and test sets
    X_train_weighted = X_train.copy()
    X_test_weighted = X_test.copy()

    # Apply severity weights
    for symptom in all_symptoms:
        if symptom in severity_dict:
            X_train_weighted[symptom] = X_train_weighted[symptom] * severity_dict.get(symptom, 1)
            X_test_weighted[symptom] = X_test_weighted[symptom] * severity_dict.get(symptom, 1)

    # Add symptom count feature
    X_train_weighted['symptom_count'] = X_train_weighted[all_symptoms].sum(axis=1)
    X_test_weighted['symptom_count'] = X_test_weighted[all_symptoms].sum(axis=1)

    # Add symptom severity score
    X_train_weighted['severity_score'] = 0
    X_test_weighted['severity_score'] = 0
    for symptom in all_symptoms:
        if symptom in severity_dict:
            X_train_weighted['severity_score'] += X_train_weighted[symptom]
            X_test_weighted['severity_score'] += X_test_weighted[symptom]

    # Add co-occurrence features
    for i, symptom1 in enumerate(top_symptoms[:-1]):
        for symptom2 in top_symptoms[i+1:]:
            feature_name = f"{symptom1}_{symptom2}"
            X_train_weighted[feature_name] = X_train_weighted[symptom1] & X_train_weighted[symptom2]
            X_test_weighted[feature_name] = X_test_weighted[symptom1] & X_test_weighted[symptom2]

    # Apply PCA for dimensionality reduction
    print("Applying PCA for dimensionality reduction...")

    # Get numeric columns
    X_train_numeric = X_train_weighted.select_dtypes(include=[np.number])
    X_test_numeric = X_test_weighted.select_dtypes(include=[np.number])

    # Fill any NaN values
    X_train_numeric = X_train_numeric.fillna(0)
    X_test_numeric = X_test_numeric.fillna(0)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # Apply PCA
    n_components = min(65, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Convert to DataFrame
    X_train_pca = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    X_test_pca = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(n_components)])

    print(f"Original shape: {X_train_numeric.shape}")
    print(f"Reduced shape: {X_train_pca.shape}")
    print(f"Explained Variance Ratio (first 10 PCs): {pca.explained_variance_ratio_[:10]}")
    print(f"Total variance retained: {np.sum(pca.explained_variance_ratio_)}")

    # Create two versions of the dataset: with and without PCA
    X_train_without_pca = X_train_weighted
    X_test_without_pca = X_test_weighted

    X_train_with_pca = X_train_pca
    X_test_with_pca = X_test_pca

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Handle class imbalance with SMOTE for both datasets
    print("Applying SMOTE to handle class imbalance...")

    # For non-PCA data
    smote = SMOTE(random_state=42)
    X_train_weighted_resampled, y_train_weighted_resampled = smote.fit_resample(X_train_weighted, y_train_encoded)
    print(f"Original training set shape (without PCA): {X_train_weighted.shape}")
    print(f"Resampled training set shape (without PCA): {X_train_weighted_resampled.shape}")

    # For PCA data
    smote_pca = SMOTE(random_state=42)
    X_train_pca_resampled, y_train_pca_resampled = smote_pca.fit_resample(X_train_pca, y_train_encoded)
    print(f"Original training set shape (with PCA): {X_train_pca.shape}")
    print(f"Resampled training set shape (with PCA): {X_train_pca_resampled.shape}")

    # We'll train models on both datasets: with and without PCA
    # First, let's train models without PCA
    print("\n=== Training models without PCA ===\n")

    # Feature selection using RFECV for non-PCA data
    print("Performing feature selection with RFECV on non-PCA data...")
    rfecv_no_pca = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        step=1,
        cv=StratifiedKFold(5),
        scoring='f1_weighted',
        min_features_to_select=20,
        n_jobs=-1
    )
    # Convert to numpy arrays for RFECV
    X_train_weighted_np = X_train_weighted_resampled.values
    y_train_weighted_np = y_train_weighted_resampled

    rfecv_no_pca.fit(X_train_weighted_np, y_train_weighted_np)

    # Get selected features for non-PCA data
    selected_features = X_train_weighted.columns[rfecv_no_pca.support_].tolist()
    print(f"Selected {len(selected_features)} out of {len(X_train_weighted.columns)} features")
    print(f"Top 10 selected features: {selected_features[:10]}")

    # Use selected features for non-PCA data
    X_train_selected = X_train_weighted_np[:, rfecv_no_pca.support_]
    X_test_selected = X_test_weighted.values[:, rfecv_no_pca.support_]

    # Hyperparameter tuning with RandomizedSearchCV for Random Forest
    print("Performing hyperparameter tuning for Random Forest...")
    rf_param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None] + list(randint(10, 50).rvs(5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=rf_param_dist,
        n_iter=20,
        cv=StratifiedKFold(5),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    rf_random.fit(X_train_selected, y_train_weighted_resampled)
    best_rf = rf_random.best_estimator_
    print(f"Best RF parameters: {rf_random.best_params_}")

    # Hyperparameter tuning for Gradient Boosting
    print("Performing hyperparameter tuning for Gradient Boosting...")
    gb_param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'max_features': ['sqrt', 'log2', None]
    }

    gb_random = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_distributions=gb_param_dist,
        n_iter=20,
        cv=StratifiedKFold(5),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    gb_random.fit(X_train_selected, y_train_weighted_resampled)
    best_gb = gb_random.best_estimator_
    print(f"Best GB parameters: {gb_random.best_params_}")

    # Hyperparameter tuning for Neural Network
    print("Performing hyperparameter tuning for Neural Network...")
    nn_param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': uniform(0.0001, 0.01),
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000]
    }

    nn_random = RandomizedSearchCV(
        estimator=MLPClassifier(random_state=42),
        param_distributions=nn_param_dist,
        n_iter=10,
        cv=StratifiedKFold(3),  # Reduced folds for speed
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    nn_random.fit(X_train_selected, y_train_weighted_resampled)
    best_nn = nn_random.best_estimator_
    print(f"Best NN parameters: {nn_random.best_params_}")

    # Create a stacking classifier
    print("Creating stacking classifier...")
    estimators = [
        ('rf', best_rf),
        ('gb', best_gb),
        ('nn', best_nn)
    ]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=SVC(probability=True, random_state=42),
        cv=5,
        n_jobs=-1
    )

    stacking_clf.fit(X_train_selected, y_train_weighted_resampled)

    # Evaluate all models
    models = {
        'Random Forest': best_rf,
        'Gradient Boosting': best_gb,
        'Neural Network': best_nn,
        'Stacking Classifier': stacking_clf
    }

    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_selected)
        y_prob = model.predict_proba(X_test_selected)

        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_pred, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')

        # Calculate ROC AUC for multiclass
        roc_auc = 0
        try:
            # One-vs-Rest approach for multiclass ROC AUC
            roc_auc = roc_auc_score(y_test_encoded, y_prob, multi_class='ovr', average='weighted')
        except:
            print(f"Could not calculate ROC AUC for {name}")

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

    # Find the best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name}")

    # Cross-validation to ensure model robustness
    print("Performing cross-validation...")
    cv_scores = cross_val_score(best_model, X_train_selected, y_train_weighted_resampled, cv=StratifiedKFold(5), scoring='f1_weighted')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f}")

    # Create a complete pipeline for production
    print("Creating production pipeline...")

    # Define the pipeline steps
    pipeline_steps = [
        ('feature_selection', rfecv_no_pca),
        ('model', best_model)
    ]

    # Create the pipeline
    pipeline = Pipeline(pipeline_steps)

    # Fit the pipeline on the resampled data
    pipeline.fit(X_train_weighted_resampled, y_train_weighted_resampled)

    # Evaluate the pipeline
    pipeline_pred = pipeline.predict(X_test_weighted)
    pipeline_accuracy = accuracy_score(y_test_encoded, pipeline_pred)
    pipeline_f1 = f1_score(y_test_encoded, pipeline_pred, average='weighted')

    print(f"Pipeline - Accuracy: {pipeline_accuracy:.4f}, F1: {pipeline_f1:.4f}")

    # Now let's train a model with PCA data
    print("\n=== Training models with PCA ===\n")

    # Train a Random Forest on PCA data
    print("Training Random Forest on PCA data...")
    rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_pca.fit(X_train_pca_resampled, y_train_pca_resampled)

    # Evaluate the PCA model
    y_pred_pca = rf_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test_encoded, y_pred_pca)
    precision_pca = precision_score(y_test_encoded, y_pred_pca, average='weighted')
    recall_pca = recall_score(y_test_encoded, y_pred_pca, average='weighted')
    f1_pca = f1_score(y_test_encoded, y_pred_pca, average='weighted')

    print(f"PCA Random Forest - Accuracy: {accuracy_pca:.4f}, Precision: {precision_pca:.4f}, Recall: {recall_pca:.4f}, F1: {f1_pca:.4f}")
    print(classification_report(y_test_encoded, y_pred_pca, target_names=label_encoder.classes_))

    # Compare models with and without PCA
    print("\n=== Comparing models with and without PCA ===\n")
    print(f"Without PCA - Best model: {best_model_name}, F1: {results[best_model_name]['f1_score']:.4f}")
    print(f"With PCA - Random Forest, F1: {f1_pca:.4f}")

    # Choose the best overall model
    if f1_pca > results[best_model_name]['f1_score']:
        print("PCA model performs better!")
        final_model = rf_pca
        final_model_name = "Random Forest with PCA"
        use_pca = True
    else:
        print("Non-PCA model performs better!")
        final_model = best_model
        final_model_name = best_model_name
        use_pca = False

    # Save the pipeline and related data
    print("Saving model and data...")

    # Save both models and choose which one to use as the main model
    if f1_pca > results[best_model_name]['f1_score']:
        # PCA model is better
        # Save the PCA model
        joblib.dump(rf_pca, os.path.join(model_path, "advanced_symptom_model.pkl"))
        # Save the PCA transformer
        joblib.dump(pca, os.path.join(model_path, "advanced_pca_transformer.pkl"))
        # Save the scaler
        joblib.dump(scaler, os.path.join(model_path, "advanced_scaler.pkl"))
        # Save a flag indicating PCA is used
        with open(os.path.join(model_path, "advanced_model_config.json"), 'w') as f:
            json.dump({"use_pca": True, "best_model": "Random Forest with PCA"}, f, indent=2)
    else:
        # Non-PCA model is better
        # Save the pipeline
        joblib.dump(pipeline, os.path.join(model_path, "advanced_symptom_pipeline.pkl"))
        # Save the best model separately
        joblib.dump(best_model, os.path.join(model_path, "advanced_symptom_model.pkl"))
        # Save the feature selector
        joblib.dump(rfecv_no_pca, os.path.join(model_path, "advanced_feature_selector.pkl"))
        # Save a flag indicating PCA is not used
        with open(os.path.join(model_path, "advanced_model_config.json"), 'w') as f:
            json.dump({"use_pca": False, "best_model": best_model_name}, f, indent=2)

    # Save the label encoder (common for both models)
    joblib.dump(label_encoder, os.path.join(model_path, "advanced_label_encoder.pkl"))

    # Save symptoms list
    with open(os.path.join(model_path, "advanced_symptoms_list.json"), 'w') as f:
        json.dump(all_symptoms, f, indent=2)

    # Save selected features
    with open(os.path.join(model_path, "advanced_selected_features.json"), 'w') as f:
        json.dump(selected_features, f, indent=2)

    # Create a mapping of diseases to symptoms
    disease_symptom_map = {}
    for disease in df['prognosis'].unique():
        disease_rows = df[df['prognosis'] == disease]
        symptoms = set()
        for _, row in disease_rows.iterrows():
            for col in all_symptoms:
                if row[col] == 1:
                    symptoms.add(col)
        disease_symptom_map[disease] = sorted(list(symptoms))

    # Save disease-symptom mapping
    with open(os.path.join(model_path, "advanced_disease_symptom_map.json"), 'w') as f:
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
    with open(os.path.join(model_path, "advanced_disease_data.json"), 'w') as f:
        json.dump(disease_data, f, indent=2)

    # Save severity dictionary
    with open(os.path.join(model_path, "advanced_symptom_severity.json"), 'w') as f:
        json.dump(severity_dict, f, indent=2)

    # Save model performance metrics
    model_metrics = {
        "without_pca": {
            "best_model": best_model_name,
            "accuracy": float(results[best_model_name]['accuracy']),
            "precision": float(results[best_model_name]['precision']),
            "recall": float(results[best_model_name]['recall']),
            "f1_score": float(results[best_model_name]['f1_score']),
            "pipeline_accuracy": float(pipeline_accuracy),
            "pipeline_f1_score": float(pipeline_f1),
            "cross_validation_scores": [float(score) for score in cv_scores],
            "mean_cv_score": float(cv_scores.mean())
        },
        "with_pca": {
            "model": "Random Forest",
            "accuracy": float(accuracy_pca),
            "precision": float(precision_pca),
            "recall": float(recall_pca),
            "f1_score": float(f1_pca),
            "pca_components": int(n_components),
            "variance_retained": float(np.sum(pca.explained_variance_ratio_))
        }
    }

    with open(os.path.join(model_path, "advanced_model_metrics.json"), 'w') as f:
        json.dump(model_metrics, f, indent=2)

    # Calculate confusion matrix for the pipeline model
    cm = confusion_matrix(y_test_encoded, pipeline_pred)

    # Calculate per-class metrics
    per_class_metrics = {}
    for i, disease in enumerate(label_encoder.classes_):
        true_pos = cm[i, i]
        false_pos = cm[:, i].sum() - true_pos
        false_neg = cm[i, :].sum() - true_pos
        true_neg = cm.sum() - (true_pos + false_pos + false_neg)

        # Calculate metrics
        if true_pos + false_pos > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0

        if true_pos + false_neg > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        accuracy = (true_pos + true_neg) / cm.sum()

        per_class_metrics[disease] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'support': int((y_test_encoded == i).sum())
        }

    # Add per-class metrics to the model metrics
    model_metrics["per_class_metrics"] = per_class_metrics
    model_metrics["class_distribution"] = {
        disease: int(count) for disease, count in zip(
            label_encoder.classes_,
            np.bincount(y_test_encoded)
        )
    }

    # Update the model metrics file
    with open(os.path.join(model_path, "advanced_model_metrics.json"), 'w') as f:
        json.dump(model_metrics, f, indent=2)

    print("Advanced symptom prediction model training completed!")
    if f1_pca > results[best_model_name]['f1_score']:
        print(f"Best model: Random Forest with PCA")
        print(f"Accuracy: {accuracy_pca:.4f}")
        print(f"F1 Score: {f1_pca:.4f}")
    else:
        print(f"Best model: {best_model_name}")
        print(f"Pipeline Accuracy: {pipeline_accuracy:.4f}")
        print(f"Pipeline F1 Score: {pipeline_f1:.4f}")

    return model_path

if __name__ == "__main__":
    train_advanced_symptom_model()
