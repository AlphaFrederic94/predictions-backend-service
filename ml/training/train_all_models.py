import os
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.training.train_symptom_model import train_symptom_model
from ml.training.train_diabetes_model import train_diabetes_model
from ml.training.train_heart_model import train_heart_model

def train_all_models():
    """Train all prediction models"""
    print("Starting training of all prediction models...")
    
    # Train symptom prediction model
    print("\n" + "="*50)
    print("TRAINING SYMPTOM PREDICTION MODEL")
    print("="*50)
    start_time = time.time()
    symptom_model_path = train_symptom_model()
    elapsed_time = time.time() - start_time
    print(f"Symptom prediction model trained in {elapsed_time:.2f} seconds")
    print(f"Model saved to: {symptom_model_path}")
    
    # Train diabetes prediction model
    print("\n" + "="*50)
    print("TRAINING DIABETES PREDICTION MODEL")
    print("="*50)
    start_time = time.time()
    diabetes_model_path = train_diabetes_model()
    elapsed_time = time.time() - start_time
    print(f"Diabetes prediction model trained in {elapsed_time:.2f} seconds")
    print(f"Model saved to: {diabetes_model_path}")
    
    # Train heart disease prediction model
    print("\n" + "="*50)
    print("TRAINING HEART DISEASE PREDICTION MODEL")
    print("="*50)
    start_time = time.time()
    heart_model_path = train_heart_model()
    elapsed_time = time.time() - start_time
    print(f"Heart disease prediction model trained in {elapsed_time:.2f} seconds")
    print(f"Model saved to: {heart_model_path}")
    
    print("\n" + "="*50)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    train_all_models()
