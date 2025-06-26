import os
import sys
import time

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ml.training.advanced_symptom_model import train_advanced_symptom_model

def train_advanced_models():
    """Train advanced prediction models"""
    print("Starting training of advanced prediction models...")
    
    # Train advanced symptom prediction model
    print("\n" + "="*50)
    print("TRAINING ADVANCED SYMPTOM PREDICTION MODEL")
    print("="*50)
    start_time = time.time()
    model_path = train_advanced_symptom_model()
    elapsed_time = time.time() - start_time
    print(f"Advanced symptom prediction model trained in {elapsed_time:.2f} seconds")
    print(f"Model saved to: {model_path}")
    
    print("\n" + "="*50)
    print("ALL ADVANCED MODELS TRAINED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    train_advanced_models()
