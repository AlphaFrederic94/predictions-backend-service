import os
import sys
import time

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(base_path)

from ml.training.finetune_symptom_model import finetune_symptom_model
from ml.training.finetune_heart_model import finetune_heart_model

def finetune_all_models():
    """Fine-tune all prediction models"""
    print("Starting fine-tuning of all prediction models...")
    
    # Fine-tune symptom prediction model
    print("\n" + "="*50)
    print("FINE-TUNING SYMPTOM PREDICTION MODEL")
    print("="*50)
    start_time = time.time()
    symptom_model_path = finetune_symptom_model()
    elapsed_time = time.time() - start_time
    print(f"Symptom prediction model fine-tuned in {elapsed_time:.2f} seconds")
    print(f"Model saved to: {symptom_model_path}")
    
    # Fine-tune heart disease prediction model
    print("\n" + "="*50)
    print("FINE-TUNING HEART DISEASE PREDICTION MODEL")
    print("="*50)
    start_time = time.time()
    heart_model_path = finetune_heart_model()
    elapsed_time = time.time() - start_time
    print(f"Heart disease prediction model fine-tuned in {elapsed_time:.2f} seconds")
    print(f"Model saved to: {heart_model_path}")
    
    print("\n" + "="*50)
    print("ALL MODELS FINE-TUNED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    finetune_all_models()
