#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install advanced dependencies
pip install -r requirements_advanced.txt

# Create necessary directories
mkdir -p ml/models/symptoms

# Train advanced model
python train_advanced_models.py

# Restart the API server
echo "Training complete. Restart the API server to use the advanced model."
echo "Run: python -m uvicorn api.main:app --reload"
