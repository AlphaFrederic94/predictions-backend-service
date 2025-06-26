#!/usr/bin/env python3
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chatbot.medical_chatbot_service import MedicalChatbotService

def main():
    # Create the service
    print("Initializing medical chatbot service...")
    service = MedicalChatbotService()
    
    # Initialize the service
    print("Testing initialization...")
    initialized = service.initialize()
    print(f"Initialized: {initialized}")
    print(f"Available: {service.available}")
    print(f"Model ID: {service.model_id}")
    
    # Test with a simple question
    question = "What is the pancreas?"
    if len(sys.argv) > 1:
        question = sys.argv[1]
    
    print(f"\nTesting with question: {question}")
    print("Generating response...")
    
    start_time = time.time()
    response = service.generate_response(question)
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
