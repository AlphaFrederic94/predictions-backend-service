#!/usr/bin/env python3
import requests
import json
import sys
import os

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print a header for the chatbot CLI."""
    print("\n" + "=" * 80)
    print("                      MEDICAL CHATBOT CLI TEST INTERFACE")
    print("=" * 80)
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'clear' to clear the screen.")
    print("=" * 80 + "\n")

def get_model_status():
    """Get the status of the chatbot model."""
    try:
        response = requests.get("http://localhost:8000/api/chatbot/status")
        if response.status_code == 200:
            status = response.json()
            if status.get("available", False):
                print(f"✅ Connected to model: {status.get('model_id', 'Unknown')}")
                print(f"   Model is {'available' if status.get('available') else 'unavailable'}")
            else:
                print(f"⚠️  Model is unavailable: {status.get('model_id', 'Unknown')}")
        else:
            print(f"❌ Failed to get model status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking model status: {str(e)}")

def send_message(message):
    """Send a message to the chatbot API and return the response."""
    try:
        response = requests.post(
            "http://localhost:8000/api/chatbot/chat",
            json={"message": message, "max_tokens": 512}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                return result["response"]
            else:
                return f"Error: Unexpected response format: {json.dumps(result)}"
        else:
            return f"Error: API returned status code {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main function to run the chatbot CLI."""
    clear_screen()
    print_header()
    get_model_status()
    
    print("\nBot: Hello! I'm your medical assistant. How can I help you today?\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Medical Chatbot CLI. Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            clear_screen()
            print_header()
            continue
        
        if not user_input.strip():
            continue
        
        print("\nProcessing your question...\n")
        response = send_message(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
