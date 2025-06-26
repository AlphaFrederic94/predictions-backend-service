#!/usr/bin/env python3
import requests
import sys

def test_status():
    """Test the status endpoint."""
    try:
        response = requests.get("http://localhost:8000/api/chatbot/status")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")

def test_chat(message):
    """Test the chat endpoint with a message."""
    try:
        response = requests.post(
            "http://localhost:8000/api/chatbot/chat",
            json={"message": message, "max_tokens": 512}
        )
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    print("Testing status endpoint...")
    test_status()
    
    if len(sys.argv) > 1:
        message = sys.argv[1]
        print(f"\nTesting chat endpoint with message: {message}")
        test_chat(message)
    else:
        print("\nTesting chat endpoint with default message: 'What is diabetes?'")
        test_chat("What is diabetes?")
