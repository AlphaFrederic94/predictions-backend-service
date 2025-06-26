import os
import requests
import json

class DockerChatbotService:
    def __init__(self):
        self.model_id = "aaditya/Llama3-OpenBioLLM-70B"
        self.api_url = "http://localhost:8000/v1/chat/completions"
        self.initialized = False
        self.available = False  # Flag to indicate if the model is available
        self.initialize()

    def initialize(self):
        """Initialize the model and check if it's available."""
        if not self.initialized:
            try:
                # Test if the model is available
                self._test_model_availability()
                self.initialized = True
                return True
            except Exception as e:
                print(f"Error initializing Docker chatbot: {str(e)}")
                self.initialized = True  # Mark as initialized even if it failed
                return False
        return True

    def _test_model_availability(self):
        """Test if the model is available for inference."""
        try:
            # Simple test query
            test_message = "Hello"

            # Try to get a response
            response = self._query_model(test_message, max_tokens=5)
            if "error" not in response:
                self.available = True
                print("Docker model is available and responding")
            else:
                self.available = False
                print(f"Docker model availability test failed: {response.get('error')}")
        except Exception as e:
            print(f"Docker model availability test failed: {str(e)}")
            self.available = False

    def _is_medical_question(self, question):
        """Determine if a question is medical-related."""
        # For the Docker model, we'll assume all questions are valid
        # as the model itself is specialized for medical content
        return True

    def _query_model(self, message, system_message=None, max_tokens=256):
        """Query the Docker-deployed model."""
        headers = {"Content-Type": "application/json"}
        
        # Prepare messages
        messages = [{"role": "user", "content": message}]
        
        # Add system message if provided
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        
        # Prepare the payload
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        # Add retry mechanism
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return {"response": content, "model": self.model_id}
                    else:
                        return {"response": "Received empty response from model", "model": self.model_id}
                else:
                    error_msg = f"Error {response.status_code}: {response.text}"
                    print(error_msg)
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return {"error": error_msg, "model": self.model_id}
            except Exception as e:
                error_msg = f"Exception during model query: {str(e)}"
                print(error_msg)
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return {"error": error_msg, "model": self.model_id}
        
        return {"error": "Maximum retries exceeded", "model": self.model_id}

    def generate_response(self, user_message, system_message=None, max_tokens=256):
        """Generate a response from the medical chatbot."""
        if not self.initialized:
            self.initialize()

        # If the model is not available, return a fallback response
        if not self.available:
            return {
                "response": "I'm sorry, but the medical AI model is currently unavailable. This could be due to access restrictions or server issues. Please try again later or contact support if the issue persists.",
                "model": self.model_id,
                "note": "This is a fallback response as the model could not be accessed."
            }

        # Query the model
        result = self._query_model(user_message, system_message, max_tokens)
        return result

# Singleton instance
docker_chatbot_service = DockerChatbotService()
