import os
import requests
from huggingface_hub import InferenceClient

# Set your Hugging Face token
# os.environ["HF_TOKEN"] = "your_huggingface_token_here"

# Create an inference client
client = InferenceClient(
    model="ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025",
    token=os.environ["HF_TOKEN"]
)

# Test the model
def test_model():
    print("Testing the Bio-Medical-Llama model...")

    # Create a prompt
    messages = [
        {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"},
        {"role": "user", "content": "What are the symptoms of diabetes?"}
    ]

    try:
        # Generate a response
        response = client.chat_completion(
            messages,
            max_tokens=256,
            temperature=0.6,
            top_p=0.9
        )

        print("\nResponse from the model:")
        print(response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_model()
