import os
import random

def test_chatbot_functionality():
    """Test the core functionality of the chatbot without importing the module."""
    print("Testing Medical Chatbot core functionality...")
    
    # Sample medical question
    question = "What are the symptoms of diabetes?"
    print(f"Question: {question}")
    
    # Determine if it's a medical question
    is_medical = is_medical_question(question)
    print(f"Is medical question: {is_medical}")
    
    if is_medical:
        # Categorize the question
        category = categorize_question(question)
        print(f"Category: {category}")
        
        # Extract the topic
        topic = extract_topic(question)
        print(f"Topic: {topic}")
        
        # Generate a response
        response = generate_sample_response(question, category, topic)
        print(f"Response: {response}")
        
        return True
    else:
        print("Not a medical question")
        return False

def is_medical_question(question):
    """Determine if a question is medical-related."""
    medical_keywords = [
        'health', 'medical', 'disease', 'symptom', 'diagnosis', 'treatment', 'medicine', 'doctor',
        'hospital', 'patient', 'clinic', 'surgery', 'drug', 'prescription', 'therapy', 'cancer',
        'diabetes', 'heart', 'blood', 'pain', 'infection', 'virus', 'bacteria', 'allergy'
    ]

    question_lower = question.lower()
    for keyword in medical_keywords:
        if keyword.lower() in question_lower:
            return True
    return False

def categorize_question(question):
    """Categorize the question into one of the sample response categories."""
    question_lower = question.lower()

    if any(keyword in question_lower for keyword in ['symptom', 'feel', 'pain', 'ache', 'hurt', 'sick']):
        return "symptoms"
    elif any(keyword in question_lower for keyword in ['prevent', 'avoid', 'risk', 'protect', 'safe']):
        return "prevention"
    elif any(keyword in question_lower for keyword in ['treat', 'cure', 'heal', 'medicine', 'drug', 'therapy']):
        return "treatment"
    else:
        return "general"

def extract_topic(question):
    """Extract the main health topic from the question."""
    question_lower = question.lower()

    # Check for specific health topics in the question
    if 'diabetes' in question_lower:
        return 'diabetes'
    elif any(word in question_lower for word in ['heart', 'cardiac', 'cardiovascular']):
        return 'heart'
    elif any(word in question_lower for word in ['headache', 'migraine']):
        return 'headache'
    elif any(word in question_lower for word in ['cold', 'flu', 'influenza']):
        return 'cold and flu'
    
    # If no specific topic is found, return a generic term
    return "your health concern"

def generate_sample_response(question, category, topic):
    """Generate a sample response based on the question category and topic."""
    # Sample responses for different categories
    sample_responses = {
        "symptoms": [
            "Common symptoms include fatigue, pain, and changes in bodily functions. It's important to track when these symptoms occur and their severity.",
            "Symptoms can vary widely between individuals. What you're describing could be related to several conditions.",
            "These symptoms should be evaluated by a healthcare professional for proper diagnosis."
        ],
        "prevention": [
            "Prevention often involves lifestyle changes such as regular exercise, balanced diet, and avoiding harmful substances.",
            "Regular screenings and check-ups are essential for early detection and prevention.",
            "Maintaining a healthy weight, staying active, and managing stress can help prevent many conditions."
        ],
        "treatment": [
            "Treatment options vary depending on the specific condition, its severity, and individual factors.",
            "Both medication and lifestyle modifications are often part of effective treatment plans.",
            "Always follow your healthcare provider's instructions when taking medications or following treatment plans."
        ],
        "general": [
            "It's important to consult with a healthcare professional for personalized medical advice.",
            "Medical knowledge is constantly evolving, and recommendations may change over time.",
            "Regular check-ups with your healthcare provider are essential for maintaining good health."
        ]
    }
    
    # Get a random response from the appropriate category
    response_text = random.choice(sample_responses[category])
    
    # Create a more detailed response based on the topic
    if topic == "diabetes" and category == "symptoms":
        response_text = "Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, extreme hunger, blurred vision, fatigue, and slow-healing sores. It's important to consult with a healthcare provider if you're experiencing these symptoms for proper diagnosis and treatment."
    
    # Add a personalized touch based on the user's question
    personalized_response = f"Regarding your question about {topic}: {response_text}"
    
    return personalized_response

if __name__ == "__main__":
    test_chatbot_functionality()
