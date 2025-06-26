import os
import requests

# Try to import huggingface_hub, but provide a fallback if it's not available
try:
    from huggingface_hub import login
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    print("Warning: huggingface_hub module not found. Using fallback mode.")
    HUGGINGFACE_AVAILABLE = False

    # Define a dummy login function
    def login(token=None):
        print("Mock login to Hugging Face Hub (module not available)")
        return None

class MedicalChatbotService:
    def __init__(self):
        # Login to Hugging Face Hub
        self._login_to_huggingface()

        # Initialize the model
        # Use a smaller model that doesn't require a Pro subscription
        self.model_id = "google/flan-t5-small"
        self.initialized = False
        self.available = False  # Flag to indicate if the model is available

    def _login_to_huggingface(self):
        """Login to Hugging Face Hub using the token from environment variable."""
        # Use the provided token
        self.hf_token = os.environ.get("HF_TOKEN")
        try:
            login(token=self.hf_token)
            print("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            print(f"Error logging in to Hugging Face Hub: {str(e)}")

    def initialize(self):
        """Initialize the model and check if it's available."""
        if not self.initialized:
            try:
                # If huggingface_hub is not available, use fallback mode
                if not HUGGINGFACE_AVAILABLE:
                    print("Using fallback mode for medical chatbot (huggingface_hub not available)")
                    self.initialized = True
                    self.available = True  # Mark as available to use fallback responses
                    return True

                # Test if the model is available
                self._test_model_availability()
                self.initialized = True
                return True
            except Exception as e:
                print(f"Error initializing medical chatbot: {str(e)}")
                self.initialized = True  # Mark as initialized even if it failed
                return False
        return True

    def _test_model_availability(self):
        """Test if the model is available for inference."""
        try:
            # Simple test query
            test_message = "Hello"

            # Try to get a response
            response = self._query_huggingface(test_message, max_tokens=5)
            if "error" not in response:
                self.available = True
            else:
                self.available = False
                print(f"Model availability test failed: {response.get('error')}")
        except Exception as e:
            print(f"Model availability test failed: {str(e)}")
            self.available = False

    def _is_medical_question(self, question):
        """Determine if a question is medical-related."""
        # Common medical conditions and diseases
        medical_conditions = [
            'malaria', 'covid', 'coronavirus', 'flu', 'influenza', 'cold', 'pneumonia', 'bronchitis',
            'asthma', 'diabetes', 'hypertension', 'cancer', 'hiv', 'aids', 'tuberculosis', 'tb',
            'hepatitis', 'measles', 'chickenpox', 'mumps', 'rubella', 'polio', 'ebola', 'zika',
            'dengue', 'cholera', 'typhoid', 'meningitis', 'arthritis', 'osteoporosis', 'alzheimer',
            'parkinson', 'epilepsy', 'stroke', 'heart attack', 'heart disease', 'kidney disease',
            'liver disease', 'cirrhosis', 'copd', 'emphysema', 'leukemia', 'lymphoma', 'melanoma',
            'carcinoma', 'tumor', 'cyst', 'ulcer', 'hernia', 'appendicitis', 'gastritis', 'colitis',
            'crohn', 'ibs', 'gerd', 'acid reflux', 'gallstones', 'kidney stones', 'migraine',
            'concussion', 'fracture', 'sprain', 'strain', 'dislocation', 'sciatica', 'scoliosis',
            'fibromyalgia', 'lupus', 'ms', 'multiple sclerosis', 'als', 'parkinsons', 'dementia',
            'schizophrenia', 'bipolar', 'depression', 'anxiety', 'ptsd', 'ocd', 'adhd', 'autism',
            'down syndrome', 'cystic fibrosis', 'hemophilia', 'sickle cell', 'thalassemia', 'anemia'
        ]

        # Body organs and anatomical terms
        body_organs = [
            'pancreas', 'liver', 'kidney', 'heart', 'lung', 'brain', 'stomach', 'intestine', 'colon',
            'rectum', 'bladder', 'uterus', 'ovary', 'testicle', 'prostate', 'thyroid', 'adrenal',
            'pituitary', 'spleen', 'gallbladder', 'esophagus', 'trachea', 'bronchi', 'diaphragm',
            'skin', 'muscle', 'bone', 'joint', 'tendon', 'ligament', 'cartilage', 'nerve', 'artery',
            'vein', 'capillary', 'blood vessel', 'lymph node', 'lymphatic', 'immune system',
            'endocrine', 'reproductive', 'digestive', 'respiratory', 'circulatory', 'nervous',
            'skeletal', 'muscular', 'integumentary', 'urinary', 'eye', 'ear', 'nose', 'throat',
            'mouth', 'tongue', 'teeth', 'gum', 'salivary', 'pharynx', 'larynx', 'vocal cord',
            'spine', 'vertebra', 'rib', 'skull', 'femur', 'tibia', 'fibula', 'humerus', 'radius',
            'ulna', 'pelvis', 'hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist', 'hand', 'foot',
            'finger', 'toe', 'nail', 'hair', 'scalp', 'forehead', 'eyebrow', 'eyelid', 'pupil',
            'iris', 'retina', 'cornea', 'lens', 'optic nerve', 'cochlea', 'eardrum', 'auditory',
            'nasal', 'sinus', 'tonsil', 'adenoid', 'appendix', 'duodenum', 'jejunum', 'ileum',
            'cecum', 'ascending colon', 'transverse colon', 'descending colon', 'sigmoid colon',
            'rectum', 'anus', 'bile duct', 'pancreatic duct', 'ureter', 'urethra', 'fallopian tube',
            'vas deferens', 'epididymis', 'seminal vesicle', 'cervix', 'vagina', 'vulva', 'penis',
            'scrotum', 'placenta', 'umbilical cord', 'amniotic', 'fetus', 'embryo', 'zygote',
            'gamete', 'sperm', 'egg', 'ovum', 'chromosome', 'gene', 'dna', 'rna', 'cell', 'tissue',
            'organ', 'system', 'body'
        ]

        # General medical keywords
        medical_keywords = [
            'health', 'medical', 'disease', 'symptom', 'diagnosis', 'treatment', 'medicine', 'doctor',
            'hospital', 'patient', 'clinic', 'surgery', 'drug', 'prescription', 'therapy', 'cancer',
            'diabetes', 'heart', 'blood', 'pain', 'infection', 'virus', 'bacteria', 'allergy',
            'chronic', 'acute', 'condition', 'disorder', 'syndrome', 'illness', 'injury', 'wound',
            'fracture', 'bone', 'joint', 'muscle', 'nerve', 'brain', 'lung', 'liver', 'kidney',
            'stomach', 'intestine', 'colon', 'skin', 'rash', 'fever', 'cough', 'headache',
            'nausea', 'vomiting', 'diarrhea', 'constipation', 'fatigue', 'dizzy', 'swelling',
            'inflammation', 'immune', 'antibody', 'vaccine', 'vaccination', 'pandemic', 'epidemic',
            'outbreak', 'contagious', 'transmission', 'prevention', 'hygiene', 'sanitize',
            'disinfect', 'sterilize', 'mask', 'ppe', 'ventilator', 'oxygen', 'respiration',
            'breathing', 'pulse', 'heart rate', 'blood pressure', 'temperature', 'fever',
            'diet', 'nutrition', 'vitamin', 'mineral', 'supplement', 'exercise', 'fitness',
            'weight', 'obesity', 'anorexia', 'bulimia', 'mental health', 'depression', 'anxiety',
            'stress', 'trauma', 'ptsd', 'bipolar', 'schizophrenia', 'adhd', 'autism',
            'alzheimer', 'dementia', 'parkinson', 'stroke', 'seizure', 'epilepsy',
            'pregnancy', 'birth', 'fertility', 'contraception', 'menopause', 'menstruation',
            'std', 'sti', 'hiv', 'aids', 'herpes', 'chlamydia', 'gonorrhea', 'syphilis',
            'antibiotic', 'antiviral', 'antifungal', 'analgesic', 'nsaid', 'opioid',
            'steroid', 'insulin', 'chemotherapy', 'radiation', 'dialysis', 'transplant',
            'donor', 'recipient', 'genetic', 'dna', 'rna', 'chromosome', 'mutation',
            'hereditary', 'congenital', 'pathology', 'biopsy', 'autopsy', 'mortality',
            'morbidity', 'prognosis', 'remission', 'relapse', 'terminal', 'palliative',
            'hospice', 'euthanasia', 'dnr', 'advance directive', 'living will',
            'anatomy', 'physiology', 'histology', 'cytology', 'microbiology', 'immunology',
            'pharmacology', 'toxicology', 'epidemiology', 'biostatistics', 'public health',
            'occupational health', 'environmental health', 'global health', 'one health',
            'telemedicine', 'telehealth', 'ehealth', 'mhealth', 'digital health', 'ai in healthcare',
            'medical device', 'implant', 'prosthetic', 'orthotic', 'wheelchair', 'crutch',
            'walker', 'cane', 'hearing aid', 'glasses', 'contact lens', 'dental', 'orthodontic',
            'braces', 'filling', 'crown', 'root canal', 'extraction', 'implant', 'denture',
            'floss', 'mouthwash', 'toothpaste', 'toothbrush', 'gum disease', 'cavity',
            'plaque', 'tartar', 'enamel', 'dentin', 'pulp', 'nerve', 'root', 'crown',
            'molar', 'premolar', 'canine', 'incisor', 'wisdom tooth', 'baby tooth',
            'permanent tooth', 'primary tooth', 'deciduous tooth', 'adult tooth'
        ]

        # Common medical question patterns
        medical_patterns = [
            'what is', 'what are', 'how to', 'how do', 'why does', 'can you', 'should i',
            'is it', 'are there', 'when should', 'what causes', 'how can i', 'what happens',
            'tell me about', 'explain', 'describe', 'symptoms of', 'signs of', 'treatment for',
            'cure for', 'remedy for', 'medicine for', 'causes of', 'risk factors',
            'how long', 'how often', 'how much', 'side effects', 'complications',
            'prevention of', 'diagnosis of', 'test for', 'screening for'
        ]

        question_lower = question.lower()

        # Check for specific medical conditions first
        for condition in medical_conditions:
            if condition.lower() in question_lower:
                print(f"Medical condition detected: {condition}")
                return True

        # Check for body organs and anatomical terms
        for organ in body_organs:
            if organ.lower() in question_lower:
                print(f"Body organ/anatomical term detected: {organ}")
                return True

        # Check for medical keywords
        for keyword in medical_keywords:
            if keyword.lower() in question_lower:
                print(f"Medical keyword detected: {keyword}")
                return True

        # Check for medical question patterns with context
        for pattern in medical_patterns:
            if pattern.lower() in question_lower:
                # If pattern is found, look for medical context in the rest of the question
                for keyword in medical_keywords + medical_conditions:
                    if keyword.lower() in question_lower:
                        print(f"Medical pattern and keyword detected: {pattern}, {keyword}")
                        return True

        # If we get here, it's not a medical question
        print(f"Not a medical question: {question}")
        return False

    def _query_huggingface(self, message, system_message=None, max_tokens=256):
        """Query the Hugging Face API."""
        # If huggingface_hub is not available, return a fallback response
        if not HUGGINGFACE_AVAILABLE:
            return self._generate_fallback_response(message)

        API_URL = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        # Default system message if none provided
        if system_message is None:
            system_message = (
                "Answer the following question about medicine, healthcare, or human biology. "
                "Provide accurate, clear, and well-explained responses using medical terminology "
                "while keeping explanations accessible to a general audience. "
                "If the question is not related to medicine or healthcare, respond with: "
                "'Sorry, I can only assist with medical-related inquiries.'"
            )

        # Prepare the payload for T5 model
        # For T5 models, we need to prefix the input with an instruction
        medical_prefix = "Answer this medical question: "
        payload = {
            "inputs": medical_prefix + message,
            "parameters": {
                "max_new_tokens": min(max_tokens, 250),  # T5 has a limit of 250 tokens
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            },
            "options": {
                "wait_for_model": True,
                "timeout": 120  # Increase timeout to 120 seconds
            }
        }

        # Add retry mechanism
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=headers, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return {"response": result[0].get("generated_text", ""), "model": self.model_id}
                    else:
                        return {"response": str(result), "model": self.model_id}
                elif response.status_code == 500 and "too busy" in response.text.lower() and attempt < max_retries - 1:
                    # If the model is too busy and we have retries left, wait and try again
                    print(f"Model busy, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(retry_delay)
                    # Increase delay for next retry
                    retry_delay *= 2
                    continue
                else:
                    return {"error": f"{response.status_code} {response.reason}: {response.text}", "model": self.model_id}
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Request failed, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    import time
                    time.sleep(retry_delay)
                    # Increase delay for next retry
                    retry_delay *= 2
                    continue
                return {"error": str(e), "model": self.model_id}

    def _generate_fallback_response(self, message):
        """Generate a fallback response when the Hugging Face API is not available."""
        # Simple keyword-based response system
        message_lower = message.lower()

        # Common medical conditions
        if any(keyword in message_lower for keyword in ['diabetes', 'blood sugar', 'insulin']):
            return {
                "response": "Diabetes is a chronic condition characterized by high blood sugar levels. It occurs when the body cannot produce enough insulin or cannot effectively use the insulin it produces. Common symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision. Management typically involves monitoring blood sugar, medication or insulin therapy, healthy eating, and regular physical activity.",
                "model": "fallback-medical-bot"
            }
        elif any(keyword in message_lower for keyword in ['heart disease', 'cardiovascular', 'heart attack', 'stroke']):
            return {
                "response": "Heart disease refers to various conditions that affect the heart, including coronary artery disease, heart rhythm problems, and heart defects. Risk factors include high blood pressure, high cholesterol, smoking, diabetes, obesity, and family history. Prevention strategies include regular exercise, healthy diet, avoiding smoking, and managing stress.",
                "model": "fallback-medical-bot"
            }
        elif any(keyword in message_lower for keyword in ['covid', 'coronavirus', 'pandemic']):
            return {
                "response": "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms can include fever, cough, shortness of breath, fatigue, body aches, headache, loss of taste or smell, sore throat, and congestion. Prevention measures include vaccination, good hand hygiene, wearing masks in crowded settings, and maintaining physical distance when appropriate.",
                "model": "fallback-medical-bot"
            }
        elif any(keyword in message_lower for keyword in ['headache', 'migraine']):
            return {
                "response": "Headaches can have various causes, including tension, migraines, sinus issues, or underlying health conditions. Migraines are severe headaches often accompanied by nausea, sensitivity to light and sound, and visual disturbances. Treatment options include over-the-counter pain relievers, prescription medications, stress management, and identifying and avoiding triggers.",
                "model": "fallback-medical-bot"
            }
        elif any(keyword in message_lower for keyword in ['fever', 'temperature']):
            return {
                "response": "A fever is a temporary increase in body temperature, often due to an infection. Adults typically have a fever when their temperature is above 100.4°F (38°C). While fevers can be uncomfortable, they're generally a sign that your body is fighting an infection. Rest, hydration, and over-the-counter fever reducers can help manage symptoms. Seek medical attention for very high fevers or if accompanied by severe symptoms.",
                "model": "fallback-medical-bot"
            }
        elif any(keyword in message_lower for keyword in ['nutrition', 'diet', 'healthy eating']):
            return {
                "response": "A balanced diet includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats. It's recommended to limit processed foods, added sugars, and excessive salt. Proper nutrition supports overall health, helps maintain a healthy weight, and reduces the risk of chronic diseases like heart disease, diabetes, and certain cancers.",
                "model": "fallback-medical-bot"
            }
        elif any(keyword in message_lower for keyword in ['exercise', 'physical activity', 'workout']):
            return {
                "response": "Regular physical activity has numerous health benefits, including improved cardiovascular health, stronger muscles and bones, better weight management, enhanced mental health, and reduced risk of chronic diseases. Adults should aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous activity per week, plus muscle-strengthening activities on 2 or more days per week.",
                "model": "fallback-medical-bot"
            }
        elif any(keyword in message_lower for keyword in ['sleep', 'insomnia', 'tired']):
            return {
                "response": "Quality sleep is essential for physical and mental health. Adults typically need 7-9 hours of sleep per night. Poor sleep can contribute to health problems including weakened immunity, increased risk of heart disease and diabetes, impaired cognitive function, and mental health issues. Good sleep hygiene includes maintaining a regular sleep schedule, creating a restful environment, limiting screen time before bed, and avoiding caffeine and alcohol close to bedtime.",
                "model": "fallback-medical-bot"
            }
        else:
            return {
                "response": "I understand you have a question about health or medicine. While I'd like to provide a detailed response, I'm currently operating in a limited capacity. For specific medical advice, please consult with a healthcare professional. General health recommendations include maintaining a balanced diet, regular physical activity, adequate sleep, stress management, and regular check-ups with your healthcare provider.",
                "model": "fallback-medical-bot"
            }

    def generate_response(self, user_message, system_message=None, max_tokens=256):
        """Generate a response from the medical chatbot."""
        if not self.initialized:
            self.initialize()

        # Check if the question is medical-related
        is_medical = self._is_medical_question(user_message)

        # If not a medical question but contains the word "you", it might be asking about capabilities
        if not is_medical and "you" in user_message.lower():
            return {
                "response": "I'm a medical AI assistant designed to answer questions about health, medical conditions, treatments, and provide general health advice. I can help explain symptoms, discuss preventive measures, and provide information about various medical topics. Please ask me any health-related questions you may have.",
                "model": self.model_id
            }
        # If not a medical question
        elif not is_medical:
            return {
                "response": "Sorry, I can only assist with medical-related inquiries. Please ask me about health conditions, symptoms, treatments, or other medical topics.",
                "model": self.model_id
            }

        # If the model is not available, return a fallback response
        if not self.available:
            return {
                "response": "I'm sorry, but the medical AI model is currently unavailable. This could be due to access restrictions or server issues. Please try again later or contact support if the issue persists.",
                "model": self.model_id,
                "note": "This is a fallback response as the model could not be accessed."
            }

        # Query the model
        result = self._query_huggingface(user_message, system_message, max_tokens)

        return result

# Singleton instance
medical_chatbot_service = MedicalChatbotService()
