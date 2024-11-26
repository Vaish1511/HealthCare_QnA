#!/usr/bin/env python
# coding: utf-8



from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser 
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import json
import numpy as np
import seaborn as sns
from llama_cpp import Llama


GEMINI_API_KEY = ""
HUGGINGFACEHUB_API_TOKEN = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# Initialize the Sentence Transformer model for intent classification
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")



# Define intents with keywords
intents = {
    "stomach": {
        "keywords": [
            "pain", "cramps", "nausea", "indigestion", "diarrhea", "bloating", "acid reflux", "constipation", 
            "ulcers", "food poisoning", "heartburn", "vomiting", "gassiness", "stomach flu", "gastritis", 
            "stomach ache", "IBS", "peptic ulcers", "acidity", "flatulence", "abdominal pain", "acid reflux disease", 
            "GERD", "feeling full quickly", "poor appetite", "belching", "sharp abdominal pain", "dull stomach ache", 
            "epigastric pain", "dyspepsia", "gurgling stomach sounds"
        ],
        "embedding_vector": None
    },
    "skin": {
        "keywords": [
            "rash", "itch", "eczema", "dry skin", "acne", "redness", "hives", "fungal infection", "psoriasis", 
            "dermatitis", "sunburn", "skin peeling", "discoloration", "swelling", "pimples", "spots", "cysts", 
            "skin tags", "lesions", "ulcers", "allergic reaction", "rosacea", "warts", "athlete's foot", "moles", 
            "boils", "flaky skin", "stretch marks", "pigmentation", "vitiligo", "skin irritation", "flaky scalp", 
            "blisters", "cracked skin", "sensitivity", "prickly heat", "bruising", "scars"
        ],
        "embedding_vector": None
    },
    "bp": {
        "keywords": [
            "hypertension", "blood pressure", "dizziness", "headache", "fatigue", "high blood pressure", 
            "low blood pressure", "hypotension", "chest pain", "palpitations", "fainting", "shortness of breath", 
            "blurred vision", "confusion", "nosebleeds", "lightheadedness", "pounding heartbeat", 
            "irregular heartbeat", "heart strain", "stroke risk", "renal issues", "diastolic pressure", 
            "systolic pressure", "blood flow issues", "heart failure", "hypertension crisis", "high pulse rate", 
            "low pulse rate", "cardiovascular disease", "pressure behind eyes"
        ],
        "embedding_vector": None
    },
    "diabetes": {
        "keywords": [
            "insulin", "sugar", "glucose", "thirst", "frequent urination", "weight loss", "blurred vision", 
            "fatigue", "tingling", "numbness", "slow healing wounds", "dry mouth", "diabetic neuropathy", "hunger", 
            "high blood sugar", "low blood sugar", "polyuria", "polydipsia", "glycemic index", "hyperglycemia", 
            "hypoglycemia", "ketoacidosis", "foot ulcers", "nerve pain", "eye problems", "retinopathy", "nephropathy", 
            "glucose intolerance", "sugar cravings", "sweating", "shakiness", "dizzy spells", "carb counting", 
            "A1C levels", "prediabetes", "metabolic syndrome"
        ],
        "embedding_vector": None
    }
}


# Precompute intent embeddings
for intent, data in intents.items():
    tokens = tokenizer(' '.join(data['keywords']), return_tensors="pt", padding=True, truncation=True)
    data['embedding_vector'] = model(**tokens).pooler_output.detach().numpy()


# Text preprocessing
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    replacements = {'bp': 'blood pressure', 'ut': 'urinary tract', 'hr': 'heart rate'}
    for abbr, full_form in replacements.items():
        text = text.replace(abbr, full_form)
    return text


# Intent classification
def classify_medical_intent(symptoms: str) -> str:
    preprocessed_symptoms = preprocess_text(symptoms)
    tokens = tokenizer(preprocessed_symptoms, return_tensors="pt", padding=True, truncation=True)
    symptoms_embedding = model(**tokens).pooler_output.detach().numpy()

    similarities = [
        cosine_similarity(symptoms_embedding, intent_data['embedding_vector'])[0][0]
        for intent_data in intents.values()
    ]

    keyword_scores = [
        sum(keyword in preprocessed_symptoms for keyword in intent_data['keywords'])
        for intent_data in intents.values()
    ]

    ensemble_scores = [
        0.7 * similarity + 0.3 * keyword_match
        for similarity, keyword_match in zip(similarities, keyword_scores)
    ]

    best_intent_index = np.argmax(ensemble_scores)
    return list(intents.keys())[best_intent_index]



def plot_enhanced_fishbone(disease: str, causes: Dict[str, List[str]]) -> plt.Figure:
    """
    Create an enhanced fishbone diagram with detailed sub-branches
    """
    # Setup
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 12), facecolor='#f0f2f6')

    # Main spine parameters
    spine_length = 14
    spine_start = 2
    spine_end = spine_start + spine_length

    # Draw main spine with arrow
    ax.arrow(spine_start, 0, spine_length, 0,
            head_width=0.4, head_length=0.6,
            fc='black', ec='black', lw=2)

    # Calculate positions
    num_causes = len(causes)
    spacing = spine_length / (num_causes + 1)
    branch_length = 3.5
    angle = 45

    # Calculate branch geometry
    dx = branch_length * np.cos(np.deg2rad(angle))
    dy = branch_length * np.sin(np.deg2rad(angle))

    # Use a colormap for different categories
    colors = plt.cm.Pastel1(np.linspace(0, 1, num_causes))

    # Draw branches and sub-branches
    for i, (cause, subcauses) in enumerate(causes.items()):
        x_pos = spine_start + (i + 1) * spacing

        # Alternate between top and bottom
        if i % 2 == 0:
            y_end = dy
            sub_y_offset = 0.5
            va = 'bottom'
        else:
            y_end = -dy
            sub_y_offset = -0.5
            va = 'top'

        # Draw main branch
        color = colors[i]
        ax.plot([x_pos, x_pos + dx], [0, y_end],
                color=color, lw=2, zorder=2)

        # Add main cause text
        ax.text(x_pos + dx, y_end + sub_y_offset,
                cause.upper(),
                ha='center',
                va=va,
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white',
                         edgecolor=color,
                         boxstyle='round,pad=0.5',
                         alpha=0.9))

        # Add sub-branches
        for j, subcause in enumerate(subcauses):
            # Calculate sub-branch positions
            sub_ratio = (j + 1) / (len(subcauses) + 1)
            sub_x = x_pos + dx * sub_ratio
            sub_y = y_end * sub_ratio

            # Draw sub-branch
            ax.plot([sub_x, sub_x + dx/2],
                   [sub_y, sub_y],
                   color=color, lw=1, zorder=2)

            # Add sub-cause text
            ax.text(sub_x + dx/2 + 0.1,
                   sub_y,
                   subcause,
                   ha='left',
                   va='center',
                   fontsize=8,
                   bbox=dict(facecolor='white',
                            edgecolor=color,
                            alpha=0.7,
                            boxstyle='round,pad=0.3'))

        # Add decorative elements
        ax.plot(x_pos, 0, 'o', color=color, markersize=6, zorder=3)

    # Add problem statement
    ax.text(spine_end + 0.7, 0,
            disease.upper(),
            ha='left',
            va='center',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='lightgray',
                     edgecolor='gray',
                     boxstyle='round,pad=0.5'))

    # Styling
    plt.title('Enhanced Root Cause Analysis (Ishikawa Diagram)',
              pad=20,
              fontsize=14,
              fontweight='bold')

    # Set limits and remove axes
    margin = 2
    ax.set_xlim(0, spine_end + 4)
    ax.set_ylim(-branch_length - margin, branch_length + margin)
    ax.axis('off')

    plt.tight_layout()
    return fig



def generate_interview_questions(initial_symptoms: str, category: str, gemini_model) -> List[str]:
    """Generate 5 specific interview questions using Gemini API based on initial symptoms and category."""
    prompt = f"""Given a patient with {category}-related symptoms: '{initial_symptoms}',
generate exactly 5 specific medical interview questions to understand their condition better.
Focus on gathering important diagnostic information for {category} conditions.

Return ONLY a JSON array of 5 questions in this exact format:
[
    "Question 1 text here",
    "Question 2 text here",
    "Question 3 text here",
    "Question 4 text here",
    "Question 5 text here"
]"""

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Handle potential formatting issues
        if not response_text.startswith('['):
            # Try to extract JSON array if it's buried in additional text
            import re
            json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            else:
                # Fallback: Convert response to list format
                questions = [q.strip() for q in response_text.split('\n') if q.strip()]
                return questions[:5]
        
        questions = json.loads(response_text)
        
        # Ensure exactly 5 questions
        if len(questions) < 5:
            questions.extend([
                "How long have you been experiencing these symptoms?",
                "Have you noticed any patterns or triggers?",
                "Are there any other symptoms you've experienced?",
                "Have you made any recent lifestyle changes?",
                "Have you tried any treatments or medications?"
            ][:5 - len(questions)])
        
        return questions[:5]
    
    except json.JSONDecodeError:
        # Fallback questions based on category
        fallback_questions = {
            "bp": [
                "How often do you check your blood pressure?",
                "Do you experience headaches or dizziness?",
                "What is your typical salt intake?",
                "Do you have a family history of hypertension?",
                "What is your current exercise routine?"
            ],
            "diabetes": [
                "When did you last check your blood sugar?",
                "Have you noticed increased thirst or urination?",
                "What is your typical daily diet?",
                "Do you have a family history of diabetes?",
                "How often do you exercise?"
            ],
            "skin": [
                "How long have you had this skin condition?",
                "Is there any itching or pain?",
                "Have you used any new products recently?",
                "Does the condition worsen at any particular time?",
                "Have you noticed any triggers?"
            ],
            "stomach": [
                "When did your stomach problems begin?",
                "How would you describe the pain or discomfort?",
                "Are symptoms related to eating specific foods?",
                "Have you noticed any changes in appetite?",
                "Do you experience nausea or vomiting?"
            ]
        }
        
        return fallback_questions.get(category.lower(), [
            "How long have you been experiencing these symptoms?",
            "Have you noticed any patterns or triggers?",
            "Are there any other symptoms you've experienced?",
            "Have you made any recent lifestyle changes?",
            "Have you tried any treatments or medications?"
        ])


# In[20]:


def conduct_interview(questions: List[str], category: str, llm) -> Dict:
    """Conduct the interview using provided questions and gather responses."""
    interview_data = {
        'intent': category,
        'initial_symptoms': questions[0],
        'detailed_responses': {}
    }
    
    print("\nStarting detailed medical interview...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        user_response = input("Your answer: ").strip()
        
        # Modified prompt to be more explicit about the required response format
        prompt = f"""As a medical professional, respond to this patient statement with empathy: '{user_response}'

Requirements:
- Respond directly to the patient
- Show understanding of their situation
- Keep response to 2-3 sentences
- Do not include any instructions or labels
- Start response with 'I' or 'Thank you' or similar direct phrases

For example, if patient says they have a headache, respond like:
"I understand you're experiencing head pain. Let's work together to identify the cause and find appropriate relief."

Now provide your response:"""
        
        # Get response and handle empty cases
        chatbot_response = get_llm_response(llm, prompt)
        clean_response = clean_llm_response(chatbot_response)
        
        # If we got an empty response after cleaning, use a fallback response
        if not clean_response:
            clean_response = generate_fallback_response(user_response)
            
        print(f"\nAssistant: {clean_response}\n")
        interview_data['detailed_responses'][f"Q{i}"] = user_response
    
    return interview_data

def clean_llm_response(response: str) -> str:
    """Clean and validate the LLM response."""
    if not response:
        return ""
    
    # List of words that indicate instruction text rather than actual response
    instruction_indicators = [
        "instructions:", "example:", "note:", "response should", "requirements:", 
        "remember to", "make sure to", "the response", "your response",
        "if they", "if patient", "keep it", "be brief", "respond with"
    ]
    
    # Get all non-empty lines
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    
    for line in lines:
        # Skip if line is too short
        if len(line) < 10:
            continue
            
        # Skip if line starts with common prefixes
        if any(line.lower().startswith(prefix) for prefix in [
            "assistant:", "ai:", "chatbot:", "response:", "answer:", 
            "example:", "note:", "question:"
        ]):
            continue
            
        # Skip if line contains instruction indicators
        if any(indicator in line.lower() for indicator in instruction_indicators):
            continue
            
        # Skip if line looks like a template or placeholder
        if '[' in line or ']' in line or '{' in line or '}' in line:
            continue
            
        # Line passes all checks - likely a valid response
        return line
    
    return ""

def generate_fallback_response(user_response: str) -> str:
    """Generate a fallback response when the LLM response is empty or invalid."""
    # Convert user response to lowercase once
    response_lower = user_response.lower()
    
    # Check for symptoms first (more specific)
    symptoms = ['pain', 'ache', 'hurt', 'dizzy', 'nausea', 'sick', 'fever', 
               'cough', 'tired', 'exhausted', 'headache', 'sore']
    if any(symptom in response_lower for symptom in symptoms):
        return "I hear that you're not feeling well, and I want you to know that your symptoms are being taken seriously. We'll work together to understand what's happening and find the right approach to help you feel better."
    
    # Check for negative responses
    if not user_response or response_lower in ['no', 'none', 'n/a', 'nope', 'nothing']:
        return "Thank you for letting me know. Please don't hesitate to mention if you experience any new symptoms or concerns. Your health is our priority."
    
    # Check for medication or treatment related responses
    if any(word in response_lower for word in ['medicine', 'medication', 'pill', 'drug', 'treatment']):
        return "Thank you for sharing these details about your medication history. This information is very helpful for understanding your situation and planning appropriate care."
    
    # Default response
    return "I appreciate you sharing this information with me. It helps us better understand your situation so we can provide the most appropriate care for your needs."



def generate_comprehensive_analysis(interview_data: Dict, category: str, gemini_model, llm) -> Dict:
    """Generate comprehensive medical analysis using both Gemini and Llama."""
    analysis_prompt = f"""Medical Analysis Request:
Patient Concern: {interview_data['intent'].capitalize()} Related Health Issue
Initial Symptoms: {interview_data['initial_symptoms']}
Detailed Interview Responses:
{chr(10).join([f"{k}: {v}" for k, v in interview_data['detailed_responses'].items()])}

Provide a detailed medical analysis in exactly this format using markdown:

**Possible Medical Diagnoses**
• First possible diagnosis with brief explanation
• Second possible diagnosis with brief explanation
• Third possible diagnosis with brief explanation

**Recommended Medical Tests**
• First recommended test with brief explanation
• Second recommended test with brief explanation
• Third recommended test with brief explanation

**Lifestyle and Dietary Recommendations**
• First lifestyle recommendation with brief explanation
• Second lifestyle recommendation with brief explanation
• Third lifestyle recommendation with brief explanation

**Signs Requiring Immediate Attention**
• First warning sign with brief explanation
• Second warning sign with brief explanation
• Third warning sign with brief explanation

**Treatment Approaches**
• First treatment approach with brief explanation
• Second treatment approach with brief explanation
• Third treatment approach with brief explanation"""

    try:
        # Get analysis from Gemini
        analysis_response = gemini_model.generate_content(analysis_prompt)
        response_text = analysis_response.text.strip()
        
        # Parse the response into structured format
        medical_analysis = parse_formatted_response(response_text)
        
        # If parsing failed or returned empty, use fallback
        if not medical_analysis:
            medical_analysis = get_fallback_analysis(category)
            
    except Exception as e:
        print(f"Error generating medical analysis: {str(e)}")
        medical_analysis = get_fallback_analysis(category)

    # Generate root cause analysis
    root_cause_prompt = f"""Based on the patient's {category} condition and responses:
{chr(10).join([f"{k}: {v}" for k, v in interview_data['detailed_responses'].items()])}

Provide a root cause analysis in exactly this format:

**Dietary Factors**
• First dietary factor with explanation
• Second dietary factor with explanation

**Stress Factors**
• First stress factor with explanation
• Second stress factor with explanation

**Lifestyle Factors**
• First lifestyle factor with explanation
• Second lifestyle factor with explanation"""

    try:
        root_cause_response = gemini_model.generate_content(root_cause_prompt)
        root_cause_text = root_cause_response.text.strip()
        root_cause_data = parse_formatted_response(root_cause_text)
        
        if not root_cause_data:
            root_cause_data = get_root_cause_template(category)
            
    except Exception as e:
        print(f"Error generating root cause analysis: {str(e)}")
        root_cause_data = get_root_cause_template(category)

    
    # Generate root cause analysis based on category
    root_cause_template = {
        "stomach": {
            "diet": [], "stress": [], "medication": [], "infection": [],
            "lifestyle": [], "hydration": [], "allergies": []
        },
        "skin": {
            "allergy": [], "hygiene": [], "environment": [], "genetics": [],
            "nutrition": [], "stress": [], "cosmetic_use": []
        },
        "bp": {
            "diet": [], "lifestyle": [], "stress": [], "physical_activity": [],
            "salt_intake": [], "sleep_disorders": []
        },
        "diabetes": {
            "diet": [], "exercise": [], "genetics": [], "insulin_resistance": [],
            "obesity": [], "stress": [], "medication": []
        }
    }
    root_cause_prompt1 = f"""Based on the patient's {category} condition and responses:
{json.dumps(interview_data['detailed_responses'], indent=2)}

Return EXACTLY this JSON structure for {category} with 2-3 specific items per category:
{json.dumps(get_root_cause_template(category), indent=2)}"""

    try:
        root_cause_response = gemini_model.generate_content(root_cause_prompt1)
        response_text = root_cause_response.text.strip()
        
        if not response_text.startswith('{'):
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            else:
                return {"medical_analysis": medical_analysis, "root_cause_data1": get_root_cause_template(category)}
        
        root_cause_data1 = json.loads(response_text)
    except (json.JSONDecodeError, Exception):
        root_cause_data1 = get_root_cause_template(category)

    # Ensure we return valid dictionaries
    result = {
        "medical_analysis": medical_analysis or get_fallback_analysis(category),
        "root_cause_data": root_cause_data or get_root_cause_template(category),
        "root_cause_data1": root_cause_data1
    }
    
    return result

def parse_formatted_response(text: str) -> Dict:
    """Parse the formatted text response into a structured dictionary."""
    result = {}
    current_section = None
    current_items = []
    
    if not text:
        return None
        
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers (bolded text)
        if line.startswith('**') and line.endswith('**'):
            # Save previous section if it exists
            if current_section and current_items:
                key = current_section.lower().replace(' ', '_')
                result[key] = current_items
                current_items = []
            
            # Start new section
            current_section = line.strip('*').strip()
            continue
            
        # Check for bullet points
        if line.startswith('•'):
            item = line[1:].strip()
            if item:  # Only add non-empty items
                current_items.append(item)
                
    # Don't forget to save the last section
    if current_section and current_items:
        key = current_section.lower().replace(' ', '_')
        result[key] = current_items
        
    # Verify we got some content
    return result if result else None

def display_analysis_results(analysis_results: Dict):
    """Display the analysis results in a formatted way."""
    if not analysis_results or not analysis_results.get("medical_analysis"):
        print("\nError: No analysis results available.")
        return
        
    print("\n=== Comprehensive Medical Analysis ===\n")
    
    # Display medical analysis
    medical_analysis = analysis_results["medical_analysis"]
    for key, value in medical_analysis.items():
        section_title = key.replace('_', ' ').title()
        print(f"\n**{section_title}**")
        if isinstance(value, list):
            for item in value:
                print(f"• {item}")
        else:
            print(f"• {value}")
    
    print("\n=== Root Cause Analysis ===\n")
    
    # Display root cause analysis
    root_cause_data = analysis_results["root_cause_data"]
    for key, value in root_cause_data.items():
        section_title = key.replace('_', ' ').title()
        print(f"\n**{section_title}**")
        if isinstance(value, list):
            for item in value:
                print(f"• {item}")
        else:
            print(f"• {value}")

def get_fallback_analysis(category: str) -> Dict:
    """Provide fallback analysis if Gemini API fails."""
    fallback_analyses = {
        "bp": {
            "possible_diagnoses": [
                "Essential Hypertension",
                "Secondary Hypertension",
                "White Coat Hypertension"
            ],
            "recommended_tests": [
                "24-hour Blood Pressure Monitoring",
                "ECG",
                "Basic Blood Work"
            ],
            "lifestyle_recommendations": [
                "Reduce Salt Intake",
                "Regular Exercise",
                "Stress Management"
            ],
            "immediate_attention_signs": [
                "Severe Headache",
                "Chest Pain",
                "Vision Problems"
            ],
            "treatment_approaches": [
                "Lifestyle Modifications",
                "Blood Pressure Medications",
                "Regular Monitoring"
            ]
        },
        "diabetes": {
            "possible_diagnoses": [
                "Type 2 Diabetes",
                "Prediabetes",
                "Insulin Resistance"
            ],
            "recommended_tests": [
                "HbA1c Test",
                "Fasting Blood Sugar",
                "Glucose Tolerance Test"
            ],
            "lifestyle_recommendations": [
                "Balanced Diet",
                "Regular Exercise",
                "Weight Management"
            ],
            "immediate_attention_signs": [
                "Very High Blood Sugar",
                "Severe Dehydration",
                "Confusion or Drowsiness"
            ],
            "treatment_approaches": [
                "Diet Control",
                "Oral Medications",
                "Blood Sugar Monitoring"
            ]
        },
        "skin": {
            "possible_diagnoses": [
                "Eczema",
                "Psoriasis",
                "Acne",
                "Skin Allergies",
                "Fungal Infections"
            ],
            "recommended_tests": [
                "Skin Patch Test",
                "Biopsy (if required)",
                "Allergy Test"
            ],
            "lifestyle_recommendations": [
                "Use of Gentle Cleansers",
                "Hydration and Moisturizing",
                "Avoiding Known Allergens"
            ],
            "immediate_attention_signs": [
                "Severe Rash with Swelling",
                "Skin Infection with Fever",
                "Rapidly Spreading Lesions"
            ],
            "treatment_approaches": [
                "Topical Ointments",
                "Antihistamines",
                "Prescription Medications (e.g., Steroids)"
            ]
        },
        "stomach": {
            "possible_diagnoses": [
                "Gastritis",
                "Irritable Bowel Syndrome (IBS)",
                "Acid Reflux (GERD)",
                "Peptic Ulcer",
                "Stomach Infection"
            ],
            "recommended_tests": [
                "Endoscopy",
                "Stool Test",
                "Helicobacter Pylori Test"
            ],
            "lifestyle_recommendations": [
                "Eating Smaller Meals",
                "Avoiding Spicy or Acidic Foods",
                "Stress Management"
            ],
            "immediate_attention_signs": [
                "Severe Abdominal Pain",
                "Blood in Stool or Vomit",
                "Unexplained Weight Loss"
            ],
            "treatment_approaches": [
                "Antacids or Acid Blockers",
                "Probiotics",
                "Medication for H. Pylori (if present)"
            ]
        }

    }
    
    return fallback_analyses.get(category.lower(), {
        "possible_diagnoses": ["Requires Medical Evaluation"],
        "recommended_tests": ["Consult Healthcare Provider"],
        "lifestyle_recommendations": ["Follow General Health Guidelines"],
        "immediate_attention_signs": ["Severe Symptoms", "Persistent Problems"],
        "treatment_approaches": ["Professional Medical Assessment"]
    })

def get_root_cause_template(category: str) -> Dict:
    """Return the template for root cause analysis with sample data."""
    templates = {
        "stomach": {
            "diet": ["Irregular eating patterns", "Spicy food consumption"],
            "stress": ["Work-related stress", "Anxiety"],
            "medication": ["Recent antibiotics", "NSAIDs"],
            "infection": ["Possible H. pylori", "Food-borne infection"],
            "lifestyle": ["Late night eating", "Fast food consumption"],
            "hydration": ["Inadequate water intake", "Excess caffeine"],
            "allergies": ["Food sensitivities", "Lactose intolerance"]
        },
        "skin": {
            "allergy": ["Contact dermatitis", "Environmental allergens"],
            "hygiene": ["Cleansing routine", "Product usage"],
            "environment": ["Sun exposure", "Pollution"],
            "genetics": ["Family history", "Predisposition"],
            "nutrition": ["Vitamin deficiency", "Diet impact"],
            "stress": ["Psychological factors", "Hormonal changes"],
            "cosmetic_use": ["Reaction to products", "Skin barrier damage"]
        },
        "bp": {
            "diet": ["Salt intake", "Processed foods"],
            "lifestyle": ["Sedentary behavior", "Smoking"],
            "stress": ["Work pressure", "Anxiety"],
            "physical_activity": ["Exercise routine", "Daily movement"],
            "salt_intake": ["Hidden sodium", "Dietary habits"],
            "sleep_disorders": ["Sleep apnea", "Insomnia"]
        },
        "diabetes": {
            "diet": ["Carbohydrate intake", "Sugar consumption"],
            "exercise": ["Activity level", "Fitness routine"],
            "genetics": ["Family history", "Genetic factors"],
            "insulin_resistance": ["Metabolic factors", "Body composition"],
            "obesity": ["Weight status", "Fat distribution"],
            "stress": ["Hormonal impact", "Lifestyle factors"],
            "medication": ["Current medications", "Treatment adherence"]
        }
    }
    
    return templates.get(category.lower(), {
        "general_factors": ["To be evaluated", "Requires assessment"],
        "lifestyle": ["To be determined", "Needs analysis"],
        "medical": ["Pending evaluation", "Professional assessment needed"]
    })




def get_llm_response(llm, prompt: str, max_tokens: int = 256) -> str:
    """Get response from Llama model with proper formatting."""
    try:
        response = llm(prompt, max_tokens=max_tokens)
        return response['choices'][0]['text'].strip()
    except ValueError:
        shortened_prompt = prompt[-500:]
        response = llm(shortened_prompt, max_tokens=max_tokens)
        return response['choices'][0]['text'].strip()



def main():
    # Initialize models
    llm = Llama.from_pretrained(
        repo_id="tensorblock/Llama3-Aloe-8B-Alpha-GGUF",
        filename="Llama3-Aloe-8B-Alpha-Q2_K.gguf",
        n_ctx=2048
    )

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

    print("\nWelcome to the Enhanced Healthcare Assistant!")
    print("Please describe your symptoms:\n")
    
    initial_input = input("You: ").strip()
    detected_category = classify_medical_intent(initial_input)  # You have this function
    
    # Generate interview questions using Gemini
    questions = generate_interview_questions(initial_input, detected_category, gemini_model)
    
    # Conduct interview
    interview_data = conduct_interview(questions, detected_category, llm)
    
    # Generate comprehensive analysis
    analysis_results = generate_comprehensive_analysis(
        interview_data, detected_category, gemini_model, llm
    )

    
    
    # # Display results
    # print("\n=== Comprehensive Medical Analysis ===\n")
    # for key, value in analysis_results["medical_analysis"].items():
    #     print(f"\n{key.replace('_', ' ').title()}:")
    #     if isinstance(value, list):
    #         for item in value:
    #             print(f"• {item}")
    #     else:
    #         print(value)
    display_analysis_results(analysis_results)
    
    # Create and display fishbone diagram
    plot_enhanced_fishbone(
        detected_category.title(),
        analysis_results["root_cause_data1"]
    )
    plt.show()



if __name__ == "__main__":
    main()


