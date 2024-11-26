import streamlit as st
from streamlit_chat import message
from HealthCareChatBot import (
    classify_medical_intent,
    generate_interview_questions,
    generate_comprehensive_analysis,
    plot_enhanced_fishbone,
    Llama, genai
)
import base64
from PIL import Image
import pytesseract

# Page Configurations
st.set_page_config(
    page_title="Healthcare Chatbot",
    page_icon="💬",
    layout="wide",
)
def format_analysis_output(analysis_data):
    """
    Formats the comprehensive medical analysis for display in Streamlit with improved readability.
    """
    formatted_output = ""
    for key, value in analysis_data.items():
        if key == "root_cause_data1":  # Skip this field
            continue
        
        # Special handling for Possible Medical Diagnoses
        if key == "Possible_medical_diagnoses":
            formatted_output += "### Possible Medical Diagnoses\n\n"
            
            # Check if value is a list and handle accordingly
            if isinstance(value, list):
                for diagnosis in value:
                    # Remove any extra ** if present
                    diagnosis = diagnosis.strip('*')
                    
                    # Split diagnosis into title and description
                    if ':**' in diagnosis:
                        title, description = diagnosis.split(':**', 1)
                        formatted_output += f"#### {title.strip()}\n\n{description.strip()}\n\n"
                    else:
                        formatted_output += f"{diagnosis}\n\n"
            elif isinstance(value, str):
                # Handle case where it might be a single string
                formatted_output += f"{value}\n\n"
            
            continue
        # Standard formatting for other sections
        formatted_output += f"### {key.replace('_', ' ').capitalize()}\n\n"
        
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                formatted_output += f"**{sub_key.capitalize()}:** {sub_value}\n\n"
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                formatted_output += f"{idx + 1}. {item}\n\n"
        else:
            formatted_output += f"{value}\n\n"
        
        # Rest of the formatting remains the same
        # ... (previous code)
    
    return formatted_output

# def format_analysis_output(analysis_data):
#     """
#     Formats the comprehensive medical analysis for display in Streamlit with improved readability.
#     """
#     formatted_output = ""
#     for key, value in analysis_data.items():
#         if key == "root_cause_data1":  # Skip this field
#             continue
        
#         # Special handling for Possible Medical Diagnoses
#         if key == "Possible_medical_diagnoses":
#             formatted_output += "### Possible Medical Diagnoses\n\n"
#             for diagnosis in value:
#                 # Remove leading and trailing ** if present
#                 diagnosis = diagnosis.strip('*')
                
#                 # Split diagnosis into title and description
#                 if ':**' in diagnosis:
#                     title, description = diagnosis.split(':**', 1)
#                     formatted_output += f"**{title.strip()}**\n\n{description.strip()}\n\n"
#                 else:
#                     formatted_output += f"{diagnosis}\n\n"
#             continue
        
#         # Standard formatting for other sections
#         formatted_output += f"### {key.replace('_', ' ').capitalize()}\n\n"
        
#         if isinstance(value, dict):
#             for sub_key, sub_value in value.items():
#                 formatted_output += f"**{sub_key.capitalize()}:** {sub_value}\n\n"
#         elif isinstance(value, list):
#             for idx, item in enumerate(value):
#                 formatted_output += f"{idx + 1}. {item}\n\n"
#         else:
#             formatted_output += f"{value}\n\n"
    
#     return formatted_output


# OCR and Report Analysis Functions
def convert_image_to_base64(uploaded_file):
    """Convert uploaded image to base64."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        encoded_string = base64.b64encode(bytes_data).decode()
        return encoded_string
    return None

def extract_text_from_image(uploaded_file):
    """Extract text from uploaded image using pytesseract."""
    try:
        image = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text
    except Exception as e:
        st.error(f"Error processing the image file: {e}")
        return None

def analyze_health_report(extracted_text, gemini_model):
    """Analyze health report using Gemini model."""
    prompt = f"""
    Analyze the following health report image and provide:
    1. Key findings in the report.
    2. Explanation of findings in report.
    3. Recommendations for lifestyle changes or immediate actions.
    4. Suggested doctor specialty to consult.

    Extracted Text from the Report:
    {extracted_text}
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error while analyzing the report: {e}")
        return None

# Initialize LLM and Gemini API
try:
    llm = Llama.from_pretrained(
        repo_id="tensorblock/Llama3-Aloe-8B-Alpha-GGUF",
        filename="Llama3-Aloe-8B-Alpha-Q2_K.gguf",
        n_ctx=2048
    )
    genai.configure(api_key="AIzaSyDE964W1AZPSnBINofLEG2gqnrpnFcABro")
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    llm, gemini_model = None, None
    st.error(f"Model initialization failed: {e}")

# Tabs Configuration
tab1, tab2 = st.tabs(["Healthcare Chatbot", "Report Analyzer"])

with tab1:
    # Title and Introduction
    st.title("💬 Healthcare Chatbot")
    st.markdown("An AI-powered chatbot to assist with healthcare queries and generate insights.")

    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "category" not in st.session_state:
        st.session_state.category = None
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "current_question_idx" not in st.session_state:
        st.session_state.current_question_idx = 0
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "fishbone_fig" not in st.session_state:
        st.session_state.fishbone_fig = None

    # Function to Handle User Input
    def handle_user_input(user_input: str):
        """Processes user input and updates the chatbot flow."""
        if st.session_state.category is None:
            # Step 1: Classify Medical Intent
            st.session_state.category = classify_medical_intent(user_input)
            st.session_state.messages.append({"role": "assistant", "content": f"Detected category: {st.session_state.category}"})

            # Step 2: Generate Initial Questions
            st.session_state.questions = generate_interview_questions(user_input, st.session_state.category, gemini_model)
            if st.session_state.questions:
                first_question = st.session_state.questions[0]
                st.session_state.messages.append({"role": "assistant", "content": first_question})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "I couldn't generate questions. Please try again."})

        elif st.session_state.current_question_idx < len(st.session_state.questions):
            # Step 3: Save Response and Ask the Next Question
            current_question = st.session_state.questions[st.session_state.current_question_idx]
            st.session_state.responses[current_question] = user_input
            st.session_state.current_question_idx += 1

            if st.session_state.current_question_idx < len(st.session_state.questions):
                next_question = st.session_state.questions[st.session_state.current_question_idx]
                st.session_state.messages.append({"role": "assistant", "content": next_question})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Thank you for your responses. Generating analysis..."})

                # Step 4: Generate Comprehensive Analysis
                interview_data = {
                    "intent": st.session_state.category,
                    "initial_symptoms": st.session_state.messages[0]["content"],
                    "detailed_responses": st.session_state.responses,
                }
                st.session_state.analysis = generate_comprehensive_analysis(
                    interview_data, st.session_state.category, gemini_model, llm
                )

                # Step 5: Generate Fishbone Diagram
                st.session_state.fishbone_fig = plot_enhanced_fishbone(
                    st.session_state.category, st.session_state.analysis["root_cause_data1"]
                )

    # Sidebar Information
    with st.sidebar:
        st.markdown("## Healthcare Chatbot Guide")
        st.markdown("""
        🩺 **How to Use:**
        - Start by describing your medical concern
        - Answer follow-up questions carefully
        - Get a comprehensive health analysis
        """)

        # Spacer to push input to bottom
        for _ in range(18):
            st.markdown("")

        # New Conversation Button (Conditionally displayed)
        if st.session_state.get('analysis'):
            if st.button("Start New Conversation", key="new_conv_btn"):
                # Reset all session states
                keys_to_reset = [
                    'messages', 'category', 'questions', 
                    'current_question_idx', 'responses', 
                    'analysis', 'fishbone_fig'
                ]
                for key in keys_to_reset:
                    st.session_state[key] = None if key != 'messages' else []
                
                # Trigger a rerun
                st.rerun()

        # Input Section - At the very bottom
        # Use a unique key for input to ensure it resets
        user_input = st.text_input("💬 Describe your health concern:", 
                                key=f"main_input_{st.session_state.get('input_reset_counter', 0)}", 
                                label_visibility="visible")
        
        # # Send Button
        # send_btn = st.button("Send", key="main_send")

        if st.button("Send", key="main_send") and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            handle_user_input(user_input)
        
        # # Handle input and button click
        # if send_btn and user_input:
        #     # Increment input reset counter to force input reset
        #     st.session_state['input_reset_counter'] = st.session_state.get('input_reset_counter', 0) + 1
            
        #     # Add user message
        #     st.session_state.messages.append({"role": "user", "content": user_input})
        #     handle_user_input(user_input)
    # # Function to Handle User Input
    # def handle_user_input(user_input: str):
    #     """Processes user input and updates the chatbot flow."""
    #     if st.session_state.category is None:
    #         # Step 1: Classify Medical Intent
    #         st.session_state.category = classify_medical_intent(user_input)
    #         st.session_state.messages.append({"role": "assistant", "content": f"Detected category: {st.session_state.category}"})

    #         # Step 2: Generate Initial Questions
    #         st.session_state.questions = generate_interview_questions(user_input, st.session_state.category, gemini_model)
    #         if st.session_state.questions:
    #             first_question = st.session_state.questions[0]
    #             st.session_state.messages.append({"role": "assistant", "content": first_question})
    #         else:
    #             st.session_state.messages.append({"role": "assistant", "content": "I couldn't generate questions. Please try again."})

    #     elif st.session_state.current_question_idx < len(st.session_state.questions):
    #         # Step 3: Save Response and Ask the Next Question
    #         current_question = st.session_state.questions[st.session_state.current_question_idx]
    #         st.session_state.responses[current_question] = user_input
    #         st.session_state.current_question_idx += 1

    #         if st.session_state.current_question_idx < len(st.session_state.questions):
    #             next_question = st.session_state.questions[st.session_state.current_question_idx]
    #             st.session_state.messages.append({"role": "assistant", "content": next_question})
    #         else:
    #             st.session_state.messages.append({"role": "assistant", "content": "Thank you for your responses. Generating analysis..."})

    #             # Step 4: Generate Comprehensive Analysis
    #             interview_data = {
    #                 "intent": st.session_state.category,
    #                 "initial_symptoms": st.session_state.messages[0]["content"],
    #                 "detailed_responses": st.session_state.responses,
    #             }
    #             st.session_state.analysis = generate_comprehensive_analysis(
    #                 interview_data, st.session_state.category, gemini_model, llm
    #             )

    #             # Step 5: Generate Fishbone Diagram
    #             st.session_state.fishbone_fig = plot_enhanced_fishbone(
    #                 st.session_state.category, st.session_state.analysis["root_cause_data1"]
    #             )

    # # Display Chat
    # for msg in st.session_state.messages:
    #     if msg["role"] == "user":
    #         message(msg["content"], is_user=True, key=msg["content"] + "_user")
    #     else:
    #         message(msg["content"], key=msg["content"] + "_assistant")

    # Display Chat
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{idx}")
        else:
            message(msg["content"], key=f"assistant_{idx}")


    # Display Comprehensive Analysis
    if st.session_state.analysis:
        st.markdown("### Comprehensive Medical Analysis")
        formatted_analysis = format_analysis_output(st.session_state.analysis)
        st.markdown(formatted_analysis, unsafe_allow_html=True)

    # Display Fishbone Diagram
    if st.session_state.fishbone_fig:
        st.markdown("### Root Cause Analysis (Fishbone Diagram)")
        st.pyplot(st.session_state.fishbone_fig)

with tab2:
    st.title("🩺 Health Report Analyzer")
    st.markdown("Upload your health report for AI-powered analysis.")

    # File Uploader
    uploaded_file = st.file_uploader("Choose a PNG file", type="png")
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Health Report", use_column_width=True)
        
        # Analyze Report Button
        if st.button("Analyze Report"):
            # Convert and extract
            base64_image = convert_image_to_base64(uploaded_file)
            extracted_text = extract_text_from_image(uploaded_file)
            
            # Analyze with Gemini
            if extracted_text:
                analysis_result = analyze_health_report(extracted_text, gemini_model)
                
                # Display Analysis
                st.markdown("### AI Report Analysis")
                st.write(analysis_result)