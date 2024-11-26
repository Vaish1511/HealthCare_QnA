import base64
from PIL import Image
import pytesseract
import google.generativeai as genai

# Step 1: Configure the Gemini API
GEMINI_API_KEY = "AIzaSyDE964W1AZPSnBINofLEG2gqnrpnFcABro"  
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Step 2: Function to convert the image to base64
def convert_image_to_base64(file_path):
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the path and try again.")
        return None

# Step 3: Function to extract text from the image using OCR (pytesseract)
def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(image)
        print("\nExtracted Text from the Image:")
        print(extracted_text)
        return extracted_text
    except Exception as e:
        print(f"Error processing the image file: {e}")
        return None

# Step 4: Function to analyze the health report using the Gemini model
def analyze_health_report(encoded_image, extracted_text):
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

        # Display the AI analysis result
        if response:
            response_text = response.text.strip()
            print("\n### AI Analysis and Recommendations ###")
            print(response_text)
        else:
            print("No response received from the AI model.")
    except Exception as e:
        print(f"Error while analyzing the report: {e}")

# Main Execution Flow
def main():
    import argparse

    # Argument parser for input file
    parser = argparse.ArgumentParser(description="Health Report Analyzer")
    parser.add_argument("image_file", type=str, help="Path to the health/laboratory report image")
    args = parser.parse_args()

    # Convert the image to base64
    encoded_image = convert_image_to_base64(args.image_file)
    if not encoded_image:
        return

    # Extract text from the image using OCR
    extracted_text = extract_text_from_image(args.image_file)
    if not extracted_text:
        return

    # Analyze the health report
    analyze_health_report(encoded_image, extracted_text)

if __name__ == "__main__":
    # Ensure pytesseract is available
    try:
        pytesseract.get_tesseract_version()
    except EnvironmentError:
        print("pytesseract is not installed or configured. Install it using `sudo apt install tesseract-ocr` or equivalent.")
        exit(1)

    main()
