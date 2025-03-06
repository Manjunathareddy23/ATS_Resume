import streamlit as st
import PyPDF2
import os
from dotenv import load_dotenv
import google.generativeai as genai  # Gemini API client

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key is loaded correctly
if GEMINI_API_KEY is None:
    st.error("⚠️ Gemini API Key is missing! Set it as an environment variable.")
    st.stop()
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Streamlit Page Configurations
st.set_page_config(page_title="Resume & Job Fit Analyzer", layout="centered")

# CSS Styling
st.markdown("""
    <style>
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .button {
            font-size: 18px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

# Helper Functions

# Extract Text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to generate HR/Placement Questions and Answers using Gemini API
def generate_placement_questions_and_answers(resume_text, job_description):
    try:
        # Combine the resume text and job description for better context
        combined_text = f"Resume: {resume_text}\nJob Description: {job_description}"
        
        # Generating HR/Placement questions and answers using the Gemini API
        response = genai.Completion.create(
            model="gemini-1.5-pro",  # Ensure this model is available
            prompt=combined_text,  # The combined input text for better results
            max_tokens=500  # Adjust the token limit as required
        )
        
        # Return the generated response text
        return response['text']
    except Exception as e:
        st.error(f"API Request Error: {e}")
        return "Failed to generate HR/Placement questions and answers."

# Streamlit Layout
st.title("Resume & Job Fit Analyzer")
st.markdown("Upload your resume and paste the job description to see how well they match.")

# Resume Upload
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Job Description Input
job_description = st.text_area("Paste Job Description", height=200)

# When both resume and job description are provided
if resume_file and job_description:
    # Extract resume text
    resume_text = extract_text_from_pdf(resume_file)

    # Match and generate HR/Placement questions and answers
    generated_content = generate_placement_questions_and_answers(resume_text, job_description)
    
    # Display Results
    st.subheader("Generated HR/Placement Questions and Answers")
    st.write(generated_content)

elif not resume_file:
    st.warning("❌ Please upload a resume file.")
elif not job_description:
    st.warning("❌ Please enter a job description.")

