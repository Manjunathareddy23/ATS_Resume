import streamlit as st
import PyPDF2
import requests
import spacy
import os
from io import StringIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key is loaded correctly
if GEMINI_API_KEY is None:
    st.error("API key is not set. Please ensure the .env file contains the GEMINI_API_KEY.")
    st.stop()

# Load spaCy model for NLP-based processing (optional for other uses)
nlp = spacy.load('en_core_web_sm')

# Streamlit Page Configurations
st.set_page_config(page_title="Resume Analyzer", layout="centered")

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
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page_num in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text

# Function to extract skills using Gemini API
def extract_skills_from_api(text):
    try:
        response = requests.post(
            "https://api.gemini.ai/your-api-endpoint",  # Replace with Gemini API endpoint
            headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},  # Using the API key from environment variable
            json={"text": text}
        )
        response.raise_for_status()
        return response.json().get("skills", [])  # Assuming the response contains skills in a "skills" field
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return []

# Function to match skills using Gemini API results
def match_skills(resume_text, job_description):
    # Extract skills from both resume and job description using the Gemini API
    resume_skills = extract_skills_from_api(resume_text)
    job_skills = extract_skills_from_api(job_description)

    # Calculate matched and missing skills
    matched_skills = set(resume_skills).intersection(job_skills)
    missing_skills = set(job_skills) - set(resume_skills)

    return matched_skills, missing_skills

# Call Gemini API for HR/Placement Questions
def generate_placement_questions(job_description):
    try:
        response = requests.post(
            "https://api.gemini.ai/your-api-endpoint",  # Replace with Gemini API endpoint
            headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},  # Using the API key from environment variable
            json={"text": job_description}
        )
        response.raise_for_status()  # Ensure we handle HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return {}

# Streamlit Layout

st.title("Resume Job Fit Analyzer")
st.markdown("Upload your resume and input a job description to see how well you match.")

# Resume Upload
resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Job Description Input
job_description = st.text_area("Paste Job Description", height=200)

if resume_file and job_description:
    # Extract resume text
    resume_text = extract_text_from_pdf(resume_file)

    # Match Skills using Gemini API
    matched_skills, missing_skills = match_skills(resume_text, job_description)
    
    # Calculate Job Fit Percentage (based on matched skills)
    total_skills = len(extract_skills_from_api(job_description))
    matched_percentage = (len(matched_skills) / total_skills) * 100 if total_skills > 0 else 0

    # Display Results
    st.subheader(f"Job Fit Percentage: {matched_percentage:.2f}%")

    st.subheader("Matched Skills")
    if matched_skills:
        st.write(", ".join(matched_skills))
    else:
        st.write("No skills matched.")

    st.subheader("Missing Skills")
    if missing_skills:
        st.write(", ".join(missing_skills))
    else:
        st.write("No missing skills.")

    # HR/Placement Question Generation
    st.subheader("HR/Placement Questions")
    questions = generate_placement_questions(job_description)
    if questions and 'questions' in questions:
        for q in questions['questions']:
            st.write(f"- **{q['question']}**: {q['answer']}")
    else:
        st.write("No questions generated or API response is missing.")

elif not resume_file:
    st.warning("Please upload a resume file.")
elif not job_description:
    st.warning("Please enter a job description.")
