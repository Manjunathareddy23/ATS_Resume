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
    try:
        pdf_reader = PyPDF2.PdfReader(file)  # New version of PyPDF2 (2.x.x)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to extract skills using Gemini API
def extract_skills_from_api(text):
    try:
        # Correctly call the generate function with the appropriate parameters
        response = genai.generate(
            model="gemini-1.5-pro",  # Ensure this model exists and is valid
            prompt=text,  # Provide the input text (resume or job description)
            max_tokens=500  # Adjust this based on your requirements
        )
        
        # Assuming the model returns a text response, we split the response into lines (skills)
        return response['text'].split("\n")
    except Exception as e:
        st.error(f"API Request Error: {e}")
        return []

# Function to match skills using Gemini API results
def match_skills(resume_text, job_description):
    resume_skills = extract_skills_from_api(resume_text)
    job_skills = extract_skills_from_api(job_description)
    
    # Calculate matched and missing skills
    matched_skills = set(resume_skills).intersection(job_skills)
    missing_skills = set(job_skills) - set(resume_skills)

    return matched_skills, missing_skills

# Call Gemini API for HR/Placement Questions
def generate_placement_questions(job_description):
    try:
        # Correctly call the generate function with the appropriate parameters for questions
        response = genai.generate(
            model="gemini-1.5-pro",  # Ensure this model exists and is valid
            prompt=job_description,  # Provide the input job description
            max_tokens=500  # Adjust based on your requirements
        )
        
        # Assuming the model returns a list of questions as text (separated by newlines)
        return response['text'].split("\n")
    except Exception as e:
        st.error(f"API Request Error: {e}")
        return []

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
    if questions:
        for q in questions:
            st.write(f"- **{q}**")
    else:
        st.write("No questions generated or API response is missing.")

elif not resume_file:
    st.warning("❌ Please upload a resume file.")
elif not job_description:
    st.warning("❌ Please enter a job description.")
