import streamlit as st
import fitz  # PyMuPDF for extracting text from PDFs
import google.generativeai as genai
import os  # For environment variables
from dotenv import load_dotenv  # To load .env file

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key exists
if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API Key is missing! Set it as an environment variable.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Function to extract text 
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to generate HR/Placement questions using Gemini API
def generate_hr_questions(resume_text, job_description, num_questions):
    if not resume_text or not job_description:
        return "❌ Please provide both resume text and job description."
    
    # Combine the resume text and job description for context
    combined_text = f"Resume: {resume_text}\n\nJob Description: {job_description}\n\nPlease generate {num_questions} HR/Placement questions based on the above content."
    
    try:
        # Call the Gemini API to generate questions based on the combined text
        response = genai.generate_text(
            model="gemini-1.5-pro",  # Ensure this model is available
            prompt=combined_text,  # The combined input text for better results
            max_tokens=500  # Adjust the token limit as required
        )
        
        # Return the generated response text
        return response.get("text", "No text returned from Gemini API.")
    
    except Exception as e:
        return f"❌ Error generating questions: {e}"

# Streamlit UI
st.title("📘 AI-Based HR/Placement Question Generator")
st.write("Upload your resume and paste the job description to generate HR/Placement questions.")

# File uploader for resume
resume_file = st.file_uploader("📂 Upload your Resume (PDF)", type=["pdf"])

# Text input for job description
job_description = st.text_area("Paste Job Description", height=200)

# Number input for questions
num_questions = st.number_input("🔢 Number of HR Questions", min_value=1, value=5)

# Generate button
if st.button("🎯 Generate HR Questions"):
    if resume_file and job_description:
        # Extract resume text
        resume_text = extract_text_from_pdf(resume_file)
        
        if resume_text:
            # Generate HR/Placement questions
            with st.spinner("⏳ Generating HR/Placement Questions... Please wait!"):
                hr_questions = generate_hr_questions(resume_text, job_description, num_questions)
            st.subheader("📜 Generated HR/Placement Questions:")
            st.write(hr_questions)
        else:
            st.error("❌ Unable to extract text from the uploaded resume.")
    else:
        st.error("❌ Please upload a resume and paste a job description.")
