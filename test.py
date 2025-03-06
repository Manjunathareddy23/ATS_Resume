import streamlit as st
import PyPDF2
import requests
import re
from io import StringIO

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

# Function to match skills
def match_skills(resume_text, job_description):
    # Basic skill matching logic (you can improve it)
    resume_skills = set(re.findall(r'\b\w+\b', resume_text.lower()))
    job_skills = set(re.findall(r'\b\w+\b', job_description.lower()))

    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - matched_skills

    return matched_skills, missing_skills

# Call Gemini API for HR/Placement Questions
def generate_placement_questions(job_description):
    # Here you should make an API call to Gemini or OpenAI
    # Example with an OpenAI-like request for illustration
    response = requests.post(
        "https://api.gemini.ai/your-api-endpoint", 
        headers={"Authorization": "Bearer YOUR_API_KEY"},
        json={"text": job_description}
    )
    return response.json()

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

    # Match Skills
    matched_skills, missing_skills = match_skills(resume_text, job_description)
    
    # Calculate Job Fit Percentage (based on matched skills)
    total_skills = len(set(re.findall(r'\b\w+\b', job_description.lower())))
    matched_percentage = (len(matched_skills) / total_skills) * 100 if total_skills > 0 else 0

    # Display Results
    st.subheader(f"Job Fit Percentage: {matched_percentage:.2f}%")

    st.subheader("Matched Skills")
    st.write(", ".join(matched_skills) if matched_skills else "No skills matched.")

    st.subheader("Missing Skills")
    st.write(", ".join(missing_skills) if missing_skills else "No missing skills.")

    # HR/Placement Question Generation
    st.subheader("HR/Placement Questions")
    questions = generate_placement_questions(job_description)
    if 'questions' in questions:
        for q in questions['questions']:
            st.write(f"- {q['question']}: {q['answer']}")
    else:
        st.write("No questions generated.")

elif not resume_file:
    st.warning("Please upload a resume file.")
elif not job_description:
    st.warning("Please enter a job description.")

