
import streamlit as st
import pdfplumber
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load SpaCy NLP model and stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# Load BERT model pipeline
similarity_model = pipeline("feature-extraction", model="bert-base-uncased")

# Expanded skills database
skills_database = ['python', 'java', 'machine learning', 'data science', 'deep learning', 'cloud computing', 
                   'sql', 'project management', 'data analysis', 'communication', 'react', 'docker', 
                   'tensorflow', 'pandas', 'kubernetes', 'aws', 'azure', 'nlp', 'statistics']

# Streamlit app configuration
st.title("AI-Powered Resume Reviewer (Advanced NLP with Skills Extraction)")

# Input for the job description
st.subheader("Enter the Job Description")
job_description = st.text_area("Paste the job description here", height=200)

# Upload the resume in PDF format
st.subheader("Upload Your Resume (PDF)")
uploaded_file = st.file_uploader("Choose a file", type="pdf")

# Function to extract text from uploaded PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle potential None return
    return text

# Clean text (remove stopwords and apply lemmatization)
def clean_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    return " ".join(tokens)

# Match using TF-IDF
def tfidf_match(resume_text, job_description_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description_text])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# Corrected BERT-based similarity function
def bert_match(resume_text, job_description_text):
    resume_tokens = similarity_model.tokenizer(resume_text, truncation=True, max_length=512, return_tensors="pt")
    job_tokens = similarity_model.tokenizer(job_description_text, truncation=True, max_length=512, return_tensors="pt")
    resume_embeddings = similarity_model.model(**resume_tokens).last_hidden_state.mean(dim=1).detach().numpy()
    job_embeddings = similarity_model.model(**job_tokens).last_hidden_state.mean(dim=1).detach().numpy()
    return cosine_similarity(resume_embeddings, job_embeddings)[0][0]

# Skill extraction from resume
def extract_skills(resume_text, skills_database):
    found_skills = [skill for skill in skills_database if skill.lower() in resume_text.lower()]
    return found_skills

# Calculate scores
def calculate_scores(resume_text, job_description, required_experience, candidate_experience):
    # Keyword Match Score
    job_keywords = job_description.split()
    cleaned_resume = clean_text(resume_text)
    matching_keywords = [kw for kw in job_keywords if kw.lower() in cleaned_resume.lower()]
    keyword_match_score = (len(matching_keywords) / len(job_keywords)) * 100 if job_keywords else 0

    # Skill Match Score
    found_skills = extract_skills(resume_text, skills_database)
    skill_match_score = (len(found_skills) / len(skills_database)) * 100

    # Experience Score
    experience_score = min((candidate_experience / required_experience) * 100, 100)

    # Contextual Match Score (average of TF-IDF and BERT scores)
    tfidf_score = tfidf_match(cleaned_resume, job_description)
    bert_score = bert_match(cleaned_resume, job_description)
    contextual_match_score = (tfidf_score + bert_score) / 2 * 100

    return keyword_match_score, skill_match_score, experience_score, contextual_match_score

# Provide contextual suggestions (e.g., missing skills or keywords)
def suggest_improvements(missing_skills):
    return f"Consider adding the following skills to match the job description better: {', '.join(missing_skills)}."

# Analyze and compare resume with job description
if uploaded_file is not None and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # Define required experience (example: 3 years)
    required_experience = 3  # Set this based on the job description or user input
    candidate_experience = 5  # This should be extracted from the resume

    # Calculate scores
    keyword_match_score, skill_match_score, experience_score, contextual_match_score = calculate_scores(resume_text, job_description, required_experience, candidate_experience)

    # Calculate the ATS score
    ats_score = (keyword_match_score * 0.3) + (skill_match_score * 0.3) + (experience_score * 0.2) + (contextual_match_score * 0.2)

    # Real-time feedback display
    st.subheader("Resume Review Results")
    st.write(f"*Keyword Match Score*: {round(keyword_match_score, 2)}%")
    st.write(f"*Skill Match Score*: {round(skill_match_score, 2)}%")
    st.write(f"*Experience Score*: {round(experience_score, 2)}%")
    st.write(f"*Contextual Match Score*: {round(contextual_match_score, 2)}%")
    st.success(f"*ATS Score*: {round(ats_score, 2)}%")

    # Provide skill-related feedback
    st.subheader("Skill Analysis")
    st.write(f"Skills found in your resume: {', '.join(extract_skills(resume_text, skills_database))}")
    
    missing_skills = [skill for skill in skills_database if skill.lower() not in [s.lower() for s in extract_skills(resume_text, skills_database)]]
    if missing_skills:
        st.warning(suggest_improvements(missing_skills))
    else:
        st.success("Your resume includes all the required skills from the job description!")

    # Show extracted text for user validation (optional)
    st.subheader("Extracted Resume Text")
    st.write(resume_text)

else:
    st.info("Please upload a resume and provide a job description for analysis.")
