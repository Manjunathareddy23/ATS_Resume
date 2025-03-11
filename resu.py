from dotenv import load_dotenv
import streamlit as st
import os
import fitz
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to generate response from Gemini API
def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input, pdf_content, prompt])
    return response.text

# Function to extract text from PDF
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text_parts = [page.get_text() for page in document]
        return " ".join(text_parts)
    else:
        raise FileNotFoundError("No file uploaded")

# Streamlit App Configuration
st.set_page_config(page_title="ATS Resume Expert")

# Page Header
st.header("ATS Tracking System")
st.subheader("Paste Your Job Description & Upload Your Resume")

# Input Fields
input_text = st.text_area("Job Description: ")
uploaded_file = st.file_uploader("Upload your Resume (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.write("✅ PDF Uploaded Successfully")
    pdf_content = input_pdf_setup(uploaded_file)
    
    # Buttons to get ATS score and insights
    if st.button("Get ATS Score"):
        response = get_gemini_response("Provide an exact ATS match percentage (only the percentage).", pdf_content, input_text)
        try:
            score = float(response.strip().replace("%", ""))
            st.subheader("📊 ATS Score")
            st.write(f"**{score:.2f}%**")
        except ValueError:
            st.write("❌ Error: Unable to retrieve an exact percentage.")
    
    if st.button("Why is my score low?"):
        response = get_gemini_response("Explain why the ATS match percentage is low.", pdf_content, input_text)
        st.subheader("📉 Reasons for Low Score")
        st.write(response)
    
    if st.button("Matched Skills"):
        response = get_gemini_response("List the skills from the resume that match the job description.", pdf_content, input_text)
        st.subheader("✅ Matched Skills")
        st.write(response)
    
    if st.button("Missing Skills"):
        response = get_gemini_response("List the skills missing in the resume compared to the job description.", pdf_content, input_text)
        st.subheader("⚠️ Missing Skills")
        st.write(response)
    
    if st.button("HR Questions"):
        response = get_gemini_response("Generate interview questions based on the resume and job description.", pdf_content, input_text)
        st.subheader("🎤 HR Interview Questions")
        st.write(response)

# Like & Dislike Counter
if "like_count" not in st.session_state:
    st.session_state.like_count = 0
if "dislike_count" not in st.session_state:
    st.session_state.dislike_count = 0

col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Like"):
        st.session_state.like_count += 1
st.write(f"**Likes: {st.session_state.like_count}**")

with col2:
    if st.button("👎 Dislike"):
        st.session_state.dislike_count += 1
st.write(f"**Dislikes: {st.session_state.dislike_count}**")

# Footer
footer = """
---
#### Developed By [Manjunathareddy]  
*Let's Connect - 6300138360*
"""
st.markdown(footer, unsafe_allow_html=True)
