from dotenv import load_dotenv
import streamlit as st
import os
import fitz
from google import genai

# ---------------- LOAD ENV ---------------- #

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found!")
    st.stop()

# ---------------- GEMINI CLIENT ---------------- #

client = genai.Client(api_key=API_KEY)

# ---------------- GEMINI RESPONSE ---------------- #

def get_gemini_response(user_prompt, pdf_content, job_description):

    try:

        final_prompt = f"""
        {user_prompt}

        JOB DESCRIPTION:
        {job_description}

        RESUME CONTENT:
        {pdf_content}
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=final_prompt
        )

        return response.text

    except Exception as e:
        return f"❌ Error: {e}"

# ---------------- PDF TEXT EXTRACTION ---------------- #

def input_pdf_setup(uploaded_file):

    if uploaded_file is not None:

        document = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        text_parts = []

        for page in document:
            text_parts.append(page.get_text())

        text = " ".join(text_parts)

        # Prevent token overflow
        return text[:7000]

    else:
        return ""

# ---------------- STREAMLIT CONFIG ---------------- #

st.set_page_config(
    page_title="ATS Resume Expert",
    page_icon="📄",
    layout="centered"
)

# ---------------- UI ---------------- #

st.title("📄 ATS Resume Expert")

st.subheader(
    "Paste Job Description & Upload Resume"
)

# Job Description
input_text = st.text_area(
    "📋 Job Description"
)

# Resume Upload
uploaded_file = st.file_uploader(
    "📂 Upload Resume (PDF)",
    type=["pdf"]
)

# ---------------- PROCESS PDF ---------------- #

pdf_content = ""

if uploaded_file is not None:

    st.success("✅ PDF Uploaded Successfully")

    pdf_content = input_pdf_setup(uploaded_file)

# ---------------- BUTTONS ---------------- #

if uploaded_file and input_text:

    # ATS Score
    if st.button("📊 Get ATS Score"):

        response = get_gemini_response(
            "Provide ATS match percentage only.",
            pdf_content,
            input_text
        )

        st.subheader("ATS Score")

        st.write(response)

    # Low Score Reason
    if st.button("📉 Why is my score low?"):

        response = get_gemini_response(
            "Explain why the ATS score is low.",
            pdf_content,
            input_text
        )

        st.subheader("Reasons for Low Score")

        st.write(response)

    # Matched Skills
    if st.button("✅ Matched Skills"):

        response = get_gemini_response(
            "List matched skills between resume and job description.",
            pdf_content,
            input_text
        )

        st.subheader("Matched Skills")

        st.write(response)

    # Missing Skills
    if st.button("⚠️ Missing Skills"):

        response = get_gemini_response(
            "List missing skills in the resume compared to the job description.",
            pdf_content,
            input_text
        )

        st.subheader("Missing Skills")

        st.write(response)

    # HR Questions
    if st.button("🎤 HR Interview Questions"):

        response = get_gemini_response(
            "Generate HR interview questions based on resume and job description.",
            pdf_content,
            input_text
        )

        st.subheader("HR Interview Questions")

        st.write(response)

else:
    st.info("📌 Upload resume and enter job description.")

# ---------------- LIKE / DISLIKE ---------------- #

if "like_count" not in st.session_state:
    st.session_state.like_count = 0

if "dislike_count" not in st.session_state:
    st.session_state.dislike_count = 0

col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Like"):
        st.session_state.like_count += 1

    st.write(f"Likes: {st.session_state.like_count}")

with col2:
    if st.button("👎 Dislike"):
        st.session_state.dislike_count += 1

    st.write(f"Dislikes: {st.session_state.dislike_count}")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.markdown(
    "#### Developed By Manjunathareddy"
)

st.markdown(
    "*Let's Connect - 6300138360*"
)
