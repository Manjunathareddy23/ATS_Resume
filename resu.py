from dotenv import load_dotenv
import streamlit as st
import os
import fitz
from groq import Groq

# ---------------- LOAD ENV ---------------- #

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found!")
    st.stop()

# ---------------- GROQ CLIENT ---------------- #

client = Groq(api_key=GROQ_API_KEY)

# ---------------- PDF TEXT EXTRACTION ---------------- #

def input_pdf_setup(uploaded_file):

    if uploaded_file is not None:

        pdf_bytes = uploaded_file.read()

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        text_parts = []

        for page in document:
            text_parts.append(page.get_text())

        text = " ".join(text_parts)

        return text[:8000]

    return ""

# ---------------- AI RESPONSE ---------------- #

def get_ai_response(user_prompt, pdf_content, job_description):

    try:

        final_prompt = f"""
        You are an advanced ATS Resume Analyzer.

        Analyze the resume against the job description carefully.

        Your analysis should be:
        - realistic
        - strict
        - detailed
        - professional

        USER REQUEST:
        {user_prompt}

        JOB DESCRIPTION:
        {job_description}

        RESUME:
        {pdf_content}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional ATS resume screening system."
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=0.3,
            max_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {e}"

# ---------------- STREAMLIT CONFIG ---------------- #

st.set_page_config(
    page_title="ATS Resume Expert",
    page_icon="📄",
    layout="wide"
)

# ---------------- HEADER ---------------- #

st.title("📄 ATS Resume Expert (Manju Reddy)")

st.markdown(
    """
    Analyze your resume against a Job Description using AI-powered ATS analysis.
    """
)

# ---------------- INPUTS ---------------- #

job_description = st.text_area(
    "📋 Paste Job Description",
    height=250
)

uploaded_file = st.file_uploader(
    "📂 Upload Resume (PDF)",
    type=["pdf"]
)

# ---------------- PROCESS PDF ---------------- #

resume_text = ""

if uploaded_file is not None:

    st.success("✅ Resume Uploaded Successfully")

    resume_text = input_pdf_setup(uploaded_file)

# ---------------- BUTTONS ---------------- #

if uploaded_file and job_description:

    col1, col2 = st.columns(2)

    with col1:

        if st.button("📊 ATS Score"):

            response = get_ai_response(
                """
                Give:
                1. ATS Match Percentage
                2. Overall evaluation
                3. Final hiring probability

                Format professionally.
                """,
                resume_text,
                job_description
            )

            st.subheader("📊 ATS Analysis")

            st.write(response)

        if st.button("✅ Matched Skills"):

            response = get_ai_response(
                """
                List all matched:
                - technical skills
                - tools
                - soft skills
                - experience alignment
                """,
                resume_text,
                job_description
            )

            st.subheader("✅ Matched Skills")

            st.write(response)

        if st.button("🎤 Interview Questions"):

            response = get_ai_response(
                """
                Generate:
                - HR questions
                - technical questions
                - project-based questions

                based on the resume and JD.
                """,
                resume_text,
                job_description
            )

            st.subheader("🎤 Interview Questions")

            st.write(response)

    with col2:

        if st.button("⚠️ Missing Skills"):

            response = get_ai_response(
                """
                Identify:
                - missing skills
                - missing tools
                - weak experience areas
                - missing keywords
                - missing projects

                Also suggest improvements.
                """,
                resume_text,
                job_description
            )

            st.subheader("⚠️ Missing Skills")

            st.write(response)

        if st.button("📉 Why ATS Score is Low?"):

            response = get_ai_response(
                """
                Explain in detail:
                - why ATS score is low
                - what hurts the resume
                - formatting issues
                - keyword issues
                - project issues
                - experience gaps

                Give actionable improvements.
                """,
                resume_text,
                job_description
            )

            st.subheader("📉 ATS Weakness Analysis")

            st.write(response)

        if st.button("🚀 Resume Improvement Tips"):

            response = get_ai_response(
                """
                Improve this resume for the given job description.

                Suggest:
                - better resume summary
                - better projects
                - better keywords
                - better technical skills
                - ATS optimization tips
                """,
                resume_text,
                job_description
            )

            st.subheader("🚀 Resume Improvement Suggestions")

            st.write(response)

else:
    st.info("📌 Upload resume and paste job description.")

# ---------------- FEEDBACK ---------------- #

st.markdown("---")

if "likes" not in st.session_state:
    st.session_state.likes = 0

if "dislikes" not in st.session_state:
    st.session_state.dislikes = 0

col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Like"):
        st.session_state.likes += 1

    st.write(f"👍 Likes: {st.session_state.likes}")

with col2:
    if st.button("👎 Dislike"):
        st.session_state.dislikes += 1

    st.write(f"👎 Dislikes: {st.session_state.dislikes}")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.markdown("### Developed By Manjunathareddy")

st.markdown("📞 Contact: 6300138360")
