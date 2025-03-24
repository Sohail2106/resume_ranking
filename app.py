import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to extract text content from PDF resumes
def extract_text_from_resume(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_content = ""
    for page in reader.pages:
        page_content = page.extract_text()
        if page_content:
            extracted_content += page_content
    return extracted_content


# Function to compare resumes with job description and assign scores
def compute_resume_scores(job_posting, resume_data):
    text_corpus = [job_posting] + resume_data  
    tfidf_vectorizer = TfidfVectorizer()
    transformed_text = tfidf_vectorizer.fit_transform(text_corpus).toarray()

    job_post_vector = transformed_text[0]  
    resume_vectors = transformed_text[1:]  

    similarity_scores = cosine_similarity([job_post_vector], resume_vectors).flatten()

    if similarity_scores.size > 0:
        highest_score = max(similarity_scores)
        normalized_scores = [(score / highest_score) * 10 for score in similarity_scores]
        final_scores = [round(score, 1) for score in normalized_scores]
    else:
        final_scores = [0] * len(resume_data)  

    return final_scores


# Streamlit User Interface
st.title("AI-Based Resume Screening & Ranking System")

# Section for job description input
st.header("Enter Job Description")
job_posting = st.text_area("Provide the job details here:")

# Resume file upload section
st.header("Upload Candidate Resumes")
uploaded_resume_files = st.file_uploader("Upload resumes in PDF format", type=["pdf"], accept_multiple_files=True)

# Processing resumes and displaying ranking
if uploaded_resume_files and job_posting:
    st.header("Resume Ranking Results")

    extracted_resume_texts = []
    for uploaded_resume in uploaded_resume_files:
        extracted_resume_texts.append(extract_text_from_resume(uploaded_resume))

    # Calculate similarity scores
    ranking_results = compute_resume_scores(job_posting, extracted_resume_texts)

    # Create a DataFrame for ranking output
    results_df = pd.DataFrame({
        "Candidate Resume": [file.name for file in uploaded_resume_files],
        "Match Score (Out of 10)": ranking_results
    }).sort_values(by="Match Score (Out of 10)", ascending=False)

    # Display ranking results
    st.write(results_df)
