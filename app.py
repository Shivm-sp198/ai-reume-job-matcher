
"""
app.py
------
Main Streamlit application for the AI Resume-Job Match & Skill Gap
Analyzer.

Run with:
    python -m streamlit run app.py
"""

import streamlit as st

from resume_parser import extract_text_from_pdf
from text_processor import (
    clean_text_for_tfidf,
    normalize_for_matching,
    basic_stats,
)
from skill_extractor import load_skills_dataset, extract_skills
from matcher import (
    compute_tfidf_similarity,
    get_skill_gap,
    compute_skill_coverage,
    compute_overall_score,
)
from recommendations import generate_recommendations

from semantic_matcher import compute_semantic_similarity
print("semantic search'")

# -----------------------------------------------------------------------
# Page configuration
# Must be the first Streamlit command.
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume-Job Match & Skill Gap Analyzer",
    page_icon="📄",
    layout="wide",
)


# -----------------------------------------------------------------------
# Cache the skills dataset
# -----------------------------------------------------------------------
@st.cache_data
def get_skills_list():
    return load_skills_dataset("data/skills.csv")


# -----------------------------------------------------------------------
# Render skill badges
# -----------------------------------------------------------------------
def render_skill_badges(skills, symbol, color):
    """Render skills as colored badge-like text."""

    if not skills:
        st.write("None")
        return

    badges_html = " ".join(
        f"<span style='background-color:{color};color:white;"
        f"padding:4px 10px;border-radius:12px;margin:3px;"
        f"display:inline-block;font-size:0.85em'>"
        f"{symbol} {skill}</span>"
        for skill in sorted(skills)
    )

    st.markdown(
        badges_html,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------
def main():

    # ===================================================================
    # HEADER
    # ===================================================================

    st.title("📄 AI Resume-Job Match & Skill Gap Analyzer")

    st.caption(
        "Analyze how closely your resume aligns with an internship "
        "or job description."
    )

    st.markdown(
        "This is an **NLP-based resume-job matching and skill-gap "
        "analysis system**. It uses TF-IDF + cosine similarity for "
        "text similarity, and rule-based keyword extraction for skill "
        "detection. It is not a deep learning or generative AI system, "
        "and the score is a similarity indicator, not a hiring prediction."
    )

    st.divider()

    # ===================================================================
    # SECTION 1: INPUT
    # ===================================================================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Your Resume (PDF)")

        uploaded_resume = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
        )

    with col2:
        st.subheader("2. Paste the Job Description")

        job_description = st.text_area(
            "Paste the full job/internship description here",
            height=220,
            placeholder=(
                "e.g., We are looking for a Python developer with "
                "experience in Django, SQL, and REST APIs..."
            ),
        )

    analyze_clicked = st.button(
        "🔍 Analyze Match",
        type="primary",
    )

    st.divider()

    # ===================================================================
    # WAIT UNTIL USER CLICKS ANALYZE
    # ===================================================================

    if not analyze_clicked:
        st.info(
            "Upload a resume and paste a job description, then click "
            "'Analyze Match' to see your results."
        )
        return

    # ===================================================================
    # INPUT VALIDATION
    # ===================================================================

    if uploaded_resume is None:
        st.error(
            "Please upload a resume PDF before analyzing."
        )
        return

    if not job_description or not job_description.strip():
        st.error(
            "Please paste a job description before analyzing."
        )
        return

    # ===================================================================
    # STEP 1: EXTRACT RESUME TEXT
    # ===================================================================

    with st.spinner("Extracting text from your resume..."):

        resume_text_raw, pdf_error = extract_text_from_pdf(
            uploaded_resume
        )

    if pdf_error:
        st.error(pdf_error)
        return

    if not resume_text_raw or not resume_text_raw.strip():
        st.error(
            "No readable text was found in the uploaded PDF. "
            "Please upload a PDF containing selectable text."
        )
        return

    # ===================================================================
    # BASIC TEXT STATISTICS
    # ===================================================================

    resume_stats = basic_stats(resume_text_raw)
    job_stats = basic_stats(job_description)

    if resume_stats["word_count"] < 20:
        st.warning(
            "Your resume text seems very short after extraction "
            f"({resume_stats['word_count']} words). Results may not be "
            "reliable. Please check that the PDF contains selectable text."
        )

    if job_stats["word_count"] < 10:
        st.warning(
            "The job description seems very short "
            f"({job_stats['word_count']} words). Results may not be "
            "reliable."
        )

    # ===================================================================
    # STEP 2: TEXT PREPROCESSING
    # ===================================================================

    resume_clean_tfidf = clean_text_for_tfidf(
        resume_text_raw
    )

    job_clean_tfidf = clean_text_for_tfidf(
        job_description
    )

    resume_for_matching = normalize_for_matching(
        resume_text_raw
    )

    job_for_matching = normalize_for_matching(
        job_description
    )

    # ===================================================================
    # STEP 3: SKILL EXTRACTION
    # ===================================================================

    with st.spinner("Detecting technical skills..."):

        skills_list = get_skills_list()

        resume_skills = extract_skills(
            resume_for_matching,
            skills_list,
        )

        job_skills = extract_skills(
            job_for_matching,
            skills_list,
        )

    if not job_skills:
        st.warning(
            "No known skills were detected in the job description. "
            "Skill-gap analysis will be limited. Text similarity will "
            "still be shown below."
        )

    # ===================================================================
    # STEP 3B: SKILL GAP
    # ===================================================================

    gap = get_skill_gap(
        resume_skills,
        job_skills,
    )

    # ===================================================================
    # STEP 4: TF-IDF + COSINE SIMILARITY
    # ===================================================================

    with st.spinner("Calculating text similarity..."):

        text_similarity, sim_error = compute_tfidf_similarity(
            resume_clean_tfidf,
            job_clean_tfidf,
        )

    if sim_error:
        st.warning(sim_error)

    # ===================================================================
    # STEP 5: SKILL COVERAGE
    # ===================================================================

    skill_coverage = compute_skill_coverage(
        gap["matched"],
        job_skills,
    )

    # ===================================================================
    # STEP 6: OVERALL SCORE
    # ===================================================================

    overall_score = compute_overall_score(
        text_similarity,
        skill_coverage,
    )

    # ===================================================================
    # RESULTS
    # ===================================================================

    st.header("📊 Results")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Overall Match Score",
        f"{overall_score}%",
    )

    m2.metric(
        "Text Similarity (TF-IDF)",
        f"{round(text_similarity * 100, 1)}%",
    )

    m3.metric(
        "Skill Coverage",
        f"{round(skill_coverage * 100, 1)}%",
    )

    # Overall progress
    st.progress(
        min(max(int(overall_score), 0), 100) / 100
    )

    # Match interpretation
    if overall_score >= 70:

        st.success(
            "Strong match with this job description."
        )

    elif overall_score >= 40:

        st.warning(
            "Moderate match. There is room for improvement."
        )

    else:

        st.error(
            "Low match with this job description."
        )

    st.divider()

    # ===================================================================
    # SKILL GAP ANALYSIS
    # ===================================================================

    st.subheader("🧩 Skill Gap Analysis")

    sk1, sk2, sk3 = st.columns(3)

    with sk1:

        st.markdown("**✅ Matched Skills**")

        render_skill_badges(
            gap["matched"],
            "✓",
            "#2e7d32",
        )

    with sk2:

        st.markdown("**⚠️ Missing Skills**")

        render_skill_badges(
            gap["missing"],
            "⚠",
            "#c62828",
        )

    with sk3:

        st.markdown(
            "**➕ Additional Skills "
            "(in resume, not in job desc.)**"
        )

        render_skill_badges(
            gap["additional"],
            "+",
            "#1565c0",
        )

    st.divider()

    # ===================================================================
    # RESUME STRENGTH ANALYSIS
    # ===================================================================

    st.subheader(
        "💪 Resume Strength Analysis (Estimated Indicators)"
    )

    st.caption(
        "These are simple, derived indicators based on text and "
        "keyword analysis — not scientifically validated psychological "
        "or recruiter scores."
    )

    total_job_skills = (
        len(job_skills)
        if job_skills
        else 0
    )

    matched_count = len(
        gap["matched"]
    )

    r1, r2 = st.columns(2)

    # Technical skill coverage
    with r1:

        st.write(
            "Technical Skill Coverage"
        )

        st.progress(
            min(max(skill_coverage, 0.0), 1.0)
        )

        st.caption(
            f"{matched_count} / "
            f"{total_job_skills} "
            "required skills detected"
        )

    # Text similarity
    with r2:

        st.write(
            "Job Relevance (Text Similarity)"
        )

        st.progress(
            min(max(text_similarity, 0.0), 1.0)
        )

        st.caption(
            "Cosine similarity score: "
            f"{round(text_similarity, 3)}"
        )

    st.divider()

    # ===================================================================
    # RECOMMENDATIONS
    # ===================================================================

    st.subheader("💡 Recommendations")

    recs = generate_recommendations(
        gap["missing"],
        overall_score,
        gap["matched"],
    )

    if recs:

        for rec in recs:
            st.markdown(
                f"- {rec}"
            )

    else:

        st.info(
            "No specific recommendations were generated."
        )

    st.divider()

    # ===================================================================
    # METHODOLOGY / TRANSPARENCY
    # ===================================================================

    with st.expander(
        "ℹ️ How is this score calculated? (Methodology)"
    ):

        st.markdown(
            f"""
**Overall Match Score = 60% × Text Similarity + 40% × Skill Coverage**

### 1. Text Similarity

Text similarity is computed using:

- TF-IDF (Term Frequency–Inverse Document Frequency)
- Cosine similarity
- Resume text
- Job-description text

This measures how similar the overall wording and vocabulary
of the two documents are.

### 2. Skill Coverage

Skill coverage is calculated as:

`matched_skills / job_required_skills`

The skills are detected using a reference dataset containing
**{len(skills_list)} skills**.

### 3. Overall Score

The final score combines:

- **60% Text Similarity**
- **40% Skill Coverage**

The weights are configurable constants in `matcher.py`.

**Important:** This score is a similarity/match indicator,
not a prediction of whether you will be hired.
"""
        )


# =======================================================================
# APPLICATION ENTRY POINT
# =======================================================================

if __name__ == "__main__":
    main()

