"""
matcher.py
----------
Core NLP matching logic:

1. TF-IDF vectorization of resume text vs. job description text.
2. Cosine similarity between the two TF-IDF vectors.
3. Skill-gap set operations.
4. Transparent overall match score.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# Overall score weights
# ---------------------------------------------------------------------

TEXT_SIMILARITY_WEIGHT = 0.6
SKILL_COVERAGE_WEIGHT = 0.4


def compute_tfidf_similarity(resume_text, job_text):
    """
    Compute TF-IDF cosine similarity between resume and job description.

    Returns:
        tuple:
            similarity_score -> float between 0.0 and 1.0
            error_message    -> None if successful
    """

    # Validate input
    if not resume_text or not job_text:
        return 0.0, "Resume text or job description text is empty."

    try:
        # Create TF-IDF vectors using both documents
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        tfidf_matrix = vectorizer.fit_transform([
            resume_text,
            job_text
        ])

        # Handle empty vocabulary
        if tfidf_matrix.shape[1] == 0:
            return (
                0.0,
                "Not enough meaningful vocabulary to compare."
            )

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )

        similarity_score = float(
            similarity_matrix[0][0]
        )

        # Guarantee 0 <= score <= 1
        similarity_score = max(
            0.0,
            min(1.0, similarity_score)
        )

        return similarity_score, None

    except ValueError as exc:
        return (
            0.0,
            f"Could not compute text similarity: {exc}"
        )

    except Exception as exc:
        return (
            0.0,
            f"Unexpected similarity error: {exc}"
        )


def get_skill_gap(resume_skills, job_skills):
    """
    Compare resume skills against job-required skills.

    Returns:
        {
            "matched": skills present in both,
            "missing": skills required by job but absent in resume,
            "additional": skills in resume but not required by job
        }
    """

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    additional = resume_skills - job_skills

    return {
        "matched": matched,
        "missing": missing,
        "additional": additional
    }


def compute_skill_coverage(matched_skills, job_skills):
    """
    Calculate the fraction of job skills found in the resume.

    Returns:
        float between 0.0 and 1.0
    """

    if not job_skills:
        return 0.0

    coverage = len(matched_skills) / len(job_skills)

    return max(0.0, min(1.0, coverage))


def compute_overall_score(text_similarity, skill_coverage):
    """
    Calculate the final match score as a percentage.

    Formula:

        60% × Text Similarity
        +
        40% × Skill Coverage
    """

    # Protect against invalid values
    text_similarity = max(
        0.0,
        min(1.0, float(text_similarity))
    )

    skill_coverage = max(
        0.0,
        min(1.0, float(skill_coverage))
    )

    combined = (
        TEXT_SIMILARITY_WEIGHT * text_similarity
        +
        SKILL_COVERAGE_WEIGHT * skill_coverage
    )

    return round(combined * 100, 1)