""""
semantic_matcher.py
-------------------
Enhanced text similarity without PyTorch.

Uses:
1. Word-level TF-IDF
2. Word n-grams
3. Character n-grams
4. Combined cosine similarity

This module is separate from matcher.py so the existing
TF-IDF implementation remains untouched.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _safe_cosine_similarity(vectorizer, texts):
    """
    Fit a vectorizer and calculate cosine similarity
    between two pieces of text.
    """
    matrix = vectorizer.fit_transform(texts)

    if matrix.shape[1] == 0:
        return 0.0

    return float(
        cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    )


def compute_semantic_similarity(resume_text, job_text):
    """
    Compute enhanced semantic-style similarity.

    Returns:
        tuple:
            similarity_score -> float between 0.0 and 1.0
            error_message    -> None if successful
    """

    if not resume_text or not job_text:
        return 0.0, "Resume text or job description text is empty."

    try:
        texts = [resume_text, job_text]

        # -----------------------------------------
        # 1. Word-level similarity
        # -----------------------------------------
        word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        word_similarity = _safe_cosine_similarity(
            word_vectorizer,
            texts
        )

        # -----------------------------------------
        # 2. Character-level similarity
        # -----------------------------------------
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        )

        char_similarity = _safe_cosine_similarity(
            char_vectorizer,
            texts
        )

        # -----------------------------------------
        # 3. Hybrid similarity
        # -----------------------------------------
        similarity = (
            0.70 * word_similarity
            + 0.30 * char_similarity
        )

        similarity = max(0.0, min(1.0, similarity))

        return float(similarity), None

    except ValueError as exc:
        return 0.0, f"Could not compute semantic similarity: {exc}"

    except Exception as exc:
        return 0.0, f"Unexpected semantic similarity error: {exc}"