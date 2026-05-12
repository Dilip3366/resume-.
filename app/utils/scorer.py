from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from app.utils.preprocessor          import preprocess, extract_skills
import numpy as np


def compute_match_score(resume_text: str, job_description: str) -> dict:
    """
    Core matching engine.

    Steps:
      1.  Preprocess both texts (NLP pipeline)
      2.  Build TF-IDF matrix
      3.  Compute cosine similarity → base score
      4.  Boost by skill overlap ratio
      5.  Return rich scoring breakdown
    """
    # ── 1. Preprocess ──────────────────────────────────────────────
    resume_clean = preprocess(resume_text)
    jd_clean     = preprocess(job_description)

    # ── 2. TF-IDF Vectorization ────────────────────────────────────
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=5000,
        sublinear_tf=True,    # log normalization
    )
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])

    # ── 3. Cosine Similarity ───────────────────────────────────────
    cos_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

    # ── 4. Skill Overlap ───────────────────────────────────────────
    resume_skills = set(extract_skills(resume_text))
    jd_skills     = set(extract_skills(job_description))

    matched_skills  = resume_skills & jd_skills
    missing_skills  = jd_skills - resume_skills
    extra_skills    = resume_skills - jd_skills

    skill_ratio = len(matched_skills) / max(len(jd_skills), 1)

    # ── 5. Composite Score (70% TF-IDF + 30% skill overlap) ────────
    final_score = round((0.70 * cos_score + 0.30 * skill_ratio) * 100, 2)
    final_score = min(final_score, 100.0)

    # ── 6. Grade ───────────────────────────────────────────────────
    if final_score >= 75:
        grade, verdict = 'A', 'Strong Match — Shortlist'
    elif final_score >= 55:
        grade, verdict = 'B', 'Good Match — Consider'
    elif final_score >= 35:
        grade, verdict = 'C', 'Partial Match — Review Manually'
    else:
        grade, verdict = 'D', 'Weak Match — Not Recommended'

    return {
        'final_score':     final_score,
        'cosine_score':    round(cos_score * 100, 2),
        'skill_score':     round(skill_ratio * 100, 2),
        'grade':           grade,
        'verdict':         verdict,
        'matched_skills':  sorted(matched_skills),
        'missing_skills':  sorted(missing_skills),
        'extra_skills':    sorted(extra_skills),
        'resume_skills':   sorted(resume_skills),
        'jd_skills':       sorted(jd_skills),
    }


def rank_resumes(resumes: list[dict], job_description: str) -> list[dict]:
    """
    Score and rank a list of resumes against one job description.

    `resumes` is a list of dicts: [{'name': str, 'text': str}, ...]
    Returns sorted list (highest score first) with rank added.
    """
    results = []
    for r in resumes:
        score_data = compute_match_score(r['text'], job_description)
        results.append({
            'name':  r['name'],
            **score_data,
        })

    results.sort(key=lambda x: x['final_score'], reverse=True)
    for idx, r in enumerate(results, 1):
        r['rank'] = idx

    return results
