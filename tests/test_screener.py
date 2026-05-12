import unittest
import sys
sys.path.append('..')

from app.utils.preprocessor import preprocess, extract_skills, extract_education, extract_experience_years
from app.utils.scorer       import compute_match_score, rank_resumes


class TestPreprocessor(unittest.TestCase):

    def test_preprocess_removes_stopwords(self):
        text   = 'I have experience working with Python and machine learning'
        result = preprocess(text)
        self.assertNotIn('have', result)
        self.assertNotIn('with', result)
        self.assertNotIn('and', result)

    def test_preprocess_returns_string(self):
        self.assertIsInstance(preprocess('Hello World'), str)

    def test_extract_skills_python(self):
        text   = 'Experienced in Python, Flask, React and MySQL'
        skills = extract_skills(text)
        self.assertIn('python', skills)
        self.assertIn('flask',  skills)
        self.assertIn('react',  skills)
        self.assertIn('mysql',  skills)

    def test_extract_education_bachelors(self):
        text = 'B.Tech in Computer Science from ANITS Visakhapatnam'
        self.assertEqual(extract_education(text), 'Bachelors')

    def test_extract_education_masters(self):
        text = 'M.Tech from IIT Bombay'
        self.assertEqual(extract_education(text), 'Masters')

    def test_extract_experience_years(self):
        text = 'I have 3 years of experience in software development'
        self.assertEqual(extract_experience_years(text), 3.0)

    def test_extract_experience_none(self):
        text = 'Fresh graduate with no work experience'
        self.assertEqual(extract_experience_years(text), 0.0)


class TestScorer(unittest.TestCase):

    STRONG_RESUME = (
        'python machine learning nlp scikit-learn lightgbm flask rest api '
        'pandas numpy tensorflow deep learning data science sql'
    )

    WEAK_RESUME = (
        'html css basic photoshop microsoft word excel powerpoint'
    )

    JD = (
        'Looking for a python developer with machine learning nlp flask '
        'experience. Must know scikit-learn pandas and rest api development.'
    )

    def test_strong_resume_scores_higher(self):
        strong = compute_match_score(self.STRONG_RESUME, self.JD)
        weak   = compute_match_score(self.WEAK_RESUME,   self.JD)
        self.assertGreater(strong['final_score'], weak['final_score'])

    def test_score_within_range(self):
        result = compute_match_score(self.STRONG_RESUME, self.JD)
        self.assertGreaterEqual(result['final_score'], 0)
        self.assertLessEqual(result['final_score'],    100)

    def test_result_has_required_keys(self):
        result = compute_match_score(self.STRONG_RESUME, self.JD)
        for key in ('final_score', 'grade', 'verdict', 'matched_skills', 'missing_skills'):
            self.assertIn(key, result)

    def test_rank_resumes(self):
        resumes = [
            {'name': 'weak.pdf',   'text': self.WEAK_RESUME},
            {'name': 'strong.pdf', 'text': self.STRONG_RESUME},
        ]
        ranked = rank_resumes(resumes, self.JD)
        self.assertEqual(ranked[0]['name'], 'strong.pdf')
        self.assertEqual(ranked[0]['rank'], 1)
        self.assertEqual(ranked[1]['rank'], 2)

    def test_grade_strong_match(self):
        result = compute_match_score(self.STRONG_RESUME, self.JD)
        self.assertIn(result['grade'], ['A', 'B'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
