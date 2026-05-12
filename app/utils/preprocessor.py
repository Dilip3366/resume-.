import re
import nltk
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK data on first run
for pkg in ('punkt', 'stopwords', 'averaged_perceptron_tagger'):
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words('english'))

# Extra domain-specific stop words for resumes
_RESUME_STOPWORDS = {
    'experience', 'work', 'project', 'projects', 'using', 'used',
    'developed', 'built', 'responsible', 'team', 'company', 'role',
    'including', 'also', 'etc', 'years', 'year', 'month', 'months',
}
_stopwords |= _RESUME_STOPWORDS


def preprocess(text: str, stem: bool = True) -> str:
    """
    Full NLP pipeline:
      1. Lowercase
      2. Remove URLs, emails, numbers
      3. Tokenize
      4. Remove stopwords
      5. Optional stemming
    Returns a clean space-joined token string for TF-IDF.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 2]

    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]

    return ' '.join(tokens)


def extract_skills(text: str) -> list[str]:
    """
    Scan raw resume text for known technical skills.
    Returns a sorted list of matched skills.
    """
    SKILL_KEYWORDS = [
        # Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go',
        'rust', 'kotlin', 'swift', 'r', 'scala', 'php', 'ruby',
        # Web
        'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
        'spring', 'springboot', 'fastapi', 'html', 'css', 'bootstrap',
        # Data / ML
        'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch',
        'scikit-learn', 'pandas', 'numpy', 'lightgbm', 'xgboost', 'keras',
        'computer vision', 'opencv',
        # Data Engg
        'sql', 'mysql', 'postgresql', 'mongodb', 'firebase', 'redis',
        'kafka', 'spark', 'hadoop', 'hive', 'airflow',
        # Cloud / DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
        'git', 'github', 'ci/cd', 'terraform', 'ansible',
        # Other
        'rest api', 'graphql', 'microservices', 'agile', 'scrum',
        'android', 'ios', 'flutter', 'react native',
    ]

    text_lower = text.lower()
    found = [s for s in SKILL_KEYWORDS if s in text_lower]
    return sorted(set(found))


def extract_education(text: str) -> str:
    """Best-effort education level extraction."""
    text_lower = text.lower()
    if any(k in text_lower for k in ('phd', 'doctorate', 'ph.d')):
        return 'PhD'
    if any(k in text_lower for k in ('master', 'm.tech', 'mba', 'msc', 'm.e')):
        return 'Masters'
    if any(k in text_lower for k in ('bachelor', 'b.tech', 'b.e', 'bsc', 'b.sc', 'b.com', 'bca')):
        return 'Bachelors'
    if any(k in text_lower for k in ('diploma', 'polytechnic')):
        return 'Diploma'
    return 'Not Detected'


def extract_experience_years(text: str) -> float:
    """
    Heuristic: find patterns like '3 years', '2+ years experience'.
    Returns the maximum found value (0 if none).
    """
    patterns = [
        r'(\d+\.?\d*)\s*\+?\s*years?\s+of\s+experience',
        r'(\d+\.?\d*)\s*\+?\s*years?\s+experience',
        r'experience\s+of\s+(\d+\.?\d*)\s*\+?\s*years?',
        r'(\d+\.?\d*)\s*\+?\s*yrs?',
    ]
    found = []
    for pat in patterns:
        matches = re.findall(pat, text.lower())
        found.extend(float(m) for m in matches)
    return max(found) if found else 0.0
