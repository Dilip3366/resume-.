# AI-Powered Resume Screening System
### Python · Flask · NLP · TF-IDF · LightGBM · REST API

An end-to-end AI system that scores and ranks resumes against a job description
using Natural Language Processing and Machine Learning.

---

## Tech Stack

| Layer       | Technology                                  |
|-------------|---------------------------------------------|
| API Server  | Python 3.11, Flask 3.0, Flask-CORS          |
| NLP         | NLTK, TF-IDF (scikit-learn), cosine similarity |
| ML Model    | LightGBM + K-Fold Cross Validation          |
| File Parsing| PyPDF2 (PDF), python-docx (DOCX)            |
| Frontend    | HTML + CSS + Vanilla JS                     |

---

## Project Structure

```
resume-screener/
├── run.py                         ← Flask entry point
├── requirements.txt
├── .env                           ← Config variables
│
├── app/
│   ├── __init__.py                ← App factory
│   ├── routes/
│   │   ├── screen.py              ← /api/screen endpoints
│   │   └── health.py              ← /api/health
│   ├── utils/
│   │   ├── extractor.py           ← PDF / DOCX text extraction
│   │   ├── preprocessor.py        ← NLP pipeline (NLTK)
│   │   └── scorer.py              ← TF-IDF cosine similarity scoring
│   └── models/
│       └── resume_model.pkl       ← Saved LightGBM model (after training)
│
├── notebooks/
│   └── model_training.ipynb       ← Full ML training pipeline
│
├── static/
│   ├── index.html                 ← Frontend UI
│   └── uploads/                   ← Temp upload folder (auto-created)
│
└── tests/
    └── test_screener.py           ← Unit tests
```

---

## Setup in VS Code

### Step 1 — Open Project
```
File → Open Folder → select resume-screener/
```

### Step 2 — Create Virtual Environment
```bash
python3 -m venv venv

# Mac / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the API Server
```bash
python run.py
```
API starts at → http://localhost:5000

### Step 5 — Open the Frontend
Open `static/index.html` in your browser (double-click or use Live Server extension in VS Code).

---

## API Endpoints

### Single Resume Screen
```
POST /api/screen
Content-Type: multipart/form-data

Fields:
  resume  (file)    : PDF / DOCX / TXT
  jd_text (string)  : Job description text
```

### Bulk Resume Ranking
```
POST /api/screen/bulk
Content-Type: multipart/form-data

Fields:
  resumes (files, multiple) : Up to 20 resume files
  jd_text (string)          : Job description
```

### Text-only (no file)
```
POST /api/screen/text
Content-Type: application/json

{
  "resume_text": "...",
  "jd_text": "..."
}
```

### Health Check
```
GET /api/health
```

---

## Sample API Response

```json
{
  "status": "success",
  "filename": "dilip_resume.pdf",
  "final_score": 74.5,
  "cosine_score": 68.2,
  "skill_score": 90.0,
  "grade": "B",
  "verdict": "Good Match — Consider",
  "matched_skills": ["flask", "machine learning", "mysql", "python", "react"],
  "missing_skills": ["docker", "kubernetes"],
  "extra_skills": ["android", "lightgbm"],
  "candidate_info": {
    "education": "Bachelors",
    "experience_years": 0.0,
    "skills_found": ["flask", "html", "java", "javascript", "lightgbm", "mysql", "python", "react"]
  }
}
```

---

## Scoring Algorithm

```
Final Score = (0.70 × TF-IDF Cosine Similarity) + (0.30 × Skill Overlap Ratio)

Grade A: ≥ 75%  →  Strong Match — Shortlist
Grade B: ≥ 55%  →  Good Match — Consider
Grade C: ≥ 35%  →  Partial Match — Review Manually
Grade D: < 35%  →  Weak Match — Not Recommended
```

---

## Run Unit Tests

```bash
cd tests
python -m pytest test_screener.py -v
```

---

## Train the ML Model (Optional)

```bash
pip install jupyter
cd notebooks
jupyter notebook model_training.ipynb
```
Run all cells — saves `app/models/resume_model.pkl`

---

## Test with Postman

```
POST http://localhost:5000/api/screen/text
Content-Type: application/json

{
  "resume_text": "Python developer with Flask, machine learning, scikit-learn, pandas, MySQL, REST API experience. B.Tech Computer Science 2027.",
  "jd_text": "Looking for Python developer with Flask, ML, NLP, API deployment, SQL experience."
}
```

---

## Interview Talking Points (Infosys AI Engineering)

1. **NLP Pipeline**: Lowercase → remove URLs/emails → tokenize → remove stopwords → Porter stemming → TF-IDF vectorization
2. **TF-IDF**: Term Frequency × Inverse Document Frequency — gives high weight to rare but important words
3. **Cosine Similarity**: Measures angle between two document vectors in high-dimensional space (0=no match, 1=perfect match)
4. **LightGBM**: Gradient boosting on trees — faster than XGBoost, handles sparse TF-IDF well
5. **K-Fold CV**: Splits data into K parts, trains K times — each fold is test once; prevents overfitting
6. **Flask Blueprint**: Modular route organization — `/api/screen` and `/api/health` are separate blueprints
7. **REST API Design**: Stateless, JSON responses, proper HTTP status codes (200/400/500)

---

Built by Joga Dilip | B.Tech CSE | ANITS Visakhapatnam
Aligned with Infosys AI Engineering Focus Area
