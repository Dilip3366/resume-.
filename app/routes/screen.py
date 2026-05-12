from flask              import Blueprint, request, jsonify, current_app
from werkzeug.utils     import secure_filename
from app.utils.extractor    import extract_text, clean_text
from app.utils.preprocessor import (
    extract_skills, extract_education, extract_experience_years
)
from app.utils.scorer       import compute_match_score, rank_resumes
import os
import uuid

screen_bp = Blueprint('screen', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}


def _allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_file(file) -> str:
    """Save uploaded file with a unique name; return the saved path."""
    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    folder   = current_app.config['UPLOAD_FOLDER']
    path     = os.path.join(folder, filename)
    file.save(path)
    return path


# ─────────────────────────────────────────────────────────────────────
# POST /api/screen
#   Form-data:
#     - resume  (file)          : resume file (PDF / DOCX / TXT)
#     - jd_text (string)        : job description text
# ─────────────────────────────────────────────────────────────────────
@screen_bp.route('/screen', methods=['POST'])
def screen_single():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file uploaded'}), 400

    file    = request.files['resume']
    jd_text = request.form.get('jd_text', '').strip()

    if not file or not _allowed(file.filename):
        return jsonify({'error': 'Invalid file type. Use PDF, DOCX, or TXT'}), 400
    if not jd_text:
        return jsonify({'error': 'Job description text is required'}), 400

    path = None
    try:
        path        = _save_file(file)
        raw_text    = extract_text(path)
        clean       = clean_text(raw_text)

        # NLP feature extraction
        skills      = extract_skills(clean)
        education   = extract_education(clean)
        exp_years   = extract_experience_years(clean)

        # Scoring
        score_data  = compute_match_score(clean, jd_text)

        return jsonify({
            'status':           'success',
            'filename':         secure_filename(file.filename),
            'candidate_info': {
                'education':        education,
                'experience_years': exp_years,
                'skills_found':     skills,
            },
            **score_data,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)   # clean up uploaded file


# ─────────────────────────────────────────────────────────────────────
# POST /api/screen/bulk
#   Form-data:
#     - resumes (files, multiple): list of resume files
#     - jd_text (string)         : job description
# ─────────────────────────────────────────────────────────────────────
@screen_bp.route('/screen/bulk', methods=['POST'])
def screen_bulk():
    files   = request.files.getlist('resumes')
    jd_text = request.form.get('jd_text', '').strip()

    if not files:
        return jsonify({'error': 'No resumes uploaded'}), 400
    if not jd_text:
        return jsonify({'error': 'Job description is required'}), 400
    if len(files) > 20:
        return jsonify({'error': 'Maximum 20 resumes per request'}), 400

    resumes   = []
    saved     = []

    try:
        for f in files:
            if not _allowed(f.filename):
                continue
            path = _save_file(f)
            saved.append(path)
            raw  = clean_text(extract_text(path))
            resumes.append({'name': secure_filename(f.filename), 'text': raw})

        if not resumes:
            return jsonify({'error': 'No valid resume files found'}), 400

        ranked = rank_resumes(resumes, jd_text)

        return jsonify({
            'status':       'success',
            'total':        len(ranked),
            'job_skills':   sorted(extract_skills(jd_text)),
            'rankings':     ranked,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        for p in saved:
            if os.path.exists(p):
                os.remove(p)


# ─────────────────────────────────────────────────────────────────────
# POST /api/screen/text
#   JSON body:
#     { "resume_text": "...", "jd_text": "..." }
#   (No file upload needed — useful for testing with Postman)
# ─────────────────────────────────────────────────────────────────────
@screen_bp.route('/screen/text', methods=['POST'])
def screen_text():
    data        = request.get_json(silent=True) or {}
    resume_text = data.get('resume_text', '').strip()
    jd_text     = data.get('jd_text',     '').strip()

    if not resume_text:
        return jsonify({'error': 'resume_text is required'}), 400
    if not jd_text:
        return jsonify({'error': 'jd_text is required'}), 400

    clean      = clean_text(resume_text)
    skills     = extract_skills(clean)
    education  = extract_education(clean)
    exp_years  = extract_experience_years(clean)
    score_data = compute_match_score(clean, jd_text)

    return jsonify({
        'status':           'success',
        'candidate_info': {
            'education':        education,
            'experience_years': exp_years,
            'skills_found':     skills,
        },
        **score_data,
    }), 200
