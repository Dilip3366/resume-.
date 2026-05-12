from flask import Blueprint, jsonify
import datetime

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':    'ok',
        'service':   'AI Resume Screener',
        'version':   '1.0.0',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    }), 200
