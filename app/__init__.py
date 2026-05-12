from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

def create_app():
    app = Flask(__name__, static_folder='../static')
    CORS(app)

    app.config['SECRET_KEY']          = os.getenv('SECRET_KEY', 'dev-secret')
    app.config['MAX_CONTENT_LENGTH']  = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    app.config['UPLOAD_FOLDER']       = os.getenv('UPLOAD_FOLDER', 'static/uploads')
    app.config['MODEL_PATH']          = os.getenv('MODEL_PATH', 'app/models/resume_model.pkl')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    from app.routes.screen  import screen_bp
    from app.routes.health  import health_bp

    app.register_blueprint(screen_bp,  url_prefix='/api')
    app.register_blueprint(health_bp,  url_prefix='/api')

    return app
