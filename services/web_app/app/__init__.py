from flask import Flask

from shared.db import init_db

from .settings import settings


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.flask_secret_key,
        MAX_CONTENT_LENGTH=settings.max_upload_mb * 1024 * 1024,
        DATA_DIR=settings.data_dir,
        INFERENCE_API_URL=settings.inference_api_url,
        REDIS_URL=settings.redis_url,
    )

    init_db()

    from .routes import main, upload, jobs
    app.register_blueprint(main.bp)
    app.register_blueprint(upload.bp)
    app.register_blueprint(jobs.bp)

    return app
