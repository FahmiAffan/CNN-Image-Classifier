from __future__ import annotations

import os
from uuid import uuid4
from flask import (
    Blueprint,
    current_app,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from .model import load_model_once, predict_image


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


bp = Blueprint("main", __name__)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part in request.")
        return redirect(url_for("main.index"))

    file = request.files["image"]

    if file.filename == "":
        flash("No selected file.")
        return redirect(url_for("main.index"))

    if not _allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG/JPG/GIF image.")
        return redirect(url_for("main.index"))

    # Persist the uploaded file
    filename = secure_filename(file.filename)
    unique_name = f"{uuid4().hex}_{filename}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    # Load model (once) and run prediction
    model = load_model_once()
    prediction, confidence = predict_image(model, save_path)

    file_url = url_for("main.uploaded_file", filename=unique_name)

    return render_template(
        "result.html",
        file_url=file_url,
        prediction=prediction,
        confidence=confidence,
    )


@bp.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename: str):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)
