from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from models.LicenseRecognizeModel import predict_license_plate
import os
import secrets

UPLOAD_FOLDER  = "upload_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

secret = secrets.token_urlsafe(32)
app = Flask(__name__)
app.config["SECRET_KEY"] = secret
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    file_type = filename.split(".")[1]
    if file_type in ALLOWED_EXTENSIONS:
        return True
    return False

@app.route("/upload", methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(url_for("upload_file"))
        file = request.files["file"]
        print(file.filename)
        if file.filename == "":
            flash("No file selected")
            return redirect(url_for("upload_file"))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            license_plate = predict_license_plate(f"upload_images/{file.filename}")
            flash("Uploaded File Completed")
            return render_template("upload.html", title="License Plate Scan", result=license_plate)
    return render_template("upload.html", title="License Plate Scan")


if __name__ == "__main__":
    app.run(debug=True)
