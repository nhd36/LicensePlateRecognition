from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import secrets
from models.LicenseRecognizeModel import predict_license_plate

app = Flask(__name__)
api = Api(app)

secret = secrets.token_urlsafe(32)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
app.config["SECRET_KEY"] = secret

def allowed_file(filename):
    file_type = filename.split(".")[1]
    if file_type in ALLOWED_EXTENSIONS:
        return True
    return False

class Upload(Resource):
    parse = reqparse.RequestParser()
    parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files', help='Please input a picture')
    def post(self):
        data = self.parse.parse_args()
        if "file" not in data:
            return {"message": "No 'file' part"}, 404
        file = data['file']
        if file.filename is None:
            return {"message": "No file selected"}, 404

        if not allowed_file(file.filename):
            return {"message": "This is not a picture, please input a picture"}, 400
        byte_image = file.read()
        np_arayPic = np.array(Image.open(BytesIO(byte_image)))
        license_plate = predict_license_plate(np_arayPic)
        return {"license_plate": license_plate,
                "message": "Success!"}, 201

api.add_resource(Upload, "/")

if __name__ == '__main__':
    app.run(debug=True)
