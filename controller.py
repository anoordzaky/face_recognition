from decouple import config

from flask import Flask, request, abort, Response, send_from_directory

from utils import convert_image, crop_face_retina, img_cosine_similarity, load_from_bytes
from face_recognition_service import FaceRecognition
from database import Database

app = Flask(__name__)

db = Database()
model = FaceRecognition()

# load swagger UI json file


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route("/register", methods=['POST'])
def face_upload_package():
    uploaded_image = request.files['image_file'].read()
    name = request.form['text']
    image = load_from_bytes(uploaded_image)

    img, _ = convert_image(image)
    detections = model.detect_face(img)

    print(len(detections))
    if len(detections) == 0:
        abort(400, "Invalid input: No face detected!")
    elif len(detections) > 1:
        abort(400, "Invalid input: Multiple faces detected!")

    face = crop_face_retina(img, detections)
    embeddings = model.get_embeddings(face)

    # wrap the package in a dictionary
    package = {
        'name': name,
        'embedding': embeddings[0].tolist()
    }

    try:
        db.push_files(package)  # push package to database
        return Response("Wajah anda berhasil diregistrasi!", status=200)
    except:
        abort(500, "Internal server error.")


@app.route("/verification", methods=['POST'])
def face_recognition():

    uploaded_image = request.files['image_file'].read()

    image = load_from_bytes(uploaded_image)
    img, _ = convert_image(image)

    detections = model.detect_face(img)

    # if no face detected or multiple faces detected, return 400
    if len(detections) == 0:
        abort(400, "Invalid input: No face detected!")
    elif len(detections) > 1:
        abort(400, "Invalid input: Multiple faces detected!")

    face = crop_face_retina(img, detections)

    # get all entries in the database
    to_check = db.get_image_names()
    all_embeddings = to_check["embedding"]

    input_embeddings = model.get_embeddings(face)

    results = []
    # compute the cosine similarity to each available embeddings
    for i in range(len(all_embeddings)):
        results.append(img_cosine_similarity(
            input_embeddings, all_embeddings[i]))

    verify = max(results)

    if verify < 0.6:
        return Response("Face is not registered!", 401)
    name_index = results.index(verify)  # finds the index of the name
    detected_face = to_check["name"][name_index]

    return Response(f"Hello {detected_face}!", 200)
