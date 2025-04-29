from flask import Flask, request, jsonify
import os
import numpy as np
import pymysql
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

app = Flask(__name__)

# Load VGG16 model without final classification layer
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)  # disable progress output
    return features.flatten()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_best_match(query_path):
    query_vec = extract_features(query_path)

    conn = pymysql.connect(host="localhost", user="u800183464_familyhub", password="Nf1US9:b*", db="u800183464_familyhub")
    cur = conn.cursor()
    cur.execute("SELECT id, name, description, image FROM items")
    rows = cur.fetchall()

    best_score = -1
    best_match = None

    for row in rows:
        id, name, desc, image_path = row
        if not os.path.exists(image_path):
            continue
        try:
            db_vec = extract_features(image_path)
            score = cosine_similarity(query_vec, db_vec)
            if score > best_score:
                best_score = score
                best_match = (id, name, desc, image_path)
        except:
            continue

    if best_match and best_score > 0.4:
        id, name, desc, match_path = best_match
        Image.open(match_path).save("matched_output.jpg")
        return {"id": id, "name": name, "description": desc, "image": "matched_output.jpg"}
    else:
        return {"id": 0, "name": "No Match", "description": "No Description", "image": "no_match_found.jpg"}

@app.route('/')
def home():
    return "Image Matching API is running."

@app.route('/match', methods=['POST'])
def match_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    save_path = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(save_path)

    try:
        result = find_best_match(save_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
