from flask import Flask, request, jsonify
import os
from your_script import find_best_match  # assuming your existing code is in your_script.py

app = Flask(__name__)

@app.route('/')
def home():
    return "Image similarity API is running."

@app.route('/match', methods=['POST'])
def match_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    save_path = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(save_path)

    try:
        output = find_best_match(save_path)
        return jsonify({'result': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
