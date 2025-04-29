import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import pymysql
import sys
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

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
        print(f"{id}|{name}|{desc}|matched_output.jpg")
    else:
        print("0|No Match|No Description|no_match_found.jpg")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("0|Error|Missing Argument|no_match_found.jpg")
    else:
        find_best_match(sys.argv[1])
