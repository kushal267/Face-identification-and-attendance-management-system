import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
import logging
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

import cv2
import pickle
import sqlite3
import numpy as np
try:
    from deepface import DeepFace
except ImportError:
    print("Error: deepface package is not installed. Install it with pip install deepface.")
    sys.exit(1)
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
model_name = "Facenet"
detector_backend = "opencv"
threshold = 0.4
frame_skip = 8
resize_scale = 0.4

model_path = os.path.join(BASE_DIR, "face_model.pkl")
if not os.path.exists(model_path):
    print("Error: face_model.pkl not found. Run training.py first.")
    sys.exit(1)

with open(model_path, "rb") as f:
    data = pickle.load(f)

if not isinstance(data, tuple):
    print("Error: face_model.pkl has unexpected data format")
    sys.exit(1)

if len(data) == 3:
    known_embeddings, known_ids, known_names = data
elif len(data) == 2:
    known_embeddings, known_names = data
    known_ids = [-1] * len(known_names)
else:
    print(f"Error: face_model.pkl tuple length {len(data)} is not supported")
    sys.exit(1)

shown_messages = set()
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Cannot open webcam. Check your camera connection.")
    sys.exit(1)
frame_count = 0
results = []


def cosine_distance(emb1, emb2):
    emb1 = np.array(emb1, dtype=np.float32)
    emb2 = np.array(emb2, dtype=np.float32)
    if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
        return 1.0
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


while True:
    ret, img = cam.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        if results:
            for result in results:
                left, top, right, bottom = result["box"]
                name = result["name"]
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                label_y = top - 10 if top - 10 > 10 else top + 25
                cv2.putText(img, name, (left, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Face Attendance System", img)
        if cv2.waitKey(1) == 27:
            break
        continue

    small_img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)

    try:
        representations = DeepFace.represent(
            img_path=small_img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=False,
        )
    except Exception as e:
        print(f"DeepFace error: {e}")
        representations = []

    results = []
    for rep in representations:
        embedding = rep.get("embedding")
        facial_area = rep.get("facial_area", {})
        left = int(facial_area.get("x", 0) / resize_scale)
        top = int(facial_area.get("y", 0) / resize_scale)
        right = int((facial_area.get("x", 0) + facial_area.get("w", 0)) / resize_scale)
        bottom = int((facial_area.get("y", 0) + facial_area.get("h", 0)) / resize_scale)

        name = "Unknown"
        user_id = None
        if known_embeddings and embedding is not None:
            distances = [cosine_distance(embedding, known_emb) for known_emb in known_embeddings]
            best_index = int(np.argmin(distances))
            best_distance = distances[best_index]
            if best_distance <= threshold:
                user_id = known_ids[best_index]
                name = known_names[best_index]

        display_name = name
        if user_id is not None and user_id != -1:
            display_name = f"{name} ({user_id})"

        if name == "Unknown":
            if "Unknown" not in shown_messages:
                print("\n⚠  : Unknown Person Detected\n")
                shown_messages.add("Unknown")
        else:
            now = datetime.now()
            date = now.strftime("%d-%m-%Y")
            time = now.strftime("%H:%M:%S")

            conn = sqlite3.connect("database/face_db.db")
            cursor = conn.cursor()
            if user_id is not None and user_id != -1:
                cursor.execute("SELECT * FROM attendance WHERE user_id=? AND date=?", (user_id, date))
            else:
                cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
            record = cursor.fetchone()

            if record is None:
                cursor.execute(
                    "INSERT INTO attendance(user_id, name, date, time) VALUES(?,?,?,?)",
                    (user_id, name, date, time)
                )
                conn.commit()
                if display_name not in shown_messages:
                    print(f"{display_name} Attendance Marked")
                    shown_messages.add(display_name)
            else:
                if display_name not in shown_messages:
                    print(f"{display_name} Already Marked Today")
                    shown_messages.add(display_name)

            conn.close()

        results.append({"box": (left, top, right, bottom), "name": display_name})

    cv2.imshow("Face Attendance System", img)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
