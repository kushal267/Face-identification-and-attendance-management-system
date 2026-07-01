import os
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
from deepface import DeepFace

dataset_path = "dataset"
if not os.path.exists(dataset_path):
    print("Error: dataset folder not found. Register at least one student first.")
    exit(1)

known_embeddings = []
known_ids = []
known_names = []
model_name = "Facenet"
detector_backend = "retinaface"

print("Loading DeepFace model... this may take a moment.")

persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
if not persons:
    print("Error: No student folders found in dataset. Register at least one student.")
    exit(1)

for person in persons:
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    if "_" in person and person.split("_", 1)[0].isdigit():
        user_id = int(person.split("_", 1)[0])
        user_name = person.split("_", 1)[1]
    else:
        user_id = -1
        user_name = person

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping {image_path}: cannot read image")
            continue

        try:
            representations = DeepFace.represent(
                img_path=image,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=False,
            )
        except Exception as e:
            print(f"Skipping {image_path}: represent failed ({e})")
            continue

        if not representations:
            print(f"Skipping {image_path}: no face found")
            continue

        embedding = representations[0]["embedding"]
        known_embeddings.append(embedding)
        known_ids.append(user_id)
        known_names.append(user_name)

if not known_embeddings:
    print("No face embeddings found. Add images to dataset and try again.")
    exit(1)

with open("face_model.pkl", "wb") as f:
    pickle.dump((known_embeddings, known_ids, known_names), f)

print("Training complete")
print(f"Saved {len(known_embeddings)} face embeddings.")
