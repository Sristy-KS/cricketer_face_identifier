import os
import pickle
import numpy as np
from deepface import DeepFace

DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "models/face_embeddings.pkl"

def build_face_embeddings():
    os.makedirs("models", exist_ok=True)
    known_embeddings = []
    known_labels = []

    player_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    print(f"Generating facial embeddings for classes: {player_names}")

    for player in player_names:
        player_folder = os.path.join(DATASET_DIR, player)
        for img_name in os.listdir(player_folder):
            img_path = os.path.join(player_folder, img_name)
            try:
                # Extract deep face embedding vector
                embedding_objs = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet512",
                    enforce_detection=False
                )
                if embedding_objs:
                    embedding = embedding_objs[0]["embedding"]
                    known_embeddings.append(embedding)
                    known_labels.append(player)
            except Exception as e:
                continue

    data = {
        "embeddings": np.array(known_embeddings),
        "labels": np.array(known_labels)
    }

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Extracted {len(known_embeddings)} face embeddings and saved to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    build_face_embeddings()