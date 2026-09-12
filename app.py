import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace
from src.face_detector import detect_and_crop_face
import cv2

st.set_page_config(page_title="Cricket Face Identification", page_icon="🏏", layout="wide")

st.title("🏏 Indian Cricket Team Face Identification")
st.caption("State-of-the-Art Face Recognition with Deep Face Embeddings (FaceNet-512).")

EMBEDDINGS_FILE = "models/face_embeddings.pkl"

@st.cache_resource
def load_db():
    if not os.path.exists(EMBEDDINGS_FILE):
        return None, None
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["labels"]

db_embeddings, db_labels = load_db()

if db_embeddings is None:
    st.error("⚠️ Embeddings database not found. Run `python train.py` first.")
else:
    uploaded_file = st.file_uploader("Upload an image of a cricketer:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_face, bbox, original_img = detect_and_crop_face(image)

        col1, col2 = st.columns([1, 1])

        with col1:
            display_img = original_img.copy()
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                st.image(display_img, caption="Detected Face Region", use_container_width=True)
            else:
                st.image(display_img, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Prediction Result")

            if bbox is None:
                st.error("❌ **No Clear Face Detected**")
                st.info("Facial features are occluded. Please upload an image with a visible face.")
            else:
                # Convert face crop back to uint8 image for DeepFace
                face_uint8 = (processed_face * 255).astype(np.uint8)
                
                try:
                    query_objs = DeepFace.represent(
                        img_path=face_uint8,
                        model_name="Facenet512",
                        enforce_detection=False
                    )
                    query_embedding = np.array(query_objs[0]["embedding"])

                    # Compute Cosine Similarity against all enrolled player embeddings
                    dot_products = np.dot(db_embeddings, query_embedding)
                    norms = np.linalg.norm(db_embeddings, axis=1) * np.linalg.norm(query_embedding)
                    similarities = dot_products / norms

                    # Group scores by player class
                    unique_labels = sorted(list(set(db_labels)))
                    class_scores = {}
                    for player in unique_labels:
                        mask = (db_labels == player)
                        # Mean of top-3 highest similarity matches for that player
                        player_sims = np.sort(similarities[mask])[::-1]
                        class_scores[player] = float(np.mean(player_sims[:3])) if len(player_sims) > 0 else 0.0

                    best_player = max(class_scores, key=class_scores.get)
                    best_score = class_scores[best_player]

                    # Scale cosine similarity (-1 to 1) to percentage
                    display_confidence = max(0.0, min(100.0, (best_score - 0.2) / 0.6 * 100))

                    if best_score < 0.35:
                        st.warning("⚠️ **Low Confidence Match**")
                        st.info("The player is likely not in the database or facial features are heavily obscured.")
                    else:
                        st.success(f"**Identified Player:** {best_player.replace('_', ' ').title()}")
                        st.metric("Confidence Level", f"{display_confidence:.2f}%")

                    st.write("---")
                    st.subheader("Similarity Distribution")
                    for player, score in sorted(class_scores.items(), key=lambda x: x[1], reverse=True):
                        pct = max(0.0, min(1.0, (score - 0.2) / 0.6))
                        st.progress(pct, text=f"{player.replace('_', ' ').title()}: {pct*100:.2f}%")

                except Exception as e:
                    st.error(f"Inference error: {e}")