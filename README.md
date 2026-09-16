Indian Cricket Team Face Identification System
An end-to-end Computer Vision and Deep Learning application designed to detect, crop, and identify Indian cricket players using pre-trained deep facial representations and cosine similarity scoring.

📌 Project Overview
Traditional image classifiers struggle with varying poses, camera angles, and occlusions because they frequently overfit to jersey patterns and backgrounds. This project resolves that by using a two-stage computer vision pipeline:

Face Localization & Validation: An OpenCV deep neural network (ResNet-10 SSD) identifies facial coordinates and filters out non-face regions or extreme occlusion angles.
Deep Representation & Matching: The aligned face crop is passed through a FaceNet-512 backbone via DeepFace to generate high-dimensional facial embeddings, which are scored against enrolled reference templates using vector cosine similarity.
🚀 Key Features
DNN Face Detection: Real-time localization using Caffe-based ResNet-10 SSD for robust detection across frontal and profile angles.
Aspect-Ratio Guardrails: Automated rejection of false positives (e.g., bats, helmets, background clutter) and heavily occluded angles.
Deep Facial Embeddings: 512-dimensional vector extraction using FaceNet-512 to evaluate facial geometry rather than background pixels.
Confidence & Margin Scoring: Dynamic thresholding and lead-margin metrics to distinguish between confident predictions and out-of-distribution inputs.
Interactive UI: A real-time dashboard built with Streamlit displaying detected face crops, match confidence, and multi-class probability distributions.
🛠️ Tech Stack
Language: Python
Computer Vision: OpenCV (DNN module)
Deep Learning Frameworks: TensorFlow / Keras, tf-keras
Facial Recognition Engine: DeepFace (FaceNet-512)
Web Interface: Streamlit
Numerical Processing: NumPy
📂 Project Structure
cricket_face_identifier/
│
├── dataset/                    # Player image folders (raw training references)
│   ├── hardik_pandya/
│   ├── jasprit_bumrah/
│   ├── ms_dhoni/
│   ├── rohit_sharma/
│   └── virat_kohli/
│
├── models/
│   ├── deploy.prototxt         # OpenCV SSD face detector architecture
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── face_embeddings.pkl     # Pre-computed reference embedding vectors
│
├── src/
│   └── face_detector.py        # Face detection, cropping, and validation logic
│
├── app.py                      # Streamlit web application
├── train.py                    # Embedding extraction & database generation
├── requirements.txt            # Environment dependencies
└── README.md
