import os
import urllib.request
import cv2
import numpy as np
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def ensure_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(PROTOTXT_PATH):
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    if not os.path.exists(CAFFEMODEL_PATH):
        urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_PATH)

def detect_and_crop_face(image_input, target_size=(224, 224), min_face_confidence=0.6):
    ensure_model_files()
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

    if isinstance(image_input, Image.Image):
        img = np.array(image_input.convert('RGB'))
    else:
        file_bytes = np.asarray(bytearray(image_input), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h_img, w_img = img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    best_face = None
    max_conf = 0.0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_face_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
            (x1, y1, x2, y2) = box.astype("int")

            w = x2 - x1
            h = y2 - y1

            # Check for valid face aspect ratio (width / height should be between 0.45 and 1.3)
            aspect_ratio = float(w) / float(h) if h > 0 else 0

            if w > 40 and h > 40 and 0.45 <= aspect_ratio <= 1.3 and confidence > max_conf:
                max_conf = confidence
                best_face = (x1, y1, w, h)

    if best_face is None:
        resized = cv2.resize(img, target_size)
        return (resized.astype(np.float32) / 255.0), None, img

    x, y, w, h = best_face

    margin = int(0.12 * max(w, h))
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(w_img, x + w + margin)
    y_end = min(h_img, y + h + margin)

    face_crop = img[y_start:y_end, x_start:x_end]
    resized_face = cv2.resize(face_crop, target_size)
    normalized_face = resized_face.astype(np.float32) / 255.0

    return normalized_face, (x, y, w, h), img