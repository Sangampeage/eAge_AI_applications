import os
import cv2
import numpy as np
import faiss
import logging
import insightface
from insightface.app import FaceAnalysis
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

# ------------------- Configuration ---------------------
SIMILARITY_THRESHOLD = 0.6
KNOWN_FACES_NPZ = 'face_embeddings.npz'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FOLDER = os.path.join(ROOT_DIR, "Images")
processed_files = set()

# ------------------- Logging ---------------------
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('recognition_log.txt', mode='a')
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("InsightFaceMonitor")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ------------------- FAISS Index Load ---------------------
def load_faiss_index(npz_path):
    data = np.load(npz_path)
    embeddings = data['embeddings']
    names = data['names']
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine similarity
    index.add(embeddings)
    return index, names


# ------------------- Face Recognition ---------------------
def recognize_face(embedding, index, names, threshold):
    embedding = embedding.astype(np.float32)
    faiss.normalize_L2(embedding.reshape(1, -1))
    distances, indices = index.search(embedding.reshape(1, -1), 1)
    similarity = distances[0][0]
    best_match_idx = indices[0][0]
    if similarity > threshold:
        return names[best_match_idx], similarity
    return "Unknown", similarity


# ------------------- Load Models ---------------------
logger.info("Loading InsightFace and FAISS index...")
model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

try:
    faiss_index, known_names = load_faiss_index(KNOWN_FACES_NPZ)
    logger.info(f"Loaded {len(known_names)} known face(s).")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    faiss_index, known_names = None, []


# ------------------- Image Handler ---------------------
def recognize_faces_in_image(image_path):
    """
    Detects and recognizes faces in the given image.

    Returns:
        tuple: (annotated_image_rgb, results_list)
            - annotated_image_rgb: numpy array (H, W, 3) in RGB, ready for st.image()
            - results_list: list of dicts with keys 'label' and 'similarity'
        Returns (None, []) on failure.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Failed to load image: {image_path}")
        return None, []

    if faiss_index is None:
        logger.error("FAISS index not loaded.")
        return None, []

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = model.get(rgb_image)

    results = []

    if len(faces) == 0:
        logger.info(f"No faces detected in: {image_path}")
        # Return the unmodified image so Streamlit can still display it
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), []

    logger.info(f"Detected {len(faces)} face(s) in: {image_path}")

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        label = "Unknown"
        similarity = 0.0
        color = (255, 0, 0)  # Red for unknown (RGB)

        if hasattr(face, 'embedding'):
            label, similarity = recognize_face(
                face.embedding, faiss_index, known_names, SIMILARITY_THRESHOLD
            )
            if label != "Unknown":
                log_msg = f" Face recognized: {label} (similarity: {similarity:.2f})"
                color = (0, 255, 0)  # Green for known (RGB)
            else:
                log_msg = " Unknown face detected."
            logger.info(log_msg)

        display_label = f"{str(label).split('.')[0]} ({similarity:.2f})" if label != "Unknown" else "Unknown"

        # Draw on the RGB image directly (model.get() uses RGB input)
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            rgb_image, display_label, (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        results.append({
            "label": str(label).split('.')[0],
            "similarity": round(float(similarity), 3),
        })

    # Optionally save to disk (non-blocking — errors are swallowed)
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join("recognized", f"{filename}_{timestamp}.jpg")
        os.makedirs("recognized", exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved recognized image to: {output_path}")
    except Exception as save_err:
        logger.warning(f"Could not save annotated image: {save_err}")

    return rgb_image, results


# ------------------- Watchdog Handler ---------------------
class NewImageHandler(FileSystemEventHandler):
    def __init__(self, processed):
        self.processed = processed

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = event.src_path
            if image_path not in self.processed:
                logger.info(f"New image detected: {image_path}")
                time.sleep(0.5)
                recognize_faces_in_image(image_path)
                self.processed.add(image_path)


# ------------------- Start Folder Monitoring ---------------------
# if __name__ == "__main__":
#     logger.info("Starting folder monitoring...")
#     event_handler = NewImageHandler(processed_files)
#     observer = Observer()
#     observer.schedule(event_handler, IMAGE_FOLDER, recursive=False)
#
#     try:
#         observer.start()
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         logger.info("Stopping monitoring...")
#         observer.stop()
#     observer.join()