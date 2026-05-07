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
KNOWN_FACES_NPZ = 'face_embeddings.npz'  # Should contain 'embeddings' and 'names'
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
    exit(1)

# ------------------- Image Handler ---------------------
def recognize_faces_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Failed to load image: {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = model.get(rgb_image)

    if len(faces) == 0:
        logger.info(f"No faces detected in: {image_path}")
        return

    logger.info(f"Detected {len(faces)} face(s) in: {image_path}")

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        label = "Unknown"
        color = (0, 0, 255)

        if hasattr(face, 'embedding'):
            label, similarity = recognize_face(face.embedding, faiss_index, known_names, SIMILARITY_THRESHOLD)
            if label != "Unknown":
                log_msg = f" Face recognized: {label} (similarity: {similarity:.2f})"
                label = f"{label.split('.')[0]} ({similarity:.2f})"
                color = (0, 255, 0)
            else:
                log_msg = " Unknown face detected."
            logger.info(log_msg)

        # Draw
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join("recognized", f"{filename}_{timestamp}.jpg")
    os.makedirs("recognized", exist_ok=True)
    cv2.imwrite(output_path, image)
    logger.info(f"Saved recognized image to: {output_path}")

# ------------------- Watchdog Handler ---------------------
class NewImageHandler(FileSystemEventHandler):
    def __init__(self, processed):
        self.processed = processed

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = event.src_path
            if image_path not in self.processed:
                logger.info(f"New image detected: {image_path}")
                time.sleep(0.5)  # Allow file to fully write
                recognize_faces_in_image(image_path)
                self.processed.add(image_path)

# ------------------- Start Folder Monitoring ---------------------
# if __name__ == "__main__":
#     logger.info("Starting folder monitoring...")
#     event_handler = NewImageHandler(processed_files)
#     observer = Observer()
#     observer.schedule(event_handler, IMAGE_FOLDER, recursive=False)

#     try:
#         observer.start()
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         logger.info("Stopping monitoring...")
#         observer.stop()
#     observer.join()
