import os
import cv2
import numpy as np
import time
import math
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLO

# ------------------- Configuration -------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FOLDER = os.path.join(ROOT_DIR, "Images")
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
FOCAL_LENGTH = 900.0
processed_files = set()

KNOWN_PARTS = {
    (5, 6): 41,
    (11, 12): 34,
    (13, 14): 48,
    (15, 16): 23,
    (0, 11): 55,
    (0, 15): 145
}

SKELETON_LINES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6)
]

# ------------------- Logging Setup -------------------
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('pose_distance_log.txt', mode='a')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("PoseDistanceMonitor")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ------------------- Load Model -------------------
pose_model = YOLO("yolov8n-pose.pt")
logger.info("YOLOv8 Pose model loaded.")


# ------------------- Core Functions -------------------
def detect_keypoints(image):
    results = pose_model(image, verbose=False)
    keypoints_list = []
    for result in results:
        if result.keypoints is not None:
            for kp_tensor, conf_tensor in zip(result.keypoints.xy, result.keypoints.conf):
                kp = kp_tensor.cpu().numpy()
                conf = conf_tensor.cpu().numpy()
                # Each keypoint is stored as a tuple of floats, or None if low confidence
                keypoints = [
                    (float(kp[i][0]), float(kp[i][1])) if conf[i] > 0.3 else None
                    for i in range(len(kp))
                ]
                keypoints_list.append(keypoints)
    return keypoints_list


def corrected_distance(pt1, pt2, real_world_len):
    """
    Estimates real-world distance (cm) from two 2-D pixel points and a known
    real-world length between them, correcting for camera angle.
    Returns None when either point is missing or the pixel distance is too small.
    """
    if pt1 is None or pt2 is None:
        return None

    # Both pt1 and pt2 are plain Python (float, float) tuples here
    dx = float(pt1[0]) - float(pt2[0])
    dy = float(pt1[1]) - float(pt2[1])
    pixel_dist = math.hypot(dx, dy)

    if pixel_dist < 5:
        return None

    angle = abs(math.atan2(dy, dx))
    correction_factor = max(math.cos(angle), 0.4)
    corrected_px = pixel_dist / correction_factor
    return (real_world_len * FOCAL_LENGTH) / corrected_px


def _estimate_distance_from_keypoints(keypoints):
    """Internal helper: returns median estimated distance (cm) for one person."""
    distances = []
    for (i, j), real_len in KNOWN_PARTS.items():
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i] is not None and keypoints[j] is not None:
                d = corrected_distance(keypoints[i], keypoints[j], real_len)
                if d is not None:
                    distances.append(d)
    return float(np.median(distances)) if distances else None


# ------------------- Public Streamlit API -------------------
def estimate_distance(image_path):
    """
    Detects people via pose estimation and estimates distance from camera.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: (annotated_image_rgb, distances_list)
            - annotated_image_rgb: numpy array (H, W, 3) in RGB for st.image()
            - distances_list: list of dicts, one per person detected:
                  [{"person": 1, "distance_m": 1.23}, ...]
              distance_m is None when estimation fails for that person.
        Returns (None, []) on read failure.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Failed to load image: {image_path}")
        return None, []

    keypoints_list = detect_keypoints(image)
    distances_list = []

    if not keypoints_list:
        logger.info(f"No people detected in: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), []

    color = (0, 255, 0)  # BGR green

    for person_id, keypoints in enumerate(keypoints_list):
        distance_cm = _estimate_distance_from_keypoints(keypoints)
        distance_m = round(distance_cm / 100, 2) if distance_cm is not None else None

        # Draw keypoints
        for pt in keypoints:
            if pt is not None:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 4, color, -1)

        # Draw skeleton
        for i, j in SKELETON_LINES:
            if i < len(keypoints) and j < len(keypoints):
                if keypoints[i] is not None and keypoints[j] is not None:
                    cv2.line(
                        image,
                        (int(keypoints[i][0]), int(keypoints[i][1])),
                        (int(keypoints[j][0]), int(keypoints[j][1])),
                        color, 2
                    )

        # Annotate distance
        label = (
            f"Person {person_id + 1}: {distance_m}m"
            if distance_m is not None
            else f"Person {person_id + 1}: N/A"
        )
        if keypoints[0] is not None:
            pos = (int(keypoints[0][0]), max(int(keypoints[0][1]) - 10, 10))
        else:
            pos = (10, 40 + 30 * person_id)

        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        logger.info(f"{label} in {os.path.basename(image_path)}")

        distances_list.append({"person": person_id + 1, "distance_m": distance_m})

    # Optionally save result to disk
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{filename}_{timestamp}.jpg")
        cv2.imwrite(output_path, image)
        logger.info(f"Saved result to {output_path}")
    except Exception as save_err:
        logger.warning(f"Could not save result image: {save_err}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), distances_list


# ------------------- Legacy process_image (file-based, no return) -----------
def process_image(image_path):
    """Original watchdog-compatible function. Does not return anything."""
    img_rgb, distances = estimate_distance(image_path)
    # side-effect: image already saved inside estimate_distance


# ------------------- Watchdog Handler -------------------
class NewImageHandler(FileSystemEventHandler):
    def __init__(self, processed):
        self.processed = processed

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = event.src_path
            if image_path not in self.processed:
                logger.info(f"New image detected: {image_path}")
                time.sleep(0.5)
                process_image(image_path)
                self.processed.add(image_path)


# # ------------------- Start Monitoring -------------------
# if __name__ == "__main__":
#     logger.info("Starting pose-based distance estimation monitor...")
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