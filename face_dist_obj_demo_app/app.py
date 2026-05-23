




import streamlit as st
import os
import time
import sys

# Ensure the current directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pipelines.face_recognition import recognize_faces_in_image
from pipelines.pose_distance import process_image
from pipelines.object_detection import run_object_detection

from db.db import (
    init_db,
    insert_input_image,
    get_new_images,
    mark_processing,
    mark_done,
    mark_failed,
    insert_face_result,
    insert_pose_result,
    insert_object_result
)

# ------------------ Initialize Database ------------------
init_db()

INPUT_DIR = "../Images"
REFRESH_INTERVAL = 3  # seconds

st.set_page_config(layout="centered")
st.title("Vision Dashboard – Auto Processing Mode")

# ------------------ Load Images ------------------
if not os.path.exists(INPUT_DIR):
    st.error("Input directory does not exist")
    st.stop()

image_files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

# Insert images into DB (idempotent)
for f in image_files:
    image_path = os.path.join(INPUT_DIR, f)
    insert_input_image(f, image_path)

# Fetch NEW images from DB
new_images = get_new_images()

st.subheader("Incoming Images")

if not new_images:
    st.info("Waiting for new images...")
else:
    for image_id, filename, image_path in new_images:
        st.markdown("---")
        st.subheader(f"Processing: {filename}")

        # mark PROCESSING
        mark_processing(image_id)

        try:
            # Create results directories
            os.makedirs("results/face", exist_ok=True)
            os.makedirs("results/distance", exist_ok=True)
            os.makedirs("results/object", exist_ok=True)

            # -------- Face Recognition --------
            st.markdown("### 🧑 Face Recognition")
            img_face, face_results = recognize_faces_in_image(image_path)

            if img_face is not None:
                st.image(img_face, channels="BGR")
                output_path = os.path.join("results/face", filename)
                insert_face_result(image_id, output_path, face_results)
            if face_results:
                st.json(face_results)
            else:
                st.info("No faces detected")

            # -------- Pose Distance --------
            st.markdown("### 🧍 Pose Distance Estimation")
            img_pose, distances = process_image(image_path)

            if img_pose is not None:
                st.image(img_pose, channels="BGR")
                output_path = os.path.join("results/distance", filename)
                insert_pose_result(image_id, output_path, len(distances), distances)
            if distances:
                st.write([
                    f"Person {i+1}: {d/100:.2f} m" if d else f"Person {i+1}: N/A"
                    for i, d in enumerate(distances)
                ])
            else:
                st.info("No people detected")

            # -------- Object Detection --------
            st.markdown("### 📦 Object Detection")
            img_obj, obj_results = run_object_detection(image_path)

            if img_obj is not None:
                st.image(img_obj, channels="BGR")
                output_path = os.path.join("results/object", filename)
                insert_object_result(image_id, output_path, len(obj_results), obj_results)
            else:
                st.info("No objects detected")

            # mark DONE
            mark_done(image_id)

        except Exception as e:
            mark_failed(image_id)
            st.error(f"Processing failed: {e}")

# ------------------ Auto Refresh Loop ------------------
time.sleep(REFRESH_INTERVAL)
st.rerun()
