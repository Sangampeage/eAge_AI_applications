import streamlit as st
import os
from Face_recognition_demo import recognize_faces_in_image
from Distance_calculation_demo import estimate_distance
# from Object_detection_demo import detect_objects

st.title("Vision Dashboard")

task = st.sidebar.selectbox(
    "Select Task",
    ["Face Recognition", "Pose Distance", "Object Detection"]
)

# Robust path handling
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FOLDER = os.path.join(ROOT_DIR, "Images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

selected_image = st.selectbox("Select Image", image_files)

image_path = os.path.join(IMAGE_FOLDER, selected_image)

if st.button("Run Task"):
    if task == "Face Recognition":
        img, results = run_face_recognition(image_path)
        st.image(img)
        st.write(results)

    elif task == "Pose Distance":
        img, distances = run_pose_distance(image_path)
        st.image(img)
        st.write(distances)

    elif task == "Object Detection":
        img = run_object_detection(image_path)
        st.image(img)
