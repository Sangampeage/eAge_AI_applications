import streamlit as st
import os
import cv2

from Face_recognition_demo import recognize_faces_in_image
from Distance_calculation_demo import estimate_distance
from Object_detection_demo import run_object_detection

st.set_page_config(page_title="Vision Dashboard", layout="wide")
st.title("Vision Dashboard")

task = st.sidebar.selectbox(
    "Select Task",
    ["Face Recognition", "Pose Distance", "Object Detection"]
)

# Robust path handling
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FOLDER = os.path.join(ROOT_DIR, "Images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

if not image_files:
    st.warning(f"No images found in `{IMAGE_FOLDER}`. Please add images and refresh.")
    st.stop()

selected_image = st.selectbox("Select Image", image_files)
image_path = os.path.join(IMAGE_FOLDER, selected_image)

# Show the raw input image for reference
st.subheader("Input Image")
raw = cv2.imread(image_path)
if raw is not None:
    st.image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), use_column_width=True)

if st.button("Run Task"):
    st.subheader("Result")

    # ── Face Recognition ──────────────────────────────────────────────────────
    if task == "Face Recognition":
        with st.spinner("Running face recognition…"):
            img, results = recognize_faces_in_image(image_path)

        if img is None:
            st.error("Could not load or process the image. Check logs for details.")
        else:
            st.image(img, caption="Annotated — Face Recognition", use_column_width=True)

            if results:
                st.markdown("**Detected Faces:**")
                for r in results:
                    similarity_pct = f"{r['similarity'] * 100:.1f}%"
                    if r["label"] == "Unknown":
                        st.write(f"• ❓ Unknown face (similarity: {similarity_pct})")
                    else:
                        st.write(f"• ✅ **{r['label']}** (similarity: {similarity_pct})")
            else:
                st.info("No faces detected in this image.")

    # ── Pose Distance ─────────────────────────────────────────────────────────
    elif task == "Pose Distance":
        with st.spinner("Running pose-based distance estimation…"):
            img, distances = estimate_distance(image_path)

        if img is None:
            st.error("Could not load or process the image. Check logs for details.")
        else:
            st.image(img, caption="Annotated — Pose Distance", use_column_width=True)

            if distances:
                st.markdown("**Estimated Distances:**")
                for d in distances:
                    dist_str = (
                        f"{d['distance_m']} m"
                        if d["distance_m"] is not None
                        else "Unable to estimate"
                    )
                    st.write(f"• Person {d['person']}: {dist_str}")
            else:
                st.info("No people detected in this image.")

    # ── Object Detection ──────────────────────────────────────────────────────
    elif task == "Object Detection":
        with st.spinner("Running object detection…"):
            img = run_object_detection(image_path)

        if img is None:
            st.error("Could not load or process the image.")
        else:
            st.image(img, caption="Annotated — Object Detection", use_column_width=True)