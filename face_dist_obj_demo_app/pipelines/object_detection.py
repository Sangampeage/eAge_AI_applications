from ultralytics import YOLO
import os
import cv2

# Load the YOLO model at the module level
model = YOLO("models/yolo12n.pt")

def run_object_detection(image_path):
    # Run inference
    results = model(image_path)

    if not results:
        return None, []

    # Get the plotted image (with bounding boxes)
    image = results[0].plot()

    # 🔹 SAVE OUTPUT
    filename = os.path.basename(image_path)
    output_dir = "results/object"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)

    # 🔹 Extract results for DB
    detections = []
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                detections.append({
                    "class": model.names.get(cls_id, f"Unknown({cls_id})"),
                    "conf": float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })

    return image, detections
