from ultralytics import YOLO
import os
import cv2

model = YOLO("models/yolo12n.pt")

def run_object_detection(image_path):
    results = model(image_path)

    if not results:
        return None

    image = results[0].plot()

    # 🔹 SAVE OUTPUT
    filename = os.path.basename(image_path)
    save_path = os.path.join("results/object", filename)
    cv2.imwrite(save_path, image)

    return image
