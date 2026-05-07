from ultralytics import YOLO

model = YOLO("yolo12n.pt")

def run_object_detection(image_path):
    results = model(image_path)
    return results[0].plot()
