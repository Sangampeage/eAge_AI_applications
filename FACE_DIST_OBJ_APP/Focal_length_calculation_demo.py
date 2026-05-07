
import cv2
from ultralytics import YOLO
import time
import numpy as np

# Load the YOLO model for person detection
detection_model = YOLO("yolov8n.pt")  # For person detection
# Load the YOLO pose estimation model for shoulder detection
pose_model = YOLO("yolov8n-pose.pt")  # For pose/keypoint detection

# Known constants
KNOWN_DISTANCE = 100  # cm (reference distance)
KNOWN_SHOULDER_WIDTH = 41  # cm (average shoulder width)

# Shoulder keypoint indices (COCO keypoint format)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

def detect_shoulders(image, confidence_threshold=0.3):
    """Detect persons and their shoulder positions using pose estimation"""
    if image is None:
        return []
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # First detect persons to get bounding boxes
    detections = detection_model(image_rgb, verbose=False)
    
    persons = []
    
    for detection in detections:
        if detection.boxes:
            for box in detection.boxes:
                if detection_model.names[int(box.cls)] == "person" and float(box.conf) > confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    persons.append((x1, y1, x2, y2))
    
    # Now perform pose estimation within each person bounding box
    shoulder_measurements = []
    
    for (x1, y1, x2, y2) in persons:
        # Crop the person region
        person_roi = image[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            continue
            
        # Get pose keypoints
        pose_results = pose_model(person_roi, verbose=False)
        
        for result in pose_results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                
                # Get shoulder points (if visible)
                left_sh = keypoints[LEFT_SHOULDER] if result.keypoints.conf[0][LEFT_SHOULDER] > 0.3 else None
                right_sh = keypoints[RIGHT_SHOULDER] if result.keypoints.conf[0][RIGHT_SHOULDER] > 0.3 else None
                
                if left_sh is not None and right_sh is not None:
                    # Convert back to original image coordinates
                    left_sh_global = (int(left_sh[0] + x1), int(left_sh[1] + y1))
                    right_sh_global = (int(right_sh[0] + x1), int(right_sh[1] + y1))
                    
                    shoulder_measurements.append({
                        'box': (x1, y1, x2, y2),
                        'left_shoulder': left_sh_global,
                        'right_shoulder': right_sh_global,
                        'shoulder_width_px': np.linalg.norm(np.array(left_sh_global) - np.array(right_sh_global))
                    })
    
    return shoulder_measurements

# Load reference images for calibration
ref_images = []
ref_image_paths = ["image4.jpg", "image5.jpg"]  # Update with your paths

focal_lengths = []

for path in ref_image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not load reference image {path}")
        continue
    
    shoulder_data = detect_shoulders(img, confidence_threshold=0.4)
    
    if shoulder_data:
        # Use the person with largest bounding box
        largest_person = max(shoulder_data, key=lambda x: (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]))
        
        if 'shoulder_width_px' in largest_person:
            focal_length = (largest_person['shoulder_width_px'] * KNOWN_DISTANCE) / KNOWN_SHOULDER_WIDTH
            focal_lengths.append(focal_length)
            print(f"Reference image: shoulder width={largest_person['shoulder_width_px']:.1f}px, focal_length={focal_length:.1f}")

if not focal_lengths:
    print("Error: Could not calculate focal length from reference images!")
    exit()

focal_length = np.mean(focal_lengths)
print(f"Calibration complete. Average focal length: {focal_length:.1f}")



# # Real-time processing
# cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture("rtsp://admin:12345@192.168.1.100:554/Streaming/Channels/101")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# COLORS = [
#     (0, 255, 0), (255, 0, 0), (0, 0, 255),
#     (255, 255, 0), (255, 0, 255), (0, 255, 255)
# ]

# print("Starting real-time measurement. Press 'q' to quit...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     start_time = time.time()
    
#     shoulder_data = detect_shoulders(frame)
    
#     for i, person in enumerate(shoulder_data):
#         color = COLORS[i % len(COLORS)]
        
#         # Draw bounding box
#         x1, y1, x2, y2 = person['box']
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
#         # Draw shoulder points and line
#         ls = person['left_shoulder']
#         rs = person['right_shoulder']
#         cv2.circle(frame, ls, 5, color, -1)
#         cv2.circle(frame, rs, 5, color, -1)
#         cv2.line(frame, ls, rs, color, 2)
        
#         # Calculate and display distance
#         distance_cm = (KNOWN_SHOULDER_WIDTH * focal_length) / person['shoulder_width_px']
#         distance_m = distance_cm / 100
        
#         cv2.putText(frame, f"{distance_m:.1f}m", (x1, y1 - 10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
#     # Show FPS
#     fps = 1 / (time.time() - start_time)
#     cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
#                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
#     cv2.imshow("Shoulder-Based Distance Measurement", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


