import cv2
import torch
from ultralytics import YOLO

# Load YOLO model (YOLOv8 pre-trained model)
model = YOLO(f".\models\yolov8n.pt")  # Use "yolov8s.pt" for a better model

def detect_objects(frame):
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame