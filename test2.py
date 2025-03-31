import torch
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for better accuracy

def detect_humans(frame):
    """Run YOLOv8 human detection on a frame."""
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Class 0 is "person" in COCO dataset
            if cls == 0 and conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Human: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Test on a sample image
frame = cv2.imread("/Users/kossakovm/Desktop/thesis/spotproject/runs/detect/images/detection_20250213-124837.jpg")  # Replace with your image
processed_frame = detect_humans(frame)
cv2.imshow("YOLO Detection", processed_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
