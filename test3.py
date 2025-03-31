import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for better accuracy

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # Keeps tracking for 30 frames

# Open video capture (0 for webcam, or replace with "video.mp4")
cap = cv2.VideoCapture(1)  # Change to 'video.mp4' if using a file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Convert detections to DeepSORT format
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0].item())  # Confidence score
            cls = int(box.cls[0].item())  # Class ID

            # Track only humans (Class 0 in COCO dataset)
            if cls == 0 and conf > 0.5:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
