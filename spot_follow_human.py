import sys
import os
import time
import cv2
import numpy as np
import tempfile

# Ensure YOLOv5 and DeepSORT are accessible
sys.path.append(os.path.abspath("/Users/kossakovm/Desktop/thesis/spotproject/yolov5"))
sys.path.append(os.path.abspath("/Users/kossakovm/Desktop/thesis/spotproject/deep_sort"))

# Import required modules
from yolov5.detect import run
from deep_sort_realtime.deepsort_tracker import DeepSort
from bosdyn.client import create_standard_sdk
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient
from bosdyn.client.power import PowerClient

# Spot Robot Credentials
ROBOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# Initialize Spot SDK
sdk = create_standard_sdk('PersonFollowing')
robot = sdk.create_robot(ROBOT_IP)
robot.authenticate(USERNAME, PASSWORD)
robot.time_sync.wait_for_sync()

# Forcefully take lease before acquiring it
lease_client = robot.ensure_client(LeaseClient.default_service_name)
try:
    lease = lease_client.take()  # First, forcefully take the lease if it's claimed
    print("⚠️ Lease was already claimed. Successfully took control!")
except bosdyn.client.lease.ResourceNotAvailableError:
    lease = lease_client.acquire()  # Only acquire if no one is using it
    print("✅ No other client was using the lease.")

lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(lease_client)
print("✅ Lease acquired successfully!")

# Power on Spot if needed
if not robot.is_powered_on():
    print("⚡ Spot is not powered on. Turning it on...")
    robot.power_on(timeout_sec=20)  # Correct power-on method
    robot.logger.info("Waiting for Spot to power on...")
    time.sleep(5)

print("✅ Spot is powered on and ready to move!")

# Access Spot's camera feed
image_client = robot.ensure_client(ImageClient.default_service_name)

# Initialize DeepSORT tracker
deepsort = DeepSort(max_age=5)

# Load YOLOv5 model
weights = "/Users/kossakovm/Desktop/thesis/spotproject/yolov5/yolov5s.pt"

# Create a directory for saved images
os.makedirs("runs/detect/images", exist_ok=True)

# Frame rate control
frame_rate = 5  # Process 5 frames per second
frame_interval = int(1000 / frame_rate)  # Time in milliseconds

# Ensure Spot is standing before movement
command_client = robot.ensure_client(RobotCommandClient.default_service_name)
print("🦿 Making Spot stand before movement...")
blocking_stand(command_client, timeout_sec=10)

# Real-time detection and tracking loop
while True:
    # Capture image from Spot's camera
    image_response = image_client.get_image_from_sources(['frontleft_fisheye_image'])
    if not image_response:
        print("❌ Failed to retrieve image.")
        continue

    image = image_response[0].shot.image
    np_arr = np.frombuffer(image.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Debug: Check frame dimensions and save it
    print(f"Frame shape: {frame.shape}")
    cv2.imwrite("debug_frame.jpg", frame)

    # Save the frame to a unique filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    image_filename = f"runs/detect/images/detection_{timestamp}.jpg"
    cv2.imwrite(image_filename, frame)
    print(f"📸 Image saved: {image_filename}")

    # Run YOLOv5 detection on the saved image file
    results = run(weights=weights, source=image_filename, imgsz=640, conf_thres=0.3, device='cpu')

    # Ensure YOLOv5 returned valid results
    detections = []
    if results is None or not hasattr(results, 'pred'):
        print("⚠️ Warning: No detections found!")
    else:
        for det in results.pred[0]:  # Accessing predictions correctly
            *xyxy, conf, cls = det.cpu().numpy()
            cls = int(cls)  # Ensure class ID is an integer
            if cls == 0:  # Class 0 is "person"
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append([x1, y1, x2, y2, conf])

        if detections:
            print(f"✅ Found {len(detections)} person(s)!")
        else:
            print("⚠️ No person detected, but other objects were found.")

    # Update DeepSORT tracker
    tracks = deepsort.update_tracks(detections, frame=frame)

    # Draw tracked objects and calculate person's position
    frame_center_x = frame.shape[1] / 2
    frame_center_y = frame.shape[0] / 2
    person_detected = False

    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Calculate person's center
        person_center_x = (x1 + x2) / 2
        person_center_y = (y1 + y2) / 2

        # Calculate offset from frame center
        offset_x = person_center_x - frame_center_x
        offset_y = person_center_y - frame_center_y

        person_detected = True

    # Define command duration (Prevents ExpiredError)
    command_duration = 5.0  # 5 seconds for each movement command

    if person_detected:
        if offset_x < -50:  # Person is to the left
            command = RobotCommandBuilder.synchro_velocity_command(v_x=0, v_y=0, v_rot=0.5)  # Turn left
        elif offset_x > 50:  # Person is to the right
            command = RobotCommandBuilder.synchro_velocity_command(v_x=0, v_y=0, v_rot=-0.5)  # Turn right
        else:  # Person is centered
            command = RobotCommandBuilder.synchro_velocity_command(v_x=0.5, v_y=0, v_rot=0)  # Move forward
    else:
        # Stop if no person is detected
        command = RobotCommandBuilder.synchro_velocity_command(v_x=0, v_y=0, v_rot=0)

    # Send command to Spot with duration to avoid expiration
    command_client.robot_command(command, end_time_secs=time.time() + command_duration)
    print("✅ Movement command sent successfully!")

    # Display the frame with bounding boxes
    cv2.imshow('Real-Time Person Following', frame)

    # Exit on 'q' key press
    if cv2.waitKey(frame_interval) & 0xFF == ord('q'):
        break

# Release lease and cleanup
lease_keepalive.shutdown()
lease_client.return_lease(lease)
cv2.destroyAllWindows()
