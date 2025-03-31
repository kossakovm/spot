import cv2
import time
import bosdyn.client
import bosdyn.client.util
import numpy as np
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, block_for_trajectory_cmd
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.power import PowerClient
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api.power_pb2 import PowerCommandRequest
from bosdyn.client.time_sync import TimeSyncClient
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from ultralytics import YOLO  # YOLOv8

# Step 1: Initialize Spot's Connection
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# Create SDK instance and authenticate
sdk = bosdyn.client.create_standard_sdk('SpotHumanFollow')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

# Step 2: Perform Time Synchronization BEFORE Commanding Spot
time_sync_client = robot.ensure_client(TimeSyncClient.default_service_name)
robot.time_sync.wait_for_sync()
print("⏳ Time Sync Completed!")

# Step 3: Acquire Lease BEFORE Powering On
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease = lease_client.take()
lease_keepalive = LeaseKeepAlive(lease_client)
print("🔑 Lease Acquired!")

# Step 4: Ensure Spot is Powered On
power_client = robot.ensure_client(PowerClient.default_service_name)
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

robot_state = robot_state_client.get_robot_state()
if robot_state.power_state.motor_power_state != 2:  # MOTOR_POWER_STATE_ON
    print("⚡ Spot is not powered on. Powering on now...")
    power_client.power_command(PowerCommandRequest.REQUEST_ON)
    time.sleep(5)  # Wait for power-up
    print("✅ Spot is now powered on!")

# Step 5: Command Client Setup
command_client = robot.ensure_client(RobotCommandClient.default_service_name)

# Step 6: Handle Behavior Faults (Clears Errors)
robot_state = robot_state_client.get_robot_state()
if robot_state.behavior_fault_state.faults:
    print("❌ Spot has behavior faults. Attempting to clear...")
    command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
    time.sleep(3)
    command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
    time.sleep(5)

# Step 7: Make Spot Stand
print("📢 Making Spot stand up...")
stand_command = RobotCommandBuilder.synchro_stand_command()
command_client.robot_command(stand_command)
time.sleep(3)
print("✅ Spot is now standing!")

# Step 8: Load YOLOv8 Model
model = YOLO('yolov8n.pt')  # Using YOLOv8
print("🎯 YOLOv8 Model Loaded!")

# Step 9: Use Spot's Front Camera Instead of Laptop Camera
image_client = robot.ensure_client(ImageClient.default_service_name)
spot_camera_source = "frontleft_fisheye_image"  # Change if needed

def get_spot_image():
    """Fetch a frame from Spot's camera in reduced resolution to reduce lag."""
    image_requests = [image_pb2.ImageRequest(image_source_name=spot_camera_source, quality_percent=50)]  # Reduced quality
    response = image_client.get_image(image_requests)
    
    if not response:
        print("❌ Failed to capture Spot's camera image.")
        return None
    
    img = response[0].shot.image
    np_arr = cv2.imdecode(np.frombuffer(img.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return np_arr

def move_spot(direction, command_client, lease_client, distance=1.0):
    """Move Spot in a specified direction using trajectory commands."""
    print(f"🚀 Moving Spot {direction} for {distance} meters...")

    # Ensure we have an active lease
    lease = lease_client.take()

    # Define movement direction
    dx, dy = 0, 0  
    if direction == "forward":
        dx = distance
    elif direction == "backward":
        dx = -distance
    elif direction == "left":
        dy = distance
    elif direction == "right":
        dy = -distance
    else:
        print("⚠️ Invalid direction! Use 'forward', 'backward', 'left', or 'right'.")
        return

    # Build trajectory command
    command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=dx, goal_y=dy, goal_heading=0, frame_name=ODOM_FRAME_NAME
    )

    # Send the command
    command_id = command_client.robot_command(command)

    # Wait for Spot to execute the movement
    success = block_for_trajectory_cmd(command_client, command_id, timeout_sec=10)
    if success:
        print(f"✅ Spot successfully moved {direction}!")
    else:
        print(f"❌ Spot failed to move {direction}. Retrying...")
        command_client.robot_command(command)

# Step 10: Human Detection and Spot Movement Loop
frame_rate_limit = 0.5  # Reduce CPU load (2 FPS)
last_frame_time = 0

while True:
    if time.time() - last_frame_time < frame_rate_limit:
        continue  # Enforce frame rate limit
    last_frame_time = time.time()

    frame = get_spot_image()
    if frame is None:
        print("❌ No image received from Spot's camera.")
        break

    # Perform YOLO object detection with reduced resolution
    results = model.predict(frame, imgsz=320)  # Lower resolution for faster inference

    # Get detections
    detections = results[0].boxes.data.cpu().numpy()  # Using YOLOv8 format
    person_detections = [d for d in detections if int(d[5]) == 0]  # Filter only "person" class

    if person_detections:
        print(f"✅ Detected {len(person_detections)} person(s).")

        # Get the closest person (based on bounding box width)
        closest_person = max(person_detections, key=lambda d: d[2] - d[0])  # Widest bounding box

        # Get center x of the detected person
        person_center_x = (closest_person[0] + closest_person[2]) / 2
        frame_center_x = frame.shape[1] / 2  # Center of the camera feed

        # Debug prints
        print(f"🎯 Person X: {person_center_x}, Frame Center X: {frame_center_x}")

        # Visualize the movement logic
        cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), frame.shape[0]), (0, 0, 255), 2)
        cv2.line(frame, (int(frame_center_x - 100), 0), (int(frame_center_x - 100), frame.shape[0]), (255, 0, 0), 2)
        cv2.line(frame, (int(frame_center_x + 100), 0), (int(frame_center_x + 100), frame.shape[0]), (255, 0, 0), 2)

        # Determine movement direction
        if person_center_x < frame_center_x - 100:
            print("Moving left")
            move_spot("left", command_client, lease_client)
        elif person_center_x > frame_center_x + 100:
            print("Moving right")
            move_spot("right", command_client, lease_client)
        else:
            print("Moving forward")
            move_spot("forward", command_client, lease_client)

    # Visualize detections
    for detection in person_detections:
        x1, y1, x2, y2, conf, cls = detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display results
    cv2.imshow("Spot Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
lease_client.return_lease(lease)
print("🔄 Lease returned. Spot is now idle.")