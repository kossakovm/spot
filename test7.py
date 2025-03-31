import sys
import time
import math
import numpy as np
import cv2
import bosdyn.client
import bosdyn.client.util
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand, block_for_trajectory_cmd)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.power import PowerClient
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from ultralytics import YOLO  # YOLOv8

# Spot Connection Details
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# **1️⃣ Initialize SDK and Authenticate**
sdk = bosdyn.client.create_standard_sdk('SpotHumanFollow')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

# **2️⃣ Check if Spot is Estopped**
if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    exit(1)

# **3️⃣ Setup Clients**
lease_client = robot.ensure_client(LeaseClient.default_service_name)
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)

# **4️⃣ Acquire Lease & Power On**
with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
    robot.time_sync.wait_for_sync()
    
    # Check if the robot is powered on
    if not robot.is_powered_on():
        print("🔌 Powering on Spot...")
        robot.power_on()
        print("✅ Spot is powered on!")
    else:
        print("✅ Spot is already powered on.")
    
    # **5️⃣ Make Spot Stand**
    blocking_stand(robot_command_client)
    print("📢 Spot is now standing!")

    # **6️⃣ Define `move_spot` Function**
    def move_spot(dx, dy, dyaw, robot_command_client, robot_state_client):
        """Moves Spot in a specified direction using frame trajectory commands."""
        print(f"🚀 Moving Spot: dx={dx}, dy={dy}, dyaw={dyaw}...")
        
        # Check robot state before moving
        state = robot_state_client.get_robot_state()
        print("🤖 Robot State:", state)
        
        # Print transforms snapshot for debugging
        transforms = state.kinematic_state.transforms_snapshot
        print("📐 Transforms Snapshot:", transforms)
        
        # Define Movement Directions
        command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=dx, goal_y=dy, goal_heading=math.radians(dyaw), frame_name=ODOM_FRAME_NAME
        )
        command_id = robot_command_client.robot_command(command)
        
        # ✅ Ensure Spot waits until movement is executed (increased timeout to 10 seconds)
        success = block_for_trajectory_cmd(robot_command_client, command_id, timeout_sec=10)
        
        if success:
            print(f"✅ Spot successfully moved: dx={dx}, dy={dy}, dyaw={dyaw}")
        else:
            print("❌ Spot failed to execute movement.")

    # **7️⃣ Load YOLO Model**
    model = YOLO('yolov8s.pt')
    print("🎯 YOLOv8s Model Loaded!")

    # **8️⃣ Spot Camera Setup**
    spot_camera_source = "frontleft_fisheye_image"
    
    def get_spot_image():
        image_requests = [image_pb2.ImageRequest(image_source_name=spot_camera_source, quality_percent=50)]
        response = image_client.get_image(image_requests)
        if not response:
            print("❌ Failed to capture Spot's camera image.")
            return None
        img = response[0].shot.image
        return cv2.imdecode(np.frombuffer(img.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    frame_rate_limit = 0.5
    last_frame_time = 0

    while True:
        if time.time() - last_frame_time < frame_rate_limit:
            continue
        last_frame_time = time.time()
        
        frame = get_spot_image()
        if frame is None:
            print("❌ No image received from Spot's camera.")
            break
        
        results = model.predict(frame, imgsz=640, conf=0.3)
        detections = results[0].boxes.data.cpu().numpy()
        person_detections = [d for d in detections if int(d[5]) == 0]
        
        if person_detections:
            print(f"✅ Detected {len(person_detections)} person(s).")
            # Test with smaller movements first
            move_spot(0.05, 0, 1, robot_command_client, robot_state_client)  # Very small forward step + slight rotation
        
        cv2.imshow("Spot Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔄 Stopping Spot and returning lease...")
            break

cv2.destroyAllWindows()