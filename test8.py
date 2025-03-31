import sys
import time
import math
import numpy as np
import cv2
import bosdyn.client
import bosdyn.client.util
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b)
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

    # **6️⃣ Load YOLO Model**
    model = YOLO('yolov8s.pt')
    print("🎯 YOLOv8s Model Loaded!")

    # **7️⃣ Spot Camera Setup**
    spot_camera_source = "frontleft_fisheye_image"
    
    def get_spot_image():
        """Fetch a frame from Spot's camera."""
        image_requests = [image_pb2.ImageRequest(image_source_name=spot_camera_source, quality_percent=50)]
        response = image_client.get_image(image_requests)
        if not response:
            print("❌ Failed to capture Spot's camera image.")
            return None
        img = response[0].shot.image
        return cv2.imdecode(np.frombuffer(img.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    def move_spot_small_step(robot_command_client, robot_state_client): 
        """Move Spot forward by a small step (e.g., 0.5 meters)."""
        print("🚀 Moving Spot forward by a small step...")
    
        # Define a small forward movement (0.5 meters)
        dx, dy, dyaw = 0.5, 0, 0  # Increased movement distance
    
        # Get the current robot state and transforms
        robot_state = robot_state_client.get_robot_state()
        transforms = robot_state.kinematic_state.transforms_snapshot
    
        # Print robot state for debugging
        print("🤖 Robot State:", robot_state)
    
        # Transform the movement goal into the ODOM frame
        body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
        out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal
    
        # Print transformed goal for debugging
        print("🎯 Transformed Goal (ODOM Frame):", out_tform_goal)
    
        # Build trajectory command
        command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME
        )
    
        # Send the command
        command_id = robot_command_client.robot_command(command)
    
        # Wait for Spot to execute the movement (increased timeout to 20 seconds)
        success = block_for_trajectory_cmd(robot_command_client, command_id, timeout_sec=20)
    
        if success:
            print("✅ Spot successfully moved forward by a small step!")
        else:
            print("❌ Spot failed to move forward.")
            # Log feedback for debugging
            feedback = robot_command_client.robot_command_feedback(command_id)
            print("📄 Movement Feedback:", feedback)
            
    # **8️⃣ Human Detection and Movement Loop**
    frame_rate_limit = 0.5  # Reduce CPU load (2 FPS)
    last_frame_time = 0

    while True:
        if time.time() - last_frame_time < frame_rate_limit:
            continue
        last_frame_time = time.time()
        
        # Fetch an image from Spot's camera
        frame = get_spot_image()
        if frame is None:
            print("❌ No image received from Spot's camera.")
            break
        
        # Perform YOLO object detection
        results = model.predict(frame, imgsz=640, conf=0.3)
        detections = results[0].boxes.data.cpu().numpy()
        person_detections = [d for d in detections if int(d[5]) == 0]  # Filter only "person" class
        
        if person_detections:
            print(f"✅ Detected {len(person_detections)} person(s).")
            
            # Perform a small forward movement when a human is detected
            move_spot_small_step(robot_command_client, robot_state_client)
        
        # Visualize detections
        for detection in person_detections:
            x1, y1, x2, y2, conf, cls = detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Display results
        cv2.imshow("Spot Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔄 Stopping Spot and returning lease...")
            break

# Cleanup
cv2.destroyAllWindows()
print("🔄 Lease returned. Spot is now idle.")