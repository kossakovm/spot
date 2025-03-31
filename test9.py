import sys
import time
import math
import numpy as np
import cv2
import bosdyn.client
import bosdyn.client.util
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b
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

# Initialize SDK and Authenticate
sdk = bosdyn.client.create_standard_sdk('SpotHumanFollow')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

# Check if Spot is Estopped
if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    exit(1)

# Setup Clients
lease_client = robot.ensure_client(LeaseClient.default_service_name)
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)

# Acquire Lease & Power On
with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
    robot.time_sync.wait_for_sync()
    
    if not robot.is_powered_on():
        print("🔌 Powering on Spot...")
        robot.power_on()
        print("✅ Spot is powered on!")
    else:
        print("✅ Spot is already powered on.")
    
    blocking_stand(robot_command_client)
    print("📢 Spot is now standing!")

    # Load YOLO Model
    model = YOLO('yolov8s.pt')
    print("🎯 YOLOv8s Model Loaded!")

    spot_camera_source = "frontleft_fisheye_image"
    
    def get_spot_image():
        image_requests = [image_pb2.ImageRequest(image_source_name=spot_camera_source, quality_percent=50)]
        response = image_client.get_image(image_requests)
        if not response:
            print("❌ Failed to capture Spot's camera image.")
            return None
        img = response[0].shot.image
        return cv2.imdecode(np.frombuffer(img.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    def move_spot_forward(distance=0.5):
        """Moves Spot forward using a trajectory-based command."""
        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        body_tform_goal = math_helpers.SE2Pose(x=distance, y=0, angle=0)
        out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal
        
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME, params=RobotCommandBuilder.mobility_params(stair_hint=False))
        
        print("🚶 Moving Spot forward...")
        cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 10)
        
        while True:
            feedback = robot_command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print("⚠️ Failed to reach the goal.")
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print("✅ Arrived at the goal.")
                return True
            time.sleep(1)
    
    detection_count = 0
    detection_threshold = 3  # Number of frames to confirm human presence
    lost_frame_count = 0
    lost_threshold = 5  # Number of frames before stopping movement
    
    while True:
        frame = get_spot_image()
        if frame is None:
            print("❌ No image received from Spot's camera.")
            break
        
        results = model.predict(frame, imgsz=640, conf=0.3)
        detections = results[0].boxes.data.cpu().numpy()
        person_detections = [d for d in detections if int(d[5]) == 0]
        
        if person_detections:
            detection_count += 1
            lost_frame_count = 0
        else:
            detection_count = max(0, detection_count - 1)
            lost_frame_count += 1
        
        if detection_count >= detection_threshold:
            print("✅ Human detected consistently! Moving Spot forward...")
            move_spot_forward(0.5)
        
        if lost_frame_count >= lost_threshold:
            print("🛑 No human detected for too long! Stopping Spot.")
            robot_command_client.robot_command(RobotCommandBuilder.stop_command())
            detection_count = 0
            lost_frame_count = 0
        
        time.sleep(0.1)  # Frame rate control

# Cleanup
cv2.destroyAllWindows()
print("🔄 Lease returned. Spot is now idle.")
