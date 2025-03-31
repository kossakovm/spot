import numpy as np
import cv2
import bosdyn.client
import bosdyn.client.util
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive  # Import LeaseClient and LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand  # Import blocking_stand
from bosdyn.client.robot_state import RobotStateClient  # Import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus  # Import RobotCommandFeedbackStatus
from ultralytics import YOLO  # YOLOv8
import time

# Spot Connection Details
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# Initialize SDK and Authenticate
sdk = bosdyn.client.create_standard_sdk('SpotRotationTest')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

# Check if Spot is Estopped
if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    exit(1)

# Setup Clients
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)

# Acquire Lease & Power On
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_keep_alive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
robot.time_sync.wait_for_sync()

if not robot.is_powered_on():
    print("🔌 Powering on Spot...")
    robot.power_on()
    print("✅ Spot is powered on!")
else:
    print("✅ Spot is already powered on.")

# Make Spot stand
blocking_stand(robot_command_client)
print("📢 Spot is now standing!")

# Load YOLO Model
model = YOLO('yolov8s.pt')
print("🎯 YOLOv8s Model Loaded!")

def get_spot_image():
    """Captures an RGB image from Spot's camera."""
    image_requests = [
        image_pb2.ImageRequest(image_source_name="frontleft_fisheye_image", quality_percent=50)
    ]
    
    response = image_client.get_image(image_requests)
    if not response or len(response) < 1:
        print("❌ Failed to capture Spot's camera image.")
        return None
    
    rgb_img = response[0].shot.image
    rgb_frame = cv2.imdecode(np.frombuffer(rgb_img.data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if rgb_frame is None:
        print("🚨 Image capture failed. Retrying...")
        return None
    
    print("📷 Image captured successfully!")
    return rgb_frame

def rotate_spot(angle):
    """
    Rotates Spot in place by the specified angle (in radians).
    Positive angles turn left; negative angles turn right.
    """
    print(f"🔄 Rotating Spot by {angle:.2f} radians...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    current_pose = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    new_heading = current_pose.angle + angle

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=current_pose.x,
        goal_y=current_pose.y,
        goal_heading=new_heading,
        frame_name=ODOM_FRAME_NAME,
        params=RobotCommandBuilder.mobility_params(stair_hint=False)
    )

    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)

    for _ in range(2):
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status == RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("✅ Rotation command is being processed.")
        elif mobility_feedback.se2_trajectory_feedback.status == mobility_feedback.se2_trajectory_feedback.STATUS_AT_GOAL:
            print("✅ Spot has completed rotation.")
            break
        else:
            print("⚠️ Spot failed to rotate.")
            break
        time.sleep(0.5)

# Define a calibration bias in pixels (adjust this value based on your testing)
horizontal_center_bias = -260  # Positive value shifts the computed center to the right

print("🔄 Starting rotation test loop...")
while True:
    print("📸 Capturing image from Spot...")
    rgb_frame = get_spot_image()
    if rgb_frame is None:
        print("❌ No image received from Spot's camera. Retrying...")
        time.sleep(1)
        continue
    
    print("🧠 Running YOLO detection...")
    results = model.predict(rgb_frame, imgsz=640, conf=0.3)
    if not results or len(results[0].boxes) == 0:
        print("⚠️ No detections found.")
        time.sleep(1)
        continue
    print(f"✅ Detected {len(results[0].boxes)} objects.")
    detections = results[0].boxes.data.cpu().numpy()

    # Calculate the adjusted center of the image
    image_center = rgb_frame.shape[1] / 2 + horizontal_center_bias
    print(f"Image Center: {image_center:.2f} pixels")

    # Track the person closest to the center horizontally
    tracked_detection = None
    min_abs_offset = float('inf')

    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, class_id = det[:6]
            if int(class_id) == 0:  # Person
                bbox_center_x = (x1 + x2) / 2
                offset = bbox_center_x - image_center
                abs_offset = abs(offset)
                print(f"Bounding Box Center: {bbox_center_x:.2f} pixels, Offset: {offset:.2f} pixels")
                if abs_offset < min_abs_offset:
                    min_abs_offset = abs_offset
                    tracked_detection = det
                    tracked_offset = offset  # Store the actual offset (not absolute)

    if tracked_detection is not None:
        x1, y1, x2, y2, conf, class_id = tracked_detection[:6]
        print(f"Tracked person, horizontal offset: {tracked_offset:.2f} pixels.")

        # Only adjust if the offset exceeds a threshold.
        rotation_threshold_pixels = 10  # Reduced deadband in pixels.
        if abs(tracked_offset) > rotation_threshold_pixels:
            # Set a maximum rotation angle (e.g., 15 degrees in radians).
            max_rotation_angle = 0.2618  # 15 degrees in radians.
        
            # Compute a proportional rotation, scaling the offset relative to half the image width.
            # This gives a value between -max_rotation_angle and +max_rotation_angle.
            rotation_angle = -max_rotation_angle * (tracked_offset / (rgb_frame.shape[1] / 2))
        
            # Log the offset and rotation angle for debugging
            print(f"Offset: {tracked_offset:.2f} pixels, Rotation Angle: {rotation_angle:.4f} radians")
        
            # Rotate Spot
            rotate_spot(rotation_angle)
        else:
            print("✅ Person is centered. No rotation needed.")
    else:
        print("⚠️ No person detected.")

    time.sleep(1)