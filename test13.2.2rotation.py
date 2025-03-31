import numpy as np
import cv2
import bosdyn.client
import bosdyn.client.util
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from ultralytics import YOLO  # YOLOv8
import time

# --- Spot Connection Details ---
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# --- Initialize SDK and Authenticate ---
sdk = bosdyn.client.create_standard_sdk('SpotRotationTest')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

# --- Check if Spot is Estopped ---
if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    exit(1)

# --- Setup Clients ---
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)

# --- Acquire Lease & Power On ---
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_keep_alive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
robot.time_sync.wait_for_sync()

if not robot.is_powered_on():
    print("🔌 Powering on Spot...")
    robot.power_on()
    print("✅ Spot is powered on!")
else:
    print("✅ Spot is already powered on.")

# --- Make Spot stand ---
blocking_stand(robot_command_client)
print("📢 Spot is now standing!")

# --- Load YOLO Model ---
model = YOLO('yolov8s.pt')
print("🎯 YOLOv8s Model Loaded!")

# Calibration biases computed from your manual annotation (in pixels)
# These biases shift the computed image center to the effective center for each camera.
left_horizontal_bias = -260
right_horizontal_bias = -260

def get_spot_images():
    """Captures RGB images from Spot's front-left and front-right fisheye cameras."""
    image_requests = [
        image_pb2.ImageRequest(image_source_name="frontleft_fisheye_image", quality_percent=50),
        image_pb2.ImageRequest(image_source_name="frontright_fisheye_image", quality_percent=50)
    ]
    
    try:
        response = image_client.get_image(image_requests)
    except Exception as e:
        print("❌ Failed to capture images:", e)
        return None, None
    
    if len(response) < 2:
        print("❌ Not enough images in the response.")
        return None, None

    # Decode left image
    left_data = response[0].shot.image.data
    left_np = np.frombuffer(left_data, dtype=np.uint8)
    left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
    
    # Decode right image
    right_data = response[1].shot.image.data
    right_np = np.frombuffer(right_data, dtype=np.uint8)
    right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)
    
    if left_frame is None or right_frame is None:
        print("🚨 Failed to decode one or both images.")
        return None, None
    
    return left_frame, right_frame

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

def get_best_detection(detections, image_width, horizontal_bias):
    """
    From a list of detections (for one camera), return the detection with the smallest absolute offset
    from the adjusted center.
    """
    best_det = None
    best_offset = float('inf')
    # Adjusted center for this camera
    cam_center = image_width / 2 + horizontal_bias

    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, class_id = det[:6]
            if int(class_id) == 0:  # Person class
                bbox_center = (x1 + x2) / 2
                offset = bbox_center - cam_center
                if abs(offset) < abs(best_offset):
                    best_offset = offset
                    best_det = det
    return best_det, best_offset

print("🔄 Starting dual-fisheye rotation test loop...")
while True:
    print("📸 Capturing images from Spot...")
    left_frame, right_frame = get_spot_images()
    if left_frame is None or right_frame is None:
        print("❌ No images received. Retrying...")
        time.sleep(1)
        continue

    # Run YOLO detection on both images
    results_left = model.predict(left_frame, imgsz=640, conf=0.3)
    results_right = model.predict(right_frame, imgsz=640, conf=0.3)
    
    left_detections = None
    right_detections = None
    
    if results_left and len(results_left[0].boxes) > 0:
        left_detections = results_left[0].boxes.data.cpu().numpy()
        print(f"✅ LEFT camera detected {len(left_detections)} objects.")
    else:
        print("⚠️ LEFT camera: No detections found.")
    
    if results_right and len(results_right[0].boxes) > 0:
        right_detections = results_right[0].boxes.data.cpu().numpy()
        print(f"✅ RIGHT camera detected {len(right_detections)} objects.")
    else:
        print("⚠️ RIGHT camera: No detections found.")
    
    # Compute adjusted centers for each camera
    left_center = left_frame.shape[1] / 2 + left_horizontal_bias
    right_center = right_frame.shape[1] / 2 + right_horizontal_bias
    print(f"Left Camera Center: {left_center:.2f} pixels, Right Camera Center: {right_center:.2f} pixels")
    
    # Get best detection from each camera if detections exist
    if left_detections is not None and left_detections.shape[0] > 0:
        best_left, offset_left = get_best_detection(left_detections, left_frame.shape[1], left_horizontal_bias)
    else:
        best_left, offset_left = None, None

    if right_detections is not None and right_detections.shape[0] > 0:
        best_right, offset_right = get_best_detection(right_detections, right_frame.shape[1], right_horizontal_bias)
    else:
        best_right, offset_right = None, None
    
    # Decide which detection to use:
    # If both are available, use the one with the smaller absolute offset.
    # Otherwise, if only one is available:
    #   - If only left camera detection exists, use it.
    #   - If only right camera detection exists, invert its offset to account for the camera's mirrored view.
    tracked_offset = None
    if best_left is not None and best_right is not None:
        tracked_offset = offset_left if abs(offset_left) < abs(offset_right) else offset_right
        print(f"Both cameras detected a person. Chosen offset: {tracked_offset:.2f} pixels")
    elif best_left is not None:
        tracked_offset = offset_left
        print(f"Using LEFT camera detection. Offset: {tracked_offset:.2f} pixels")
    elif best_right is not None:
        # Invert the right camera offset for proper rotation
        tracked_offset = -offset_right
        print(f"Using RIGHT camera detection (inverted). Offset: {tracked_offset:.2f} pixels")
    else:
        print("⚠️ No person detected in either camera.")
    
    # If a detection is found, compute and perform rotation
    if tracked_offset is not None:
        rotation_threshold_pixels = 10  # Deadband threshold in pixels.
        if abs(tracked_offset) > rotation_threshold_pixels:
            max_rotation_angle = 0.785  # 15 degrees in radians.
            # Scale the rotation angle proportionally to the offset relative to half the image width
            rotation_angle = -max_rotation_angle * (tracked_offset / (left_frame.shape[1] / 2))
            print(f"Final Computation - Offset: {tracked_offset:.2f} pixels, Rotation Angle: {rotation_angle:.4f} radians")
            rotate_spot(rotation_angle)
        else:
            print("✅ Person is centered. No rotation needed.")
    else:
        print("⚠️ No valid detection available for rotation command.")

    time.sleep(1)
