import sys
import time
import numpy as np
import cv2
import bosdyn.client
import bosdyn.client.util
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from ultralytics import YOLO
import signal
import select
import termios
import tty
import math

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
try:
    lease_keep_alive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
except bosdyn.client.lease.ResourceAlreadyClaimedError:
    print("⚠️ Spot's lease is already held. Taking over...")
    lease_client.take()
    lease_keep_alive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
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

def get_spot_image():
    """Captures an RGB image and depth data from Spot's camera."""
    image_requests = [
        image_pb2.ImageRequest(image_source_name="frontleft_depth_in_visual_frame", pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16),
        image_pb2.ImageRequest(image_source_name="frontleft_fisheye_image", quality_percent=50)
    ]
    
    response = image_client.get_image(image_requests)
    if not response or len(response) < 2:
        print("❌ Failed to capture Spot's camera image.")
        return None, None
    
    rgb_img = response[1].shot.image
    depth_img = response[0].shot.image
    
    rgb_frame = cv2.imdecode(np.frombuffer(rgb_img.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    depth_frame = np.frombuffer(depth_img.data, dtype=np.uint16).reshape(rgb_frame.shape[:2])

    if rgb_frame is None or depth_frame is None:
        print("🚨 Image capture failed. Retrying...")
        return None, None
    
    print("📷 Image captured successfully!")
    return rgb_frame, depth_frame

def estimate_distance(bbox, depth_frame):
    """Estimate distance using depth data from Spot's camera."""
    x1, y1, x2, y2 = bbox
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    depth_values = depth_frame[max(0, center_y - 5): min(depth_frame.shape[0], center_y + 5),
                               max(0, center_x - 5): min(depth_frame.shape[1], center_x + 5)]
    
    valid_depths = depth_values[depth_values > 0]
    if valid_depths.size > 0:
        distance = np.median(valid_depths) / 1000.0
        return distance
    else:
        print("⚠️ No valid depth data. Using estimated bounding box method.")
        return (depth_frame.shape[1] / (x2 - x1)) * 1.5

def move_spot_forward(distance=0.2):
    """Moves Spot forward using a trajectory-based command and ensures execution."""
    print(f"🔄 Sending move command: {distance:.2f} meters forward...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    body_tform_goal = math_helpers.SE2Pose(x=distance, y=0, angle=0)
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME, params=RobotCommandBuilder.mobility_params(stair_hint=False))

    print("🚶 Sending movement command to Spot...")
    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)  # Stop previous motion
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)

    # Monitor feedback to ensure the command is executed
    for _ in range(2):
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback

        if mobility_feedback.status == RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("✅ Movement command is being processed.")
        elif mobility_feedback.se2_trajectory_feedback.status == mobility_feedback.se2_trajectory_feedback.STATUS_AT_GOAL:
            print("✅ Spot has arrived at the goal.")
            break
        else:
            print("⚠️ Spot failed to move.")
            break

        time.sleep(0.5)

def move_spot_backward(distance=0.2):
    """Moves Spot backward using a trajectory-based command and ensures execution."""
    print(f"🔄 Sending move command: {distance:.2f} meters backward...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    body_tform_goal = math_helpers.SE2Pose(x=-distance, y=0, angle=0)
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME, params=RobotCommandBuilder.mobility_params(stair_hint=False))

    print("🚶 Sending movement command to Spot...")
    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)

    for _ in range(2):
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback

        if mobility_feedback.status == RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("✅ Movement command is being processed.")
        elif mobility_feedback.se2_trajectory_feedback.status == mobility_feedback.se2_trajectory_feedback.STATUS_AT_GOAL:
            print("✅ Spot has arrived at the goal.")
            break
        else:
            print("⚠️ Spot failed to move.")
            break

        time.sleep(0.5)

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

def cleanup_and_exit():
    """Stops Spot, releases the lease, and exits cleanly."""
    print("🛑 Cleaning up... Spot sitting down and releasing lease.")
    try:
        robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Error while stopping Spot: {e}")

    lease_keep_alive.shutdown()
    sys.exit(0)

def is_key_pressed(key):
    """Check if a key is pressed without blocking execution."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# Main control parameters
horizontal_center_bias = 15  # Positive shifts center right
rotation_threshold_pixels = 20  # Deadband for rotation
target_distance = 1.5  # meters
distance_threshold = 0.3  # meters
max_rotation_angle = 0.26  # radians (~15 degrees)

print("🔄 Starting main loop...")
while True:
    if is_key_pressed('q'):
        print("🛑 'q' pressed. Spot sitting down and stopping...")
        cleanup_and_exit()
        break

    print("📸 Capturing image from Spot...")
    rgb_frame, depth_frame = get_spot_image()
    if rgb_frame is None or depth_frame is None:
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

    # Track the person closest to the center horizontally
    tracked_detection = None
    min_offset = float('inf')

    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, class_id = det[:6]
            if int(class_id) == 0:  # Person
                bbox_center_x = (x1 + x2) / 2
                offset = bbox_center_x - image_center
                if abs(offset) < min_offset:
                    min_offset = abs(offset)
                    tracked_detection = det

    if tracked_detection is not None:
        x1, y1, x2, y2, conf, class_id = tracked_detection[:6]
        distance = estimate_distance([x1, y1, x2, y2], depth_frame)
        print(f"Tracked person at {distance:.2f} meters, horizontal offset: {min_offset:.2f} pixels.")

        # Calculate the offset between the person center and the (possibly biased) image center.
        offset = ((x1 + x2) / 2) - image_center

        # Only adjust if the offset exceeds a threshold.
        if abs(offset) > rotation_threshold_pixels:
            # Compute a proportional rotation
            rotation_angle = -max_rotation_angle * (offset / (rgb_frame.shape[1] / 2))
            print(f"Offset: {offset:.2f} pixels, Rotation Angle: {rotation_angle:.4f} radians")
            rotate_spot(rotation_angle)
            # Skip distance adjustment this frame to allow rotation to complete
            time.sleep(0.5)
            continue

        # Distance adjustment (only when centered)
        print("✅ Centered - Checking distance...")
        if distance > target_distance + distance_threshold:
            move_distance = min(0.5, (distance - target_distance) * 0.5)
            print(f"📏 Too far ({distance:.2f}m). Moving forward {move_distance:.2f}m")
            move_spot_forward(move_distance)
        elif distance < target_distance - distance_threshold:
            move_distance = min(0.5, (target_distance - distance) * 0.5)
            print(f"📏 Too close ({distance:.2f}m). Moving backward {move_distance:.2f}m")
            move_spot_backward(move_distance)
        else:
            print(f"🎯 Ideal distance maintained ({distance:.2f}m)")

    time.sleep(0.3)  # Reduced delay for more responsive tracking