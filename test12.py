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
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracking
import signal
import select
import termios
import tty

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

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30)
print("🔍 DeepSORT Tracker Initialized!")

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

def move_spot_forward(distance=0.1):
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
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)  # Reduced to 1 second

    # Monitor feedback to ensure the command is executed
    for _ in range(2):  # Check status for up to 1 second
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

def move_spot_backward(distance=0.1):
    """Moves Spot backward using a trajectory-based command and ensures execution."""
    print(f"🔄 Sending move command: {distance:.2f} meters backward...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    body_tform_goal = math_helpers.SE2Pose(x=-distance, y=0, angle=0)  # Negative distance for backward movement
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME, params=RobotCommandBuilder.mobility_params(stair_hint=False))

    print("🚶 Sending movement command to Spot...")
    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)  # Stop previous motion
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)  # Reduced to 1 second

    # Monitor feedback to ensure the command is executed
    for _ in range(2):  # Check status for up to 1 second
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

def cleanup_and_exit():
    """Stops Spot, releases the lease, and exits cleanly."""
    print("🛑 Cleaning up... Spot sitting down and releasing lease.")
    try:
        robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
        time.sleep(2)  # Give it time to sit
    except Exception as e:
        print(f"⚠️ Error while stopping Spot: {e}")

    lease_keep_alive.shutdown()  # Release the lease
    sys.exit(0)

# Handle Ctrl+C to clean up properly
signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_exit())

def is_key_pressed(key):
    """Check if a key is pressed without blocking execution."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

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
        continue  # Retry instead of exiting
    
    print("🧠 Running YOLO detection...")
    results = model.predict(rgb_frame, imgsz=640, conf=0.3)
    if not results or len(results[0].boxes) == 0:
        print("⚠️ No detections found.")
        time.sleep(1)  # Wait before retrying
        continue
    print(f"✅ Detected {len(results[0].boxes)} objects.")
    detections = results[0].boxes.data.cpu().numpy()
    
    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, class_id = det[:6]
            if int(class_id) == 0:  # Class ID 0 is for "person"
                distance = estimate_distance([x1, y1, x2, y2], depth_frame)
                print(f"Person detected at {distance:.2f} meters.")
                
                target_distance = 0.5  # Target distance to maintain
                distance_threshold = 0.1  # Allowed deviation from target distance

                if distance > target_distance + distance_threshold:
                    # Move forward proportionally to the distance
                    move_distance = min(0.5, distance - target_distance)  # Move up to 0.5 meters at a time
                    print(f"📏 Person is too far ({distance:.2f}m). Moving forward {move_distance:.2f} meters...")
                    move_spot_forward(move_distance)
                elif distance < target_distance - distance_threshold:
                    # Move backward proportionally to the distance
                    move_distance = min(0.5, target_distance - distance)  # Move up to 0.5 meters at a time
                    print(f"📏 Person is too close ({distance:.2f}m). Moving backward {move_distance:.2f} meters...")
                    move_spot_backward(move_distance)
                else:
                    print(f"✅ Keeping distance ({distance:.2f}m). No movement needed.")

    time.sleep(1)  # Wait 1 second before the next iteration