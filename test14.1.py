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
import sys
import signal
import select
import termios
import tty
import math
from bosdyn.api import geometry_pb2

# ----------------- Spot Connection Details -----------------
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# ----------------- Initialize SDK and Authenticate -----------------
sdk = bosdyn.client.create_standard_sdk('SpotCombinedControl')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    exit(1)

# ----------------- Setup Clients -----------------
lease_client = robot.ensure_client(LeaseClient.default_service_name)
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)

# Acquire lease & power on
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

# ----------------- Load YOLO Model -----------------
model = YOLO('yolov8s.pt')
print("🎯 YOLOv8s Model Loaded!")

# ----------------- Calibration Biases -----------------
# For rotation (dual-fisheye cameras)
left_horizontal_bias = -260
right_horizontal_bias = -260

# For movement (single left camera) – adjust this bias based on testing.
horizontal_center_bias = -15

# Movement parameters
target_distance = 0.5   # in meters
distance_threshold = 0.1

# ----------------- Helper Functions -----------------
def get_spot_images():
    """
    Captures images from Spot’s left fisheye, right fisheye, and depth camera.
    Returns:
      left_frame: RGB image from frontleft_fisheye_image.
      right_frame: RGB image from frontright_fisheye_image.
      depth_frame: Depth image from frontleft_depth_in_visual_frame.
    """
    image_requests = [
        image_pb2.ImageRequest(image_source_name="frontleft_fisheye_image", quality_percent=50),
        image_pb2.ImageRequest(image_source_name="frontright_fisheye_image", quality_percent=50),
        image_pb2.ImageRequest(image_source_name="frontleft_depth_in_visual_frame", pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16)
    ]
    
    try:
        response = image_client.get_image(image_requests)
    except Exception as e:
        print("❌ Failed to capture images:", e)
        return None, None, None
    
    if len(response) < 3:
        print("❌ Not enough images received.")
        return None, None, None

    # Decode left fisheye image
    left_data = response[0].shot.image.data
    left_np = np.frombuffer(left_data, dtype=np.uint8)
    left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
    
    # Decode right fisheye image
    right_data = response[1].shot.image.data
    right_np = np.frombuffer(right_data, dtype=np.uint8)
    right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)
    
    # Process depth image; use left_frame shape for reshaping if available.
    depth_img = response[2].shot.image
    if left_frame is not None:
        depth_frame = np.frombuffer(depth_img.data, dtype=np.uint16).reshape(left_frame.shape[:2])
    else:
        depth_frame = None

    if left_frame is None or right_frame is None or depth_frame is None:
        print("🚨 Failed to decode one or more images.")
        return None, None, None

    return left_frame, right_frame, depth_frame

def get_best_detection(detections, image_width, horizontal_bias):
    """
    For rotation: From a list of detections, return the detection with the smallest
    absolute offset from the adjusted camera center.
    """
    best_det = None
    best_offset = float('inf')
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

def estimate_distance(bbox, depth_frame):
    """Estimate distance using depth data from Spot's camera."""
    x1, y1, x2, y2 = bbox
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    depth_values = depth_frame[max(0, center_y - 5): min(depth_frame.shape[0], center_y + 5),
                               max(0, center_x - 5): min(depth_frame.shape[1], center_x + 5)]
    
    valid_depths = depth_values[depth_values > 0]
    if valid_depths.size > 0:
        distance = np.median(valid_depths) / 1000.0  # convert mm to meters
        return distance
    else:
        print("⚠️ No valid depth data. Using fallback estimation.")
        return (depth_frame.shape[1] / (x2 - x1)) * 1.5

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

def move_spot_forward(distance=0.2):
    """Moves Spot forward using a trajectory-based command."""
    print(f"🔄 Moving forward {distance:.2f} meters...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    body_tform_goal = math_helpers.SE2Pose(x=distance, y=0, angle=0)
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME, params=RobotCommandBuilder.mobility_params(stair_hint=False))

    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)

    for _ in range(2):
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback

        if mobility_feedback.status == RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("✅ Forward movement processing.")
        elif mobility_feedback.se2_trajectory_feedback.status == mobility_feedback.se2_trajectory_feedback.STATUS_AT_GOAL:
            print("✅ Reached forward goal.")
            break
        else:
            print("⚠️ Forward movement failed.")
            break
        time.sleep(0.5)

def move_spot_backward(distance=0.2):
    """Moves Spot backward using a trajectory-based command."""
    print(f"🔄 Moving backward {distance:.2f} meters...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    body_tform_goal = math_helpers.SE2Pose(x=-distance, y=0, angle=0)
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME, params=RobotCommandBuilder.mobility_params(stair_hint=False))

    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)

    for _ in range(2):
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback

        if mobility_feedback.status == RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("✅ Backward movement processing.")
        elif mobility_feedback.se2_trajectory_feedback.status == mobility_feedback.se2_trajectory_feedback.STATUS_AT_GOAL:
            print("✅ Reached backward goal.")
            break
        else:
            print("⚠️ Backward movement failed.")
            break
        time.sleep(0.5)

def quat_from_euler(roll, pitch, yaw):
    """Convert Euler angles to a quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return math_helpers.Quat(w=w, x=x, y=y, z=z)

if not hasattr(math_helpers.Quat, 'to_quaternion'):
    def to_quaternion(self):
        return geometry_pb2.Quaternion(w=self.w, x=self.x, y=self.y, z=self.z)
    math_helpers.Quat.to_quaternion = to_quaternion

def adjust_spot_posture(height=0.1, pitch=0.2):
    """Adjust Spot's body posture (height and pitch)."""
    print(f"🔄 Adjusting posture: height={height:.2f}m, pitch={pitch:.2f}rad...")
    footprint_R_body = quat_from_euler(roll=0.0, pitch=pitch, yaw=0.0)
    robot_cmd = RobotCommandBuilder.synchro_stand_command(
        body_height=height,
        footprint_R_body=footprint_R_body
    )
    robot_command_client.robot_command(robot_cmd)

def cleanup_and_exit():
    """Stops Spot, releases lease, and exits cleanly."""
    print("🛑 Cleaning up... Spot sitting and releasing lease.")
    try:
        robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Error while stopping Spot: {e}")
    lease_keep_alive.shutdown()
    sys.exit(0)

def is_key_pressed(key):
    """Check if a key is pressed (non-blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ----------------- Main Loop -----------------
print("🔄 Starting combined control loop...")
while True:
    # Exit if 'q' is pressed.
    if is_key_pressed('q'):
        print("🛑 'q' pressed. Exiting...")
        cleanup_and_exit()
        break

    # Capture images: left and right fisheye and depth.
    left_frame, right_frame, depth_frame = get_spot_images()
    if left_frame is None or right_frame is None or depth_frame is None:
        print("❌ Image capture failed. Retrying...")
        time.sleep(1)
        continue

    # ----- Rotation Detection using Dual-Fisheye -----
    # Run YOLO on both left and right fisheye images.
    results_left = model.predict(left_frame, imgsz=640, conf=0.3)
    results_right = model.predict(right_frame, imgsz=640, conf=0.3)
    
    left_detections = None
    right_detections = None

    if results_left and len(results_left[0].boxes) > 0:
        left_detections = results_left[0].boxes.data.cpu().numpy()
        print(f"✅ LEFT camera: {len(left_detections)} objects detected.")
    else:
        print("⚠️ LEFT camera: No detections.")

    if results_right and len(results_right[0].boxes) > 0:
        right_detections = results_right[0].boxes.data.cpu().numpy()
        print(f"✅ RIGHT camera: {len(right_detections)} objects detected.")
    else:
        print("⚠️ RIGHT camera: No detections.")

    # Compute adjusted centers and best detections for rotation.
    tracked_offset = None
    if left_detections is not None or right_detections is not None:
        best_left, offset_left = (get_best_detection(left_detections, left_frame.shape[1], left_horizontal_bias)
                                   if left_detections is not None else (None, None))
        best_right, offset_right = (get_best_detection(right_detections, right_frame.shape[1], right_horizontal_bias)
                                     if right_detections is not None else (None, None))
        if best_left is not None and best_right is not None:
            tracked_offset = offset_left if abs(offset_left) < abs(offset_right) else offset_right
            print(f"Both cameras detected a person. Chosen offset: {tracked_offset:.2f} pixels")
        elif best_left is not None:
            tracked_offset = offset_left
            print(f"Using LEFT camera detection for rotation. Offset: {tracked_offset:.2f} pixels")
        elif best_right is not None:
            # Invert right camera offset for proper rotation.
            tracked_offset = -offset_right
            print(f"Using RIGHT camera detection (inverted) for rotation. Offset: {tracked_offset:.2f} pixels")
        else:
            print("⚠️ No person detected for rotation.")
    else:
        print("⚠️ No detections on either camera for rotation.")

    # If the horizontal offset exceeds the deadband, rotate.
    rotation_threshold_pixels = 10  # pixels deadband.
    if tracked_offset is not None and abs(tracked_offset) > rotation_threshold_pixels:
        max_rotation_angle = 0.785  # 15 degrees in radians.
        # Scale rotation angle proportionally (negative sign for correct turning direction).
        rotation_angle = -max_rotation_angle * (tracked_offset / (left_frame.shape[1] / 2))
        print(f"Rotation: Offset={tracked_offset:.2f} px, Rotation Angle={rotation_angle:.4f} rad")
        rotate_spot(rotation_angle)
    else:
        print("✅ Person is horizontally centered; no rotation needed.")

    # ----- Movement Detection using Left Camera + Depth -----
    # For movement, use detection from left image with its own bias.
    movement_detections = left_detections  # reuse results from left image.
    tracked_detection = None
    min_offset = float('inf')
    if movement_detections is not None:
        image_center = left_frame.shape[1] / 2 + horizontal_center_bias
        for det in movement_detections:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, class_id = det[:6]
                if int(class_id) == 0:  # Person
                    bbox_center_x = (x1 + x2) / 2
                    offset = abs(bbox_center_x - image_center)
                    if offset < min_offset:
                        min_offset = offset
                        tracked_detection = det
    else:
        print("⚠️ No detections from left camera for movement.")

    if tracked_detection is not None:
        x1, y1, x2, y2, conf, class_id = tracked_detection[:6]
        distance = estimate_distance([x1, y1, x2, y2], depth_frame)
        print(f"Movement: Tracked person at {distance:.2f} m (offset: {min_offset:.2f} px)")

        # Adjust rotation for movement if needed (could be redundant with dual-fisheye rotation).
        if abs(min_offset) > rotation_threshold_pixels:
            # Optionally, you could also rotate here or let the dual-fisheye logic handle it.
            pass

        # Check distance and move forward/backward.
        if distance > target_distance + distance_threshold:
            move_distance = min(0.5, distance - target_distance)
            print(f"Person is too far ({distance:.2f}m). Moving forward {move_distance:.2f} m...")
            move_spot_forward(move_distance)
            adjust_spot_posture(height=0.0, pitch=0.0)
        elif distance < target_distance - distance_threshold:
            move_distance = min(0.5, target_distance - distance)
            print(f"Person is too close ({distance:.2f}m). Moving backward {move_distance:.2f} m...")
            move_spot_backward(move_distance)
            adjust_spot_posture(height=0.1, pitch=0.2)
        else:
            print(f"✅ Distance is optimal ({distance:.2f}m). No movement needed.")
            adjust_spot_posture(height=0.0, pitch=0.0)
    else:
        print("⚠️ No valid detection for movement control.")

    time.sleep(1)
